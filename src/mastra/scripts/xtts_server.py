#!/usr/bin/env python3
"""
XTTS-v2 Inference Server for Vast.ai VM.
Exposes a simple HTTP API for Polish text-to-speech.

Deployed by the Mastra workflow via SSH on a Vast.ai GPU instance.
Listens on port 5002.
"""

import io
import os
import sys
import base64
import logging
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, Xtts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xtts-server")

app = FastAPI(title="XTTS-v2 Polish TTS Server")

# Globals populated on startup
model = None
config = None
gpt_cond_latent = None
speaker_embedding = None

# Default reference audio path (set via env or will be generated)
REF_AUDIO_PATH = os.environ.get("XTTS_REF_AUDIO", "/root/ref_audio.wav")
MODEL_PATH = os.environ.get("XTTS_MODEL_PATH", "")  # Empty = use base model


class TTSRequest(BaseModel):
    text: str
    language: str = "pl"
    temperature: float = 0.65
    top_p: float = 0.85
    top_k: int = 50
    speed: float = 1.0
    repetition_penalty: float = 5.0
    length_penalty: float = 1.0


class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_seconds: float


@app.on_event("startup")
async def load_model():
    global model, config, gpt_cond_latent, speaker_embedding

    logger.info("Loading XTTS-v2 model...")

    if MODEL_PATH and Path(MODEL_PATH).exists():
        logger.info(f"Loading fine-tuned model from {MODEL_PATH}")
        config = XttsConfig()
        config.load_json(str(Path(MODEL_PATH) / "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=MODEL_PATH,
            use_deepspeed=False,
        )
    else:
        logger.info("Loading base XTTS-v2 model from TTS library...")
        from TTS.api import TTS as TTSApi

        tts_api = TTSApi("tts_models/multilingual/multi-dataset/xtts_v2")
        model = tts_api.synthesizer.tts_model
        config = tts_api.synthesizer.tts_config

    model.cuda()
    model.eval()

    # Compute speaker embedding from reference audio
    if Path(REF_AUDIO_PATH).exists():
        logger.info(f"Computing speaker embedding from {REF_AUDIO_PATH}")
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[REF_AUDIO_PATH],
            gpt_cond_len=30,
            gpt_cond_chunk_len=4,
            max_ref_length=60,
        )
    else:
        logger.warning(
            f"No reference audio at {REF_AUDIO_PATH}. "
            "Will generate one from a Polish sentence on first request."
        )

    logger.info("XTTS-v2 model loaded and ready!")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "has_speaker": gpt_cond_latent is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }


@app.post("/tts", response_model=TTSResponse)
async def generate_tts(req: TTSRequest):
    global gpt_cond_latent, speaker_embedding

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if gpt_cond_latent is None:
        raise HTTPException(
            status_code=503,
            detail="No speaker embedding. Upload reference audio to /upload_ref first.",
        )

    try:
        with torch.no_grad():
            output = model.inference(
                text=req.text,
                language=req.language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                speed=req.speed,
                repetition_penalty=req.repetition_penalty,
                length_penalty=req.length_penalty,
                enable_text_splitting=True,
            )

        wav = torch.tensor(output["wav"]).unsqueeze(0)
        sample_rate = 24000

        # Convert to MP3 via torchaudio
        buf = io.BytesIO()
        torchaudio.save(buf, wav, sample_rate, format="wav")
        buf.seek(0)

        audio_b64 = base64.b64encode(buf.read()).decode("utf-8")
        duration = wav.shape[1] / sample_rate

        return TTSResponse(
            audio_base64=audio_b64,
            sample_rate=sample_rate,
            duration_seconds=round(duration, 2),
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_ref")
async def upload_reference(audio_base64: str):
    """Upload a reference audio clip (base64 WAV) for voice cloning."""
    global gpt_cond_latent, speaker_embedding

    try:
        audio_bytes = base64.b64decode(audio_base64)
        with open(REF_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[REF_AUDIO_PATH],
            gpt_cond_len=30,
            gpt_cond_chunk_len=4,
            max_ref_length=60,
        )
        return {"status": "ok", "ref_audio_path": REF_AUDIO_PATH}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5002)
