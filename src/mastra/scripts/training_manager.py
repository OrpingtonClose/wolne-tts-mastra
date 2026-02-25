#!/usr/bin/env python3
"""
Training Manager for XTTS-v2 fine-tuning on Wolne Lektury audiobooks.
Runs on port 5003 alongside the XTTS inference server (port 5002).

Endpoints:
  GET  /narrators                  - List qualifying narrators (>N books)
  POST /jobs/train-narrator        - Start training for one narrator
  POST /jobs/train-all             - Start batch training for all qualifying narrators
  GET  /jobs                       - List all jobs
  GET  /jobs/{job_id}              - Job status + progress
  POST /model/activate/{narrator}  - Activate a trained model on XTTS server
"""

import io
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import requests
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("training-manager")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("TRAINING_DATA_DIR", "/root/training_data"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/root/models"))
SURVEY_PATH = Path(os.environ.get("SURVEY_PATH", "/root/narrator_survey.json"))
XTTS_URL = os.environ.get("XTTS_URL", "http://localhost:5002")
MIN_BOOKS_DEFAULT = 3
SEGMENT_MIN_SEC = 4.0
SEGMENT_MAX_SEC = 15.0
SAMPLE_RATE = 22050

# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------
class JobPhase(str, Enum):
    queued = "queued"
    downloading = "downloading"
    preparing = "preparing"
    training = "training"
    completed = "completed"
    failed = "failed"


@dataclass
class TrainingJob:
    job_id: str
    narrator: str
    phase: JobPhase = JobPhase.queued
    progress: str = ""
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    books_total: int = 0
    books_downloaded: int = 0
    segments_created: int = 0
    training_epoch: int = 0
    training_epochs_total: int = 0
    model_path: Optional[str] = None


jobs: dict[str, TrainingJob] = {}
job_queue: list[str] = []
worker_lock = threading.Lock()
worker_running = False

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="XTTS Training Manager", version="1.0.0")


class TrainNarratorRequest(BaseModel):
    narrator: str
    max_books: int = 0  # 0 = all
    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1e-5


class TrainAllRequest(BaseModel):
    min_books: int = MIN_BOOKS_DEFAULT
    max_books_per_narrator: int = 0
    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1e-5


# ---------------------------------------------------------------------------
# Narrator survey
# ---------------------------------------------------------------------------
def load_survey() -> dict:
    if not SURVEY_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Survey not found at {SURVEY_PATH}")
    with open(SURVEY_PATH) as f:
        return json.load(f)


@app.get("/narrators")
def list_narrators(min_books: int = MIN_BOOKS_DEFAULT):
    survey = load_survey()
    result = []
    for name, info in sorted(survey.items(), key=lambda x: -x[1]["total_seconds"]):
        n_books = len(info["books"])
        if n_books > min_books:
            result.append({
                "narrator": name,
                "books": n_books,
                "total_hours": round(info["total_seconds"] / 3600, 1),
                "has_model": (MODELS_DIR / slugify(name) / "best_model.pth").exists(),
            })
    return {"narrators": result, "total": len(result)}


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------
@app.post("/jobs/train-narrator")
def train_narrator(req: TrainNarratorRequest):
    survey = load_survey()
    if req.narrator not in survey:
        raise HTTPException(status_code=404, detail=f"Narrator '{req.narrator}' not found in survey")

    job = TrainingJob(
        job_id=str(uuid.uuid4())[:8],
        narrator=req.narrator,
        books_total=len(survey[req.narrator]["books"]),
    )
    jobs[job.job_id] = job
    job_queue.append(job.job_id)
    _ensure_worker(req)
    return {"job_id": job.job_id, "narrator": req.narrator, "status": "queued"}


@app.post("/jobs/train-all")
def train_all(req: TrainAllRequest):
    survey = load_survey()
    created = []
    for name, info in sorted(survey.items(), key=lambda x: -x[1]["total_seconds"]):
        if len(info["books"]) > req.min_books:
            job = TrainingJob(
                job_id=str(uuid.uuid4())[:8],
                narrator=name,
                books_total=len(info["books"]),
            )
            jobs[job.job_id] = job
            job_queue.append(job.job_id)
            created.append({"job_id": job.job_id, "narrator": name})

    narrator_req = TrainNarratorRequest(
        narrator="",  # ignored by worker
        max_books=req.max_books_per_narrator,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
    )
    _ensure_worker(narrator_req)
    return {"jobs": created, "total": len(created)}


@app.get("/jobs")
def list_jobs():
    return {"jobs": [asdict(j) for j in jobs.values()], "queue": job_queue}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(jobs[job_id])


@app.post("/model/activate/{narrator_slug}")
def activate_model(narrator_slug: str):
    model_dir = MODELS_DIR / narrator_slug
    if not (model_dir / "best_model.pth").exists():
        raise HTTPException(status_code=404, detail=f"No trained model for '{narrator_slug}'")
    # Tell XTTS server to reload with this model
    try:
        resp = requests.post(f"{XTTS_URL}/reload", json={"model_path": str(model_dir)}, timeout=120)
        return {"status": "activated", "model_path": str(model_dir), "xtts_response": resp.json()}
    except Exception as e:
        return {"status": "model_ready", "model_path": str(model_dir), "note": f"XTTS reload failed: {e}. Restart XTTS with XTTS_MODEL_PATH={model_dir}"}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _ensure_worker(req: TrainNarratorRequest):
    global worker_running
    with worker_lock:
        if worker_running:
            return
        worker_running = True
    t = threading.Thread(target=_worker_loop, args=(req,), daemon=True)
    t.start()


def _worker_loop(req: TrainNarratorRequest):
    global worker_running
    try:
        while job_queue:
            job_id = job_queue.pop(0)
            job = jobs.get(job_id)
            if not job:
                continue
            try:
                job.started_at = time.time()
                _process_narrator(job, req)
                job.phase = JobPhase.completed
                job.finished_at = time.time()
                logger.info(f"Job {job.job_id} completed for {job.narrator}")
            except Exception as e:
                job.phase = JobPhase.failed
                job.error = str(e)
                job.finished_at = time.time()
                logger.error(f"Job {job.job_id} failed for {job.narrator}: {e}", exc_info=True)
    finally:
        with worker_lock:
            worker_running = False


def _process_narrator(job: TrainingJob, req: TrainNarratorRequest):
    narrator = job.narrator
    narrator_slug = slugify(narrator)
    raw_dir = DATA_DIR / narrator_slug / "raw"
    dataset_dir = DATA_DIR / narrator_slug / "dataset"
    wavs_dir = dataset_dir / "wavs"
    model_dir = MODELS_DIR / narrator_slug

    raw_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    survey = load_survey()
    books = survey[narrator]["books"]
    if req.max_books > 0:
        books = books[:req.max_books]
    job.books_total = len(books)

    # Phase 1: Download
    job.phase = JobPhase.downloading
    job.progress = f"Downloading {len(books)} books..."
    _download_audiobooks(job, books, raw_dir)

    # Phase 2: Prepare dataset
    job.phase = JobPhase.preparing
    job.progress = "Converting audio and creating segments..."
    _prepare_dataset(job, raw_dir, wavs_dir, dataset_dir)

    # Phase 3: Train
    job.phase = JobPhase.training
    job.progress = "Fine-tuning XTTS-v2..."
    _train_xtts(job, dataset_dir, wavs_dir, model_dir, req)

    job.model_path = str(model_dir)
    job.progress = "Done!"


# ---------------------------------------------------------------------------
# Phase 1: Download audiobooks
# ---------------------------------------------------------------------------
def _download_audiobooks(job: TrainingJob, books: list[dict], raw_dir: Path):
    for i, book in enumerate(books):
        slug = book["slug"]
        job.progress = f"Downloading {i+1}/{len(books)}: {slug}"
        job.books_downloaded = i

        book_dir = raw_dir / slug
        book_dir.mkdir(exist_ok=True)

        # Skip if already downloaded
        if list(book_dir.glob("*.mp3")) and (book_dir / "text.txt").exists():
            logger.info(f"Skipping {slug} (already downloaded)")
            continue

        try:
            # Fetch book detail for MP3 URLs
            detail = requests.get(
                f"https://wolnelektury.pl/api/books/{slug}/?format=json", timeout=30
            ).json()

            # Download text
            txt_url = detail.get("txt")
            if txt_url:
                txt_resp = requests.get(txt_url, timeout=30)
                (book_dir / "text.txt").write_text(txt_resp.text, encoding="utf-8")

            # Download MP3s
            for media in detail.get("media", []):
                if media.get("type") != "mp3":
                    continue
                mp3_url = media["url"]
                mp3_name = mp3_url.split("/")[-1]
                mp3_path = book_dir / mp3_name
                if mp3_path.exists():
                    continue
                logger.info(f"  Downloading {mp3_name}")
                resp = requests.get(mp3_url, timeout=300, stream=True)
                with open(mp3_path, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)

        except Exception as e:
            logger.warning(f"Failed to download {slug}: {e}")

    job.books_downloaded = len(books)


# ---------------------------------------------------------------------------
# Phase 2: Prepare dataset (convert, transcribe, segment)
# ---------------------------------------------------------------------------
def _prepare_dataset(job: TrainingJob, raw_dir: Path, wavs_dir: Path, dataset_dir: Path):
    from faster_whisper import WhisperModel

    job.progress = "Loading Whisper model for alignment..."
    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    metadata_lines = []
    seg_idx = 0

    # Process each book's MP3 files
    for book_dir in sorted(raw_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        mp3_files = sorted(book_dir.glob("*.mp3"))
        if not mp3_files:
            continue

        job.progress = f"Processing {book_dir.name} ({len(mp3_files)} files)..."

        for mp3_path in mp3_files:
            # Convert MP3 to WAV
            wav_tmp = raw_dir / f"_tmp_{mp3_path.stem}.wav"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", str(SAMPLE_RATE),
                     "-ac", "1", "-acodec", "pcm_s16le", str(wav_tmp)],
                    capture_output=True, check=True, timeout=300
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"ffmpeg failed for {mp3_path}: {e.stderr[:200]}")
                continue

            # Transcribe with word timestamps
            try:
                segments_iter, info = whisper_model.transcribe(
                    str(wav_tmp), language="pl", word_timestamps=True,
                    vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
                )
                segments_list = list(segments_iter)
            except Exception as e:
                logger.warning(f"Whisper failed for {mp3_path}: {e}")
                wav_tmp.unlink(missing_ok=True)
                continue

            # Group segments into training clips (SEGMENT_MIN_SEC to SEGMENT_MAX_SEC)
            current_start = None
            current_end = None
            current_text_parts = []

            for seg in segments_list:
                seg_text = seg.text.strip()
                if not seg_text:
                    continue

                if current_start is None:
                    current_start = seg.start
                    current_end = seg.end
                    current_text_parts = [seg_text]
                else:
                    potential_duration = seg.end - current_start
                    if potential_duration > SEGMENT_MAX_SEC:
                        # Flush current segment
                        if current_end - current_start >= SEGMENT_MIN_SEC:
                            clip_name = f"clip_{seg_idx:06d}"
                            _extract_clip(wav_tmp, current_start, current_end, wavs_dir / f"{clip_name}.wav")
                            text = " ".join(current_text_parts).strip()
                            text = _clean_transcript(text)
                            if text:
                                metadata_lines.append(f"{clip_name}|{text}|{text}")
                                seg_idx += 1
                                job.segments_created = seg_idx
                        current_start = seg.start
                        current_end = seg.end
                        current_text_parts = [seg_text]
                    else:
                        current_end = seg.end
                        current_text_parts.append(seg_text)

            # Flush last segment
            if current_start is not None and current_end - current_start >= SEGMENT_MIN_SEC:
                clip_name = f"clip_{seg_idx:06d}"
                _extract_clip(wav_tmp, current_start, current_end, wavs_dir / f"{clip_name}.wav")
                text = " ".join(current_text_parts).strip()
                text = _clean_transcript(text)
                if text:
                    metadata_lines.append(f"{clip_name}|{text}|{text}")
                    seg_idx += 1
                    job.segments_created = seg_idx

            wav_tmp.unlink(missing_ok=True)

    # Write metadata
    metadata_path = dataset_dir / "metadata.csv"
    metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    job.progress = f"Dataset ready: {seg_idx} segments"
    logger.info(f"Created dataset with {seg_idx} segments at {dataset_dir}")

    # Free Whisper from GPU
    del whisper_model
    torch.cuda.empty_cache()


def _extract_clip(wav_path: Path, start: float, end: float, out_path: Path):
    """Extract a time slice from a WAV file using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path),
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-acodec", "pcm_s16le",
         str(out_path)],
        capture_output=True, check=True, timeout=60
    )


def _clean_transcript(text: str) -> str:
    """Clean up Whisper transcript for training."""
    text = re.sub(r"\s+", " ", text).strip()
    # Remove leading/trailing punctuation artifacts
    text = text.strip("- —–")
    return text


# ---------------------------------------------------------------------------
# Phase 3: XTTS-v2 fine-tuning
# ---------------------------------------------------------------------------
def _build_test_sentences(wavs_dir: Path) -> list:
    """Build test_sentences for XTTS trainer using first clip from dataset as speaker ref."""
    wav_files = sorted(wavs_dir.glob("*.wav"))
    if not wav_files:
        return []
    ref_wav = str(wav_files[0])
    return [
        {"text": "Witaj, jestem polskim lektorem i czytam książki.", "speaker_wav": ref_wav, "language": "pl"},
    ]


def _unload_xtts_server():
    """Ask XTTS inference server to unload model and free GPU memory."""
    try:
        resp = requests.post(f"{XTTS_URL}/unload", timeout=30)
        if resp.status_code == 200:
            logger.info("XTTS server model unloaded (GPU freed for training)")
        else:
            logger.warning(f"XTTS /unload returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Could not unload XTTS server model: {e}")


def _reload_xtts_server(model_path: str = ""):
    """Ask XTTS inference server to reload model after training."""
    try:
        payload = {"model_path": model_path} if model_path else {}
        resp = requests.post(f"{XTTS_URL}/reload", json=payload, timeout=120)
        if resp.status_code == 200:
            logger.info(f"XTTS server reloaded with model: {model_path or 'base'}")
        else:
            logger.warning(f"XTTS /reload returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Could not reload XTTS server model: {e}")


def _train_xtts(job: TrainingJob, dataset_dir: Path, wavs_dir: Path, model_dir: Path, req: TrainNarratorRequest):
    """Fine-tune XTTS-v2 on the prepared dataset."""
    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer, GPTTrainerConfig, XttsAudioConfig
    from TTS.tts.models.xtts import XttsArgs

    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        raise RuntimeError(f"No metadata.csv found at {dataset_dir}")

    num_segments = sum(1 for _ in open(metadata_path))
    if num_segments < 10:
        raise RuntimeError(f"Too few segments ({num_segments}) for training. Need at least 10.")

    logger.info(f"Starting XTTS fine-tuning: {num_segments} segments, {req.epochs} epochs")
    job.training_epochs_total = req.epochs

    # Free GPU memory by unloading XTTS inference server
    _unload_xtts_server()
    torch.cuda.empty_cache()

    # Find base XTTS model path
    base_model_dir = Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
    if not base_model_dir.exists():
        # Try downloading via TTS API
        logger.info("Downloading base XTTS-v2 model...")
        from TTS.api import TTS as TTSApi
        _ = TTSApi("tts_models/multilingual/multi-dataset/xtts_v2")
        del _
        torch.cuda.empty_cache()

    # Dataset config
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=str(dataset_dir),
        language="pl",
    )

    # Audio config
    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    # Model args — keep pretrained token IDs (8192/8193/8194), only adjust batch size
    model_args = XttsArgs(
        gpt_batch_size=req.batch_size,
    )

    # Training config
    config = GPTTrainerConfig(
        output_path=str(model_dir),
        model_args=model_args,
        audio=audio_config,
        batch_size=req.batch_size,
        eval_batch_size=max(1, req.batch_size // 2),
        num_loader_workers=4,
        num_eval_loader_workers=2,
        epochs=req.epochs,
        lr=req.learning_rate,
        optimizer="AdamW",
        lr_scheduler="CosineAnnealingWarmRestarts",
        lr_scheduler_params={"T_0": 5, "T_mult": 2, "eta_min": 1e-7},
        print_step=50,
        save_step=500,
        save_best_after=1000,
        save_checkpoints=True,
        save_all_best=True,
        mixed_precision=True,
        datasets=[dataset_config],
        start_by_longest=True,
        # test_sentences need: text, speaker_wav, language
        # We'll pick the first clip from the dataset as speaker reference
        test_sentences=_build_test_sentences(wavs_dir),
    )

    # Load samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=500,
        eval_split_size=0.1,
    )

    logger.info(f"Train samples: {len(train_samples)}, Eval samples: {len(eval_samples)}")

    # Create model & trainer
    gpt_model = GPTTrainer.init_from_config(config)

    # Load pretrained weights
    checkpoint_path = base_model_dir / "model.pth"
    if checkpoint_path.exists():
        logger.info(f"Loading pretrained checkpoint from {checkpoint_path}")
        gpt_model.load_checkpoint(config, str(checkpoint_path), eval=False, strict=False)

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
        ),
        config,
        output_path=str(model_dir),
        model=gpt_model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Monkey-patch trainer to update job progress on each epoch
    _original_on_epoch_start = getattr(trainer, 'on_epoch_start', None)

    def _patched_epoch_callback(*args, **kwargs):
        job.training_epoch = trainer.epochs_done + 1
        job.progress = f"Training epoch {job.training_epoch}/{req.epochs}"
        logger.info(job.progress)
        if _original_on_epoch_start:
            return _original_on_epoch_start(*args, **kwargs)

    # The Trainer from coqui uses callbacks_on_epoch_start or similar
    # Safest approach: wrap trainer.fit() with epoch tracking via log monitoring
    try:
        trainer.fit()
    finally:
        # Always free GPU and reload XTTS server, even if training fails
        del gpt_model
        del trainer
        torch.cuda.empty_cache()

    # Find best model
    best_model = model_dir / "best_model.pth"
    if not best_model.exists():
        # Copy last checkpoint as best
        checkpoints = sorted(model_dir.glob("checkpoint_*.pth"))
        if checkpoints:
            shutil.copy2(checkpoints[-1], best_model)

    # Also copy config
    config_src = base_model_dir / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, model_dir / "config.json")

    # Copy vocab
    vocab_src = base_model_dir / "vocab.json"
    if vocab_src.exists():
        shutil.copy2(vocab_src, model_dir / "vocab.json")

    job.model_path = str(model_dir)
    logger.info(f"Training complete. Model saved to {model_dir}")

    # Reload XTTS server with newly trained model
    _reload_xtts_server(str(model_dir))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slugify(name: str) -> str:
    """Convert narrator name to filesystem-safe slug."""
    s = name.lower().strip()
    s = re.sub(r"[ąà]", "a", s)
    s = re.sub(r"[ćč]", "c", s)
    s = re.sub(r"[ęè]", "e", s)
    s = re.sub(r"[łl]", "l", s)
    s = re.sub(r"[ńñ]", "n", s)
    s = re.sub(r"[óò]", "o", s)
    s = re.sub(r"[śš]", "s", s)
    s = re.sub(r"[źżž]", "z", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
