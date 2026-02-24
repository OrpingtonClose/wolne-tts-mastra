import { createWorkflow, createStep } from "@mastra/core/workflows";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Step 1: Fetch book text from Wolne Lektury
// ---------------------------------------------------------------------------
const fetchBookStep = createStep({
  id: "fetch-book",
  inputSchema: z.object({
    bookSlug: z.string(),
  }),
  outputSchema: z.object({
    title: z.string(),
    author: z.string(),
    text: z.string(),
    textLength: z.number(),
  }),
  execute: async ({ inputData }) => {
    const { bookSlug } = inputData;

    const metaRes = await fetch(
      `https://wolnelektury.pl/api/books/${bookSlug}/?format=json`
    );
    if (!metaRes.ok) {
      throw new Error(`Book '${bookSlug}' not found (HTTP ${metaRes.status})`);
    }

    const meta = (await metaRes.json()) as {
      title: string;
      authors: Array<{ name: string }>;
      txt: string;
    };
    if (!meta.txt) throw new Error(`No text version for '${bookSlug}'`);

    const textRes = await fetch(meta.txt);
    const rawText = await textRes.text();

    // Clean WL headers/footers
    let text = rawText;
    const hdrEnd = text.indexOf("\n\n\n");
    if (hdrEnd !== -1 && hdrEnd < 500) text = text.slice(hdrEnd).trim();
    text = text
      .replace(/\n-{5,}\n[\s\S]*$/, "")
      .replace(/\nTa lektura[\s\S]*$/, "")
      .replace(/\nTen utwór[\s\S]*$/, "")
      .replace(/\n{3,}/g, "\n\n")
      .trim();

    return {
      title: meta.title,
      author: meta.authors?.map((a) => a.name).join(", ") || "Nieznany",
      text,
      textLength: text.length,
    };
  },
});

// ---------------------------------------------------------------------------
// Step 2: Provision Vast.ai GPU VM
// ---------------------------------------------------------------------------
const provisionVmStep = createStep({
  id: "provision-vm",
  inputSchema: z.object({
    title: z.string(),
    author: z.string(),
    text: z.string(),
    textLength: z.number(),
  }),
  outputSchema: z.object({
    title: z.string(),
    author: z.string(),
    text: z.string(),
    textLength: z.number(),
    instanceId: z.number(),
    serverUrl: z.string(),
    gpuName: z.string(),
    pricePerHour: z.number(),
  }),
  execute: async ({ inputData }) => {
    const apiKey = process.env.VASTAI_API_KEY;
    if (!apiKey) throw new Error("VASTAI_API_KEY not set");

    const headers = {
      Accept: "application/json",
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    };

    // Check for existing instance first
    const listRes = await fetch("https://cloud.vast.ai/api/v0/instances/", {
      headers,
    });
    if (listRes.ok) {
      const data = (await listRes.json()) as { instances: Array<any> };
      const existing = data.instances?.find(
        (i: any) =>
          i.label === "wolne-tts-xtts" &&
          i.actual_status === "running" &&
          i.ssh_host
      );
      if (existing) {
        // Determine the public port mapped to 5002
        const portMap = existing.ports || {};
        const ttsPort = portMap["5002/tcp"]?.[0]?.HostPort || 5002;
        return {
          ...inputData,
          instanceId: existing.id,
          serverUrl: `http://${existing.public_ipaddr}:${ttsPort}`,
          gpuName: existing.gpu_name || "unknown",
          pricePerHour: existing.dph_total || 0,
        };
      }
    }

    // Search for an offer
    const searchRes = await fetch("https://cloud.vast.ai/api/v0/bundles/", {
      method: "POST",
      headers,
      body: JSON.stringify({
        q: {
          gpu_ram: { gte: 24 * 1024 },
          dph_total: { lte: 0.50 },
          reliability2: { gte: 0.95 },
          num_gpus: { eq: 1 },
          disk_space: { gte: 80 },
          cuda_max_good: { gte: 12.0 },
          rented: { eq: false },
          type: "on-demand",
        },
        sort: [["dph_total", "asc"]],
        limit: 3,
      }),
    });
    if (!searchRes.ok) {
      throw new Error(`Vast.ai search failed: ${await searchRes.text()}`);
    }
    const offers = (await searchRes.json()) as { offers: Array<any> };
    if (!offers.offers?.length) {
      throw new Error("No suitable Vast.ai GPU offers found");
    }
    const best = offers.offers[0];

    // Create instance with XTTS setup in onstart
    const createRes = await fetch(
      `https://cloud.vast.ai/api/v0/asks/${best.id}/`,
      {
        method: "PUT",
        headers,
        body: JSON.stringify({
          client_id: "me",
          image: "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
          disk: 80,
          label: "wolne-tts-xtts",
          env: { XTTS_PORT: "5002" },
          onstart:
            "apt-get update -qq && apt-get install -y -qq curl wget git ffmpeg sox > /dev/null 2>&1 && " +
            "pip install -q TTS==0.22.0 fastapi uvicorn python-multipart && " +
            "echo 'Deps installed'",
          runtype: "ssh ssh_proxy",
          ports: { "5002/http": {} },
        }),
      }
    );
    if (!createRes.ok) {
      throw new Error(`Instance creation failed: ${await createRes.text()}`);
    }
    const created = (await createRes.json()) as {
      success: boolean;
      new_contract: number;
    };
    if (!created.success) throw new Error("Instance creation unsuccessful");

    const instanceId = created.new_contract;

    // Poll until running
    let info: any = null;
    for (let i = 0; i < 60; i++) {
      await new Promise((r) => setTimeout(r, 5000));
      const res = await fetch(
        `https://cloud.vast.ai/api/v0/instances/${instanceId}/`,
        { headers }
      );
      if (res.ok) {
        info = await res.json();
        if (info.actual_status === "running" && info.public_ipaddr) break;
      }
    }
    if (!info?.public_ipaddr) {
      throw new Error("VM failed to start within 5 minutes");
    }

    const portMap = info.ports || {};
    const ttsPort = portMap["5002/tcp"]?.[0]?.HostPort || 5002;

    return {
      ...inputData,
      instanceId,
      serverUrl: `http://${info.public_ipaddr}:${ttsPort}`,
      gpuName: info.gpu_name || best.gpu_name,
      pricePerHour: info.dph_total || best.dph_total,
    };
  },
});

// ---------------------------------------------------------------------------
// Step 3: Deploy XTTS server and wait for it to be healthy
// ---------------------------------------------------------------------------
const deployXttsStep = createStep({
  id: "deploy-xtts",
  inputSchema: z.object({
    title: z.string(),
    author: z.string(),
    text: z.string(),
    textLength: z.number(),
    instanceId: z.number(),
    serverUrl: z.string(),
    gpuName: z.string(),
    pricePerHour: z.number(),
  }),
  outputSchema: z.object({
    title: z.string(),
    author: z.string(),
    text: z.string(),
    instanceId: z.number(),
    serverUrl: z.string(),
    serverReady: z.boolean(),
  }),
  execute: async ({ inputData }) => {
    const { serverUrl, instanceId } = inputData;

    // Check if server is already running
    try {
      const healthRes = await fetch(`${serverUrl}/health`, {
        signal: AbortSignal.timeout(5000),
      });
      if (healthRes.ok) {
        const h = (await healthRes.json()) as { model_loaded: boolean };
        if (h.model_loaded) {
          return {
            title: inputData.title,
            author: inputData.author,
            text: inputData.text,
            instanceId,
            serverUrl,
            serverReady: true,
          };
        }
      }
    } catch {
      // Server not up yet, need to deploy
    }

    // Deploy XTTS server via Vast.ai execute API
    const apiKey = process.env.VASTAI_API_KEY;
    const headers = {
      Accept: "application/json",
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    };

    // Send the XTTS server setup command
    await fetch(
      `https://cloud.vast.ai/api/v0/instances/${instanceId}/execute/`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({
          command: `bash -c '
            pip install -q TTS==0.22.0 fastapi uvicorn python-multipart 2>/dev/null
            
            # Generate reference audio
            python3 -c "
from TTS.api import TTS
tts = TTS(\"tts_models/multilingual/multi-dataset/xtts_v2\").to(\"cuda\")
tts.tts_to_file(text=\"Witaj, jestem polskim lektorem.\", language=\"pl\", file_path=\"/root/ref_audio.wav\", speaker=\"Claribel Dervla\")
print(\"Ref audio ready\")
" 2>/dev/null

            # Start inference server
            python3 -c "
import io, os, sys, base64, torch, torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS as TTSApi
import uvicorn

app = FastAPI()
tts_api = None
model = None
gpt_cond_latent = None
speaker_embedding = None

@app.on_event(\"startup\")
async def startup():
    global tts_api, model, gpt_cond_latent, speaker_embedding
    tts_api = TTSApi(\"tts_models/multilingual/multi-dataset/xtts_v2\").to(\"cuda\")
    model = tts_api.synthesizer.tts_model
    if os.path.exists(\"/root/ref_audio.wav\"):
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[\"/root/ref_audio.wav\"], gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60
        )

class Req(BaseModel):
    text: str
    language: str = \"pl\"
    speed: float = 1.0
    temperature: float = 0.65
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 5.0
    length_penalty: float = 1.0

@app.get(\"/health\")
async def health():
    return {\"status\": \"ok\", \"model_loaded\": model is not None, \"has_speaker\": gpt_cond_latent is not None}

@app.post(\"/tts\")
async def gen(req: Req):
    if not model or gpt_cond_latent is None:
        raise HTTPException(503, \"Not ready\")
    with torch.no_grad():
        out = model.inference(text=req.text, language=req.language, gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding, temperature=req.temperature, top_p=req.top_p,
            top_k=req.top_k, speed=req.speed, repetition_penalty=req.repetition_penalty,
            length_penalty=req.length_penalty, enable_text_splitting=True)
    wav = torch.tensor(out[\"wav\"]).unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, wav, 24000, format=\"wav\")
    buf.seek(0)
    return {\"audio_base64\": base64.b64encode(buf.read()).decode(), \"sample_rate\": 24000,
        \"duration_seconds\": round(wav.shape[1] / 24000, 2)}

uvicorn.run(app, host=\"0.0.0.0\", port=5002)
" &
          '`,
        }),
      }
    );

    // Wait for the server to become healthy
    for (let i = 0; i < 120; i++) {
      await new Promise((r) => setTimeout(r, 5000));
      try {
        const h = await fetch(`${serverUrl}/health`, {
          signal: AbortSignal.timeout(3000),
        });
        if (h.ok) {
          const data = (await h.json()) as { model_loaded: boolean };
          if (data.model_loaded) {
            return {
              title: inputData.title,
              author: inputData.author,
              text: inputData.text,
              instanceId,
              serverUrl,
              serverReady: true,
            };
          }
        }
      } catch {
        // Not ready yet
      }
    }

    throw new Error("XTTS server failed to become healthy within 10 minutes");
  },
});

// ---------------------------------------------------------------------------
// Step 4: Generate TTS audio from the book text
// ---------------------------------------------------------------------------
const generateTtsStep = createStep({
  id: "generate-tts",
  inputSchema: z.object({
    title: z.string(),
    author: z.string(),
    text: z.string(),
    instanceId: z.number(),
    serverUrl: z.string(),
    serverReady: z.boolean(),
  }),
  outputSchema: z.object({
    title: z.string(),
    author: z.string(),
    audioBase64: z.string(),
    totalChunks: z.number(),
    totalChars: z.number(),
    totalDurationSec: z.number(),
    instanceId: z.number(),
  }),
  execute: async ({ inputData }) => {
    const { serverUrl, text, title, author, instanceId } = inputData;
    const CHUNK_SIZE = 400;

    // Split text into chunks
    const chunks = splitText(text, CHUNK_SIZE);
    const audioBuffers: Buffer[] = [];
    let totalDuration = 0;

    for (let i = 0; i < chunks.length; i++) {
      const res = await fetch(`${serverUrl}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: chunks[i],
          language: "pl",
          speed: 1.0,
          temperature: 0.65,
          top_p: 0.85,
          top_k: 50,
          repetition_penalty: 5.0,
          length_penalty: 1.0,
        }),
      });

      if (!res.ok) {
        throw new Error(`TTS chunk ${i + 1}/${chunks.length} failed: ${await res.text()}`);
      }

      const data = (await res.json()) as {
        audio_base64: string;
        duration_seconds: number;
      };
      audioBuffers.push(Buffer.from(data.audio_base64, "base64"));
      totalDuration += data.duration_seconds;
    }

    const combined = Buffer.concat(audioBuffers);

    return {
      title,
      author,
      audioBase64: combined.toString("base64"),
      totalChunks: chunks.length,
      totalChars: text.length,
      totalDurationSec: Math.round(totalDuration),
      instanceId,
    };
  },
});

function splitText(text: string, max: number): string[] {
  if (text.length <= max) return [text];
  const chunks: string[] = [];
  let rem = text;
  while (rem.length > 0) {
    if (rem.length <= max) {
      chunks.push(rem.trim());
      break;
    }
    const region = rem.slice(0, max);
    let idx = -1;
    for (const e of [". ", "! ", "? ", ".\n", "!\n", "?\n"]) {
      const i = region.lastIndexOf(e);
      if (i > idx) idx = i + e.length;
    }
    if (idx === -1) {
      for (const s of [", ", "; ", "\n"]) {
        const i = region.lastIndexOf(s);
        if (i > idx) idx = i + s.length;
      }
    }
    if (idx === -1) {
      const i = region.lastIndexOf(" ");
      if (i !== -1) idx = i + 1;
    }
    if (idx === -1) idx = max;
    chunks.push(rem.slice(0, idx).trim());
    rem = rem.slice(idx);
  }
  return chunks.filter((c) => c.length > 0);
}

// ---------------------------------------------------------------------------
// Workflow: Book slug -> Polish TTS audio
// ---------------------------------------------------------------------------
export const bookTtsWorkflow = createWorkflow({
  id: "book-tts-workflow",
  description:
    "End-to-end: fetch a Polish book from Wolne Lektury, provision a Vast.ai GPU VM with XTTS-v2, generate TTS audio, return the result.",
  inputSchema: z.object({
    bookSlug: z.string().describe("Book slug from wolnelektury.pl"),
  }),
  outputSchema: z.object({
    title: z.string(),
    author: z.string(),
    audioBase64: z.string(),
    totalChunks: z.number(),
    totalChars: z.number(),
    totalDurationSec: z.number(),
    instanceId: z.number(),
  }),
})
  .then(fetchBookStep)
  .then(provisionVmStep)
  .then(deployXttsStep)
  .then(generateTtsStep)
  .commit();
