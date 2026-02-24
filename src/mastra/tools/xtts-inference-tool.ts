import { createTool } from "@mastra/core/tools";
import { z } from "zod";

const TTS_CHUNK_MAX = 400; // XTTS works best with shorter chunks (~250-500 chars)

/**
 * Sends text to an XTTS-v2 inference server running on a Vast.ai VM.
 * Handles chunking and sequential generation.
 */
export const xttsInferenceTool = createTool({
  id: "xtts-inference",
  description:
    "Generates Polish speech audio from text using an XTTS-v2 server on a Vast.ai VM. Provide the server URL and text.",
  inputSchema: z.object({
    serverUrl: z
      .string()
      .describe("Base URL of the XTTS server, e.g. http://host:5002"),
    text: z.string().describe("Polish text to convert to speech"),
    language: z.string().default("pl"),
    speed: z.number().default(1.0),
  }),
  outputSchema: z.object({
    audioBase64: z.string().describe("Base64-encoded WAV audio"),
    chunks: z.number(),
    totalChars: z.number(),
    totalDurationSec: z.number(),
  }),
  execute: async ({ inputData }) => {
    const { serverUrl, text, language, speed } = inputData;

    // 1. Check server health
    const healthRes = await fetch(`${serverUrl}/health`);
    if (!healthRes.ok) {
      throw new Error(`XTTS server not healthy: ${healthRes.status}`);
    }
    const health = (await healthRes.json()) as {
      status: string;
      model_loaded: boolean;
      has_speaker: boolean;
    };
    if (!health.model_loaded) {
      throw new Error("XTTS model not loaded on server");
    }
    if (!health.has_speaker) {
      throw new Error("No speaker embedding loaded. Upload reference audio first.");
    }

    // 2. Split text into chunks suitable for XTTS
    const chunks = splitForXtts(text, TTS_CHUNK_MAX);

    // 3. Generate audio for each chunk sequentially (XTTS uses GPU, one at a time)
    const audioBuffers: Buffer[] = [];
    let totalDuration = 0;

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const ttsRes = await fetch(`${serverUrl}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: chunk,
          language,
          speed,
          temperature: 0.65,
          top_p: 0.85,
          top_k: 50,
          repetition_penalty: 5.0,
          length_penalty: 1.0,
        }),
      });

      if (!ttsRes.ok) {
        const err = await ttsRes.text();
        throw new Error(`TTS failed on chunk ${i + 1}/${chunks.length}: ${err}`);
      }

      const result = (await ttsRes.json()) as {
        audio_base64: string;
        sample_rate: number;
        duration_seconds: number;
      };

      audioBuffers.push(Buffer.from(result.audio_base64, "base64"));
      totalDuration += result.duration_seconds;
    }

    // 4. Concatenate WAV buffers (simple concatenation — all same sample rate)
    const combined = Buffer.concat(audioBuffers);

    return {
      audioBase64: combined.toString("base64"),
      chunks: chunks.length,
      totalChars: text.length,
      totalDurationSec: Math.round(totalDuration),
    };
  },
});

/**
 * Split text into chunks for XTTS-v2.
 * XTTS handles shorter chunks better — split at sentence boundaries.
 */
function splitForXtts(text: string, maxChars: number): string[] {
  if (text.length <= maxChars) return [text];

  const chunks: string[] = [];
  let remaining = text;

  while (remaining.length > 0) {
    if (remaining.length <= maxChars) {
      chunks.push(remaining.trim());
      break;
    }

    const region = remaining.slice(0, maxChars);
    let splitIdx = -1;

    // Prefer sentence boundaries
    for (const end of [". ", "! ", "? ", ".\n", "!\n", "?\n"]) {
      const idx = region.lastIndexOf(end);
      if (idx > splitIdx) splitIdx = idx + end.length;
    }
    // Fallback to comma or semicolon
    if (splitIdx === -1) {
      for (const sep of [", ", "; "]) {
        const idx = region.lastIndexOf(sep);
        if (idx > splitIdx) splitIdx = idx + sep.length;
      }
    }
    // Fallback to newline
    if (splitIdx === -1) {
      const idx = region.lastIndexOf("\n");
      if (idx !== -1) splitIdx = idx + 1;
    }
    // Last resort: space
    if (splitIdx === -1) {
      const idx = region.lastIndexOf(" ");
      if (idx !== -1) splitIdx = idx + 1;
    }
    if (splitIdx === -1) splitIdx = maxChars;

    chunks.push(remaining.slice(0, splitIdx).trim());
    remaining = remaining.slice(splitIdx);
  }

  return chunks.filter((c) => c.length > 0);
}
