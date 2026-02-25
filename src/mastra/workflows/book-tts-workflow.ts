import { createWorkflow, createStep } from "@mastra/core/workflows";
import { z } from "zod";

const XTTS_URL = process.env.XTTS_URL || "http://localhost:5002";

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

    let text = rawText;
    const hdrEnd = text.indexOf("\n\n\n");
    if (hdrEnd !== -1 && hdrEnd < 500) text = text.slice(hdrEnd).trim();
    text = text
      .replace(/\n-{5,}\n[\s\S]*$/, "")
      .replace(/\nTa lektura[\s\S]*$/, "")
      .replace(/\nTen utwór[\s\S]*$/, "")
      .replace(/\nInformacja o utworze[\s\S]*$/, "")
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
// Step 2: Generate TTS audio (XTTS-v2 runs locally on same VM)
// ---------------------------------------------------------------------------
const generateTtsStep = createStep({
  id: "generate-tts",
  inputSchema: z.object({
    title: z.string(),
    author: z.string(),
    text: z.string(),
    textLength: z.number(),
  }),
  outputSchema: z.object({
    title: z.string(),
    author: z.string(),
    audioBase64: z.string(),
    totalChunks: z.number(),
    totalChars: z.number(),
    totalDurationSec: z.number(),
  }),
  execute: async ({ inputData }) => {
    const { text, title, author } = inputData;
    const CHUNK_SIZE = 400;

    // Verify XTTS server is healthy
    const healthRes = await fetch(`${XTTS_URL}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!healthRes.ok) {
      throw new Error(`XTTS server not reachable at ${XTTS_URL}`);
    }
    const health = (await healthRes.json()) as { model_loaded: boolean };
    if (!health.model_loaded) {
      throw new Error("XTTS model not loaded yet — try again in a minute");
    }

    const chunks = splitText(text, CHUNK_SIZE);
    const audioBuffers: Buffer[] = [];
    let totalDuration = 0;

    for (let i = 0; i < chunks.length; i++) {
      const res = await fetch(`${XTTS_URL}/tts`, {
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
        throw new Error(
          `TTS chunk ${i + 1}/${chunks.length} failed: ${await res.text()}`
        );
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
// Workflow: Book slug → Polish TTS audio (2 steps, XTTS runs locally)
// ---------------------------------------------------------------------------
export const bookTtsWorkflow = createWorkflow({
  id: "book-tts-workflow",
  description:
    "Fetch a Polish book from Wolne Lektury and generate TTS audio using the local XTTS-v2 server.",
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
  }),
})
  .then(fetchBookStep)
  .then(generateTtsStep)
  .commit();
