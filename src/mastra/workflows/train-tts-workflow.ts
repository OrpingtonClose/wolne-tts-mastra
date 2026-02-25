import { createWorkflow, createStep } from "@mastra/core/workflows";
import { z } from "zod";

const TRAINING_URL =
  process.env.TRAINING_MANAGER_URL || "http://localhost:5003";

// ---------------------------------------------------------------------------
// Step 1: Fetch qualifying narrators
// ---------------------------------------------------------------------------
const fetchNarratorsStep = createStep({
  id: "fetch-narrators",
  inputSchema: z.object({
    minBooks: z.number(),
  }),
  outputSchema: z.object({
    narrators: z.array(
      z.object({
        narrator: z.string(),
        books: z.number(),
        total_hours: z.number(),
        has_model: z.boolean(),
      })
    ),
    total: z.number(),
    minBooks: z.number(),
  }),
  execute: async (inputData) => {
    const res = await fetch(
      `${TRAINING_URL}/narrators?min_books=${inputData.minBooks}`
    );
    if (!res.ok)
      throw new Error(`Training manager unreachable: ${await res.text()}`);
    const data = (await res.json()) as {
      narrators: Array<{
        narrator: string;
        books: number;
        total_hours: number;
        has_model: boolean;
      }>;
      total: number;
    };
    return { ...data, minBooks: inputData.minBooks };
  },
});

// ---------------------------------------------------------------------------
// Step 2: Queue training for all narrators without a model
// ---------------------------------------------------------------------------
const queueTrainingStep = createStep({
  id: "queue-training",
  inputSchema: z.object({
    narrators: z.array(
      z.object({
        narrator: z.string(),
        books: z.number(),
        total_hours: z.number(),
        has_model: z.boolean(),
      })
    ),
    total: z.number(),
    minBooks: z.number(),
  }),
  outputSchema: z.object({
    queued: z.array(z.object({ job_id: z.string(), narrator: z.string() })),
    skipped: z.array(z.string()),
    totalQueued: z.number(),
    totalSkipped: z.number(),
  }),
  execute: async (inputData) => {
    // Filter out narrators that already have a trained model
    const toTrain = inputData.narrators.filter((n) => !n.has_model);
    const skipped = inputData.narrators
      .filter((n) => n.has_model)
      .map((n) => n.narrator);

    if (toTrain.length === 0) {
      return { queued: [], skipped, totalQueued: 0, totalSkipped: skipped.length };
    }

    // Call train-all endpoint on training manager
    const res = await fetch(`${TRAINING_URL}/jobs/train-all`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        min_books: inputData.minBooks,
        max_books_per_narrator: 0,
        epochs: 15,
        batch_size: 8,
        learning_rate: 1e-5,
      }),
    });
    if (!res.ok)
      throw new Error(`Failed to queue training: ${await res.text()}`);

    const data = (await res.json()) as {
      jobs: Array<{ job_id: string; narrator: string }>;
      total: number;
    };

    return {
      queued: data.jobs,
      skipped,
      totalQueued: data.total,
      totalSkipped: skipped.length,
    };
  },
});

// ---------------------------------------------------------------------------
// Workflow: Train TTS models for all qualifying narrators
// ---------------------------------------------------------------------------
export const trainTtsWorkflow = createWorkflow({
  id: "train-tts-workflow",
  description:
    "Fine-tunes XTTS-v2 for every Wolne Lektury narrator with more than N books. " +
    "Downloads audiobooks, prepares training datasets, and queues sequential GPU training. " +
    "Returns job IDs for monitoring progress.",
  inputSchema: z.object({
    minBooks: z
      .number()
      .default(3)
      .describe(
        "Minimum number of audiobooks a narrator must have to qualify for training"
      ),
  }),
  outputSchema: z.object({
    queued: z.array(z.object({ job_id: z.string(), narrator: z.string() })),
    skipped: z.array(z.string()),
    totalQueued: z.number(),
    totalSkipped: z.number(),
  }),
})
  .then(fetchNarratorsStep)
  .then(queueTrainingStep)
  .commit();
