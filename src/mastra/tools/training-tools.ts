import { createTool } from "@mastra/core/tools";
import { z } from "zod";

const TRAINING_URL =
  process.env.TRAINING_MANAGER_URL || "http://localhost:5003";

/**
 * List narrators from Wolne Lektury with enough audiobook data for training.
 */
export const listNarratorsTool = createTool({
  id: "list-narrators",
  description:
    "Lists Wolne Lektury narrators who have enough audiobook recordings for TTS training. Returns narrator name, book count, total hours, and whether a trained model exists.",
  inputSchema: z.object({
    minBooks: z
      .number()
      .default(3)
      .describe("Minimum number of books a narrator must have"),
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
  }),
  execute: async ({ inputData }) => {
    const res = await fetch(
      `${TRAINING_URL}/narrators?min_books=${inputData.minBooks}`
    );
    if (!res.ok) throw new Error(`Training manager error: ${await res.text()}`);
    return (await res.json()) as {
      narrators: Array<{
        narrator: string;
        books: number;
        total_hours: number;
        has_model: boolean;
      }>;
      total: number;
    };
  },
});

/**
 * Start training for a single narrator.
 */
export const trainNarratorTool = createTool({
  id: "train-narrator",
  description:
    "Starts XTTS-v2 fine-tuning for a specific Wolne Lektury narrator. Downloads their audiobooks, prepares a training dataset, and runs fine-tuning.",
  inputSchema: z.object({
    narrator: z.string().describe("Exact narrator name from the survey"),
    maxBooks: z.number().default(0).describe("Max books to use (0 = all)"),
    epochs: z.number().default(15),
    batchSize: z.number().default(8),
    learningRate: z.number().default(1e-5),
  }),
  outputSchema: z.object({
    job_id: z.string(),
    narrator: z.string(),
    status: z.string(),
  }),
  execute: async ({ inputData }) => {
    const res = await fetch(`${TRAINING_URL}/jobs/train-narrator`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        narrator: inputData.narrator,
        max_books: inputData.maxBooks,
        epochs: inputData.epochs,
        batch_size: inputData.batchSize,
        learning_rate: inputData.learningRate,
      }),
    });
    if (!res.ok) throw new Error(`Train request failed: ${await res.text()}`);
    return (await res.json()) as {
      job_id: string;
      narrator: string;
      status: string;
    };
  },
});

/**
 * Start batch training for all qualifying narrators.
 */
export const trainAllNarratorsTool = createTool({
  id: "train-all-narrators",
  description:
    "Starts XTTS-v2 fine-tuning for ALL Wolne Lektury narrators that have more than minBooks audiobooks. Jobs are queued and processed sequentially on the GPU.",
  inputSchema: z.object({
    minBooks: z
      .number()
      .default(3)
      .describe("Minimum books threshold for qualifying narrators"),
    maxBooksPerNarrator: z
      .number()
      .default(0)
      .describe("Max books per narrator (0 = all)"),
    epochs: z.number().default(15),
    batchSize: z.number().default(8),
    learningRate: z.number().default(1e-5),
  }),
  outputSchema: z.object({
    jobs: z.array(z.object({ job_id: z.string(), narrator: z.string() })),
    total: z.number(),
  }),
  execute: async ({ inputData }) => {
    const res = await fetch(`${TRAINING_URL}/jobs/train-all`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        min_books: inputData.minBooks,
        max_books_per_narrator: inputData.maxBooksPerNarrator,
        epochs: inputData.epochs,
        batch_size: inputData.batchSize,
        learning_rate: inputData.learningRate,
      }),
    });
    if (!res.ok) throw new Error(`Batch train failed: ${await res.text()}`);
    return (await res.json()) as {
      jobs: Array<{ job_id: string; narrator: string }>;
      total: number;
    };
  },
});

/**
 * Check training job status.
 */
export const trainingStatusTool = createTool({
  id: "training-status",
  description: "Check the status and progress of a training job.",
  inputSchema: z.object({
    jobId: z.string().describe("Job ID returned from train-narrator or train-all"),
  }),
  outputSchema: z.object({
    job_id: z.string(),
    narrator: z.string(),
    phase: z.string(),
    progress: z.string(),
    error: z.string().nullable(),
    books_downloaded: z.number(),
    books_total: z.number(),
    segments_created: z.number(),
    training_epoch: z.number(),
    training_epochs_total: z.number(),
    model_path: z.string().nullable(),
  }),
  execute: async ({ inputData }) => {
    const res = await fetch(`${TRAINING_URL}/jobs/${inputData.jobId}`);
    if (!res.ok) throw new Error(`Status check failed: ${await res.text()}`);
    return (await res.json()) as {
      job_id: string;
      narrator: string;
      phase: string;
      progress: string;
      error: string | null;
      books_downloaded: number;
      books_total: number;
      segments_created: number;
      training_epoch: number;
      training_epochs_total: number;
      model_path: string | null;
    };
  },
});

/**
 * List all training jobs.
 */
export const listJobsTool = createTool({
  id: "list-training-jobs",
  description: "List all training jobs with their current status.",
  inputSchema: z.object({}),
  outputSchema: z.object({
    jobs: z.array(
      z.object({
        job_id: z.string(),
        narrator: z.string(),
        phase: z.string(),
        progress: z.string(),
      })
    ),
    queue: z.array(z.string()),
  }),
  execute: async () => {
    const res = await fetch(`${TRAINING_URL}/jobs`);
    if (!res.ok) throw new Error(`List jobs failed: ${await res.text()}`);
    return (await res.json()) as {
      jobs: Array<{
        job_id: string;
        narrator: string;
        phase: string;
        progress: string;
      }>;
      queue: string[];
    };
  },
});
