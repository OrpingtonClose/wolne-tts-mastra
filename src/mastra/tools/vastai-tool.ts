import { createTool } from "@mastra/core/tools";
import { z } from "zod";

const VAST_API_BASE = "https://cloud.vast.ai/api/v0";

function vastHeaders(): Record<string, string> {
  const key = process.env.VASTAI_API_KEY;
  if (!key) throw new Error("VASTAI_API_KEY env var not set");
  return {
    Accept: "application/json",
    "Content-Type": "application/json",
    Authorization: `Bearer ${key}`,
  };
}

// ---------------------------------------------------------------------------
// Tool: Provision a Vast.ai GPU instance for XTTS-v2
// ---------------------------------------------------------------------------
export const vastaiProvisionTool = createTool({
  id: "vastai-provision",
  description:
    "Provisions a Vast.ai GPU VM for XTTS-v2 TTS inference. Searches for a cheap 24GB+ GPU, creates the instance, and returns connection details.",
  inputSchema: z.object({
    minVram: z.number().default(24).describe("Minimum GPU VRAM in GB"),
    maxPrice: z.number().default(0.50).describe("Max $/hr on-demand price"),
    diskGb: z.number().default(80).describe("Disk space in GB"),
  }),
  outputSchema: z.object({
    instanceId: z.number(),
    status: z.string(),
    sshHost: z.string(),
    sshPort: z.number(),
    gpuName: z.string(),
    pricePerHour: z.number(),
  }),
  execute: async (inputData) => {
    const { minVram, maxPrice, diskGb } = inputData;

    // 1. Check for existing running instances tagged for our use
    const existing = await findExistingInstance();
    if (existing) return existing;

    // 2. Search for offers
    const searchRes = await fetch(`${VAST_API_BASE}/bundles/`, {
      method: "POST",
      headers: vastHeaders(),
      body: JSON.stringify({
        q: {
          gpu_ram: { gte: minVram * 1024 }, // API uses MB
          dph_total: { lte: maxPrice },
          reliability2: { gte: 0.95 },
          num_gpus: { eq: 1 },
          disk_space: { gte: diskGb },
          inet_up: { gte: 100 },
          cuda_max_good: { gte: 12.0 },
          rented: { eq: false },
          type: "on-demand",
        },
        sort: [["dph_total", "asc"]],
        limit: 5,
      }),
    });

    if (!searchRes.ok) {
      throw new Error(`Vast.ai search failed: ${searchRes.status} ${await searchRes.text()}`);
    }

    const offers = (await searchRes.json()) as { offers: Array<{
      id: number;
      gpu_name: string;
      gpu_ram: number;
      dph_total: number;
      disk_space: number;
    }> };

    if (!offers.offers || offers.offers.length === 0) {
      throw new Error(
        `No Vast.ai offers found matching: ${minVram}GB VRAM, <$${maxPrice}/hr`
      );
    }

    const best = offers.offers[0];

    // 3. Create instance
    const createRes = await fetch(`${VAST_API_BASE}/asks/${best.id}/`, {
      method: "PUT",
      headers: vastHeaders(),
      body: JSON.stringify({
        client_id: "me",
        image: "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
        disk: diskGb,
        label: "wolne-tts-xtts",
        onstart: "apt-get update -qq && apt-get install -y -qq curl wget git ffmpeg sox",
        runtype: "ssh ssh_proxy",
      }),
    });

    if (!createRes.ok) {
      throw new Error(`Failed to create instance: ${await createRes.text()}`);
    }

    const created = (await createRes.json()) as {
      success: boolean;
      new_contract: number;
    };

    if (!created.success) {
      throw new Error("Vast.ai instance creation failed");
    }

    const instanceId = created.new_contract;

    // 4. Wait for instance to be running (poll up to 3 min)
    let instanceInfo: any = null;
    for (let i = 0; i < 36; i++) {
      await sleep(5000);
      const info = await getInstanceInfo(instanceId);
      if (info && info.actual_status === "running" && info.ssh_host) {
        instanceInfo = info;
        break;
      }
    }

    if (!instanceInfo) {
      throw new Error(`Instance ${instanceId} failed to start within 3 minutes`);
    }

    return {
      instanceId,
      status: instanceInfo.actual_status,
      sshHost: instanceInfo.ssh_host,
      sshPort: instanceInfo.ssh_port,
      gpuName: instanceInfo.gpu_name || best.gpu_name,
      pricePerHour: instanceInfo.dph_total || best.dph_total,
    };
  },
});

// ---------------------------------------------------------------------------
// Tool: Check instance status
// ---------------------------------------------------------------------------
export const vastaiStatusTool = createTool({
  id: "vastai-status",
  description: "Check status of a Vast.ai instance",
  inputSchema: z.object({
    instanceId: z.number(),
  }),
  outputSchema: z.object({
    instanceId: z.number(),
    status: z.string(),
    sshHost: z.string(),
    sshPort: z.number(),
    gpuName: z.string(),
    uptimeHours: z.number(),
  }),
  execute: async (inputData) => {
    const info = await getInstanceInfo(inputData.instanceId);
    if (!info) throw new Error(`Instance ${inputData.instanceId} not found`);
    return {
      instanceId: inputData.instanceId,
      status: info.actual_status || "unknown",
      sshHost: info.ssh_host || "",
      sshPort: info.ssh_port || 0,
      gpuName: info.gpu_name || "",
      uptimeHours: info.duration ? info.duration / 3600 : 0,
    };
  },
});

// ---------------------------------------------------------------------------
// Tool: Destroy a Vast.ai instance
// ---------------------------------------------------------------------------
export const vastaiDestroyTool = createTool({
  id: "vastai-destroy",
  description: "Destroys a Vast.ai instance to stop billing",
  inputSchema: z.object({
    instanceId: z.number(),
  }),
  outputSchema: z.object({
    success: z.boolean(),
    instanceId: z.number(),
  }),
  execute: async (inputData) => {
    const res = await fetch(
      `${VAST_API_BASE}/instances/${inputData.instanceId}/`,
      {
        method: "DELETE",
        headers: vastHeaders(),
      }
    );
    return {
      success: res.ok,
      instanceId: inputData.instanceId,
    };
  },
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
async function findExistingInstance(): Promise<{
  instanceId: number;
  status: string;
  sshHost: string;
  sshPort: number;
  gpuName: string;
  pricePerHour: number;
} | null> {
  const res = await fetch(`${VAST_API_BASE}/instances/`, {
    headers: vastHeaders(),
  });
  if (!res.ok) return null;

  const data = (await res.json()) as { instances: Array<any> };
  const running = data.instances?.find(
    (i: any) =>
      i.label === "wolne-tts-xtts" &&
      i.actual_status === "running" &&
      i.ssh_host
  );

  if (!running) return null;

  return {
    instanceId: running.id,
    status: running.actual_status,
    sshHost: running.ssh_host,
    sshPort: running.ssh_port,
    gpuName: running.gpu_name || "",
    pricePerHour: running.dph_total || 0,
  };
}

async function getInstanceInfo(id: number): Promise<any | null> {
  const res = await fetch(`${VAST_API_BASE}/instances/${id}/`, {
    headers: vastHeaders(),
  });
  if (!res.ok) return null;
  return res.json();
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
