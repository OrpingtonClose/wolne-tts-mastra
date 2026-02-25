import { Agent } from "@mastra/core/agent";
import { wolneLekturyTool } from "../tools/wolne-lektury-tool";
import { xttsInferenceTool } from "../tools/xtts-inference-tool";

export const polishTtsAgent = new Agent({
  id: "polish-tts-agent",
  name: "Polish TTS Agent",
  description:
    "Generates high-quality Polish text-to-speech audio from books in the Wolne Lektury digital library.",
  instructions: `You are a Polish TTS assistant. Generate Polish audiobook narrations from Wolne Lektury books.

1. Fetch the book text with wolne-lektury-tool
2. Send text to the local XTTS-v2 server with xtts-inference
3. Return the audio

Popular books: pan-tadeusz, lalka-tom-pierwszy, studnia-i-wahadlo, quo-vadis-tom-pierwszy`,
  model: "openai/gpt-4.1-mini",
  tools: {
    wolneLekturyTool,
    xttsInferenceTool,
  },
});
