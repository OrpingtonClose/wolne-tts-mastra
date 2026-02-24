import { Mastra } from "@mastra/core/mastra";
import { polishTtsAgent } from "./agents/polish-tts-agent";
import { bookTtsWorkflow } from "./workflows/book-tts-workflow";

export const mastra = new Mastra({
  agents: { polishTtsAgent },
  workflows: { bookTtsWorkflow },
});
