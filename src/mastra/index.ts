import { Mastra } from "@mastra/core/mastra";
import { polishTtsAgent } from "./agents/polish-tts-agent";
import { bookTtsWorkflow } from "./workflows/book-tts-workflow";
import { trainTtsWorkflow } from "./workflows/train-tts-workflow";

export const mastra = new Mastra({
  agents: { polishTtsAgent },
  workflows: { bookTtsWorkflow, trainTtsWorkflow },
  server: {
    host: "0.0.0.0",
    port: 4111,
  },
});
