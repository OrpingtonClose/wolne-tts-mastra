import { createTool } from "@mastra/core/tools";
import { z } from "zod";

/**
 * Fetches a book's text content from the Wolne Lektury public API.
 * API docs: https://wolnelektury.pl/api/
 */
export const wolneLekturyTool = createTool({
  id: "wolne-lektury-tool",
  description:
    "Fetches a Polish book from the Wolne Lektury digital library. Provide a book slug like 'pan-tadeusz' or 'studnia-i-wahadlo'. Returns the full plain text.",
  inputSchema: z.object({
    slug: z
      .string()
      .describe("Book slug from wolnelektury.pl, e.g. 'pan-tadeusz'"),
  }),
  outputSchema: z.object({
    title: z.string(),
    author: z.string(),
    language: z.string(),
    text: z.string(),
    textLength: z.number(),
    bookUrl: z.string(),
  }),
  execute: async ({ inputData }) => {
    const { slug } = inputData;

    const metaRes = await fetch(
      `https://wolnelektury.pl/api/books/${slug}/?format=json`
    );
    if (!metaRes.ok) {
      throw new Error(
        `Book '${slug}' not found on Wolne Lektury (HTTP ${metaRes.status})`
      );
    }

    const meta = (await metaRes.json()) as {
      title: string;
      authors: Array<{ name: string }>;
      language: string;
      txt: string;
      url: string;
    };

    if (!meta.txt) {
      throw new Error(`Book '${slug}' has no plain text version`);
    }

    const textRes = await fetch(meta.txt);
    if (!textRes.ok) {
      throw new Error(`Failed to fetch text (HTTP ${textRes.status})`);
    }

    const rawText = await textRes.text();
    const text = cleanText(rawText);
    const author =
      meta.authors?.map((a) => a.name).join(", ") || "Nieznany";

    return {
      title: meta.title,
      author,
      language: meta.language || "pol",
      text,
      textLength: text.length,
      bookUrl: meta.url,
    };
  },
});

function cleanText(raw: string): string {
  let text = raw;
  // Strip WL header (usually first triple-newline block)
  const headerEnd = text.indexOf("\n\n\n");
  if (headerEnd !== -1 && headerEnd < 500) {
    text = text.slice(headerEnd).trim();
  }
  // Strip footer
  text = text
    .replace(/\n-{5,}\n[\s\S]*$/, "")
    .replace(/\nTa lektura[\s\S]*$/, "")
    .replace(/\nTen utwór[\s\S]*$/, "")
    .replace(/\nInformacja o utworze[\s\S]*$/, "");
  // Normalize whitespace
  text = text.replace(/\n{3,}/g, "\n\n").trim();
  return text;
}
