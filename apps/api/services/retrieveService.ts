import { PINECONE_INDEX, OPENAI_EMBEDDING_MODEL, EMBEDDING_DIM, embeddingQueue, EMBEDDING_CACHE } from "./config";
import type { Match } from "./config";
import { pinecone } from "./pinecone";
import { clean } from "./cleanQuery";
import { openai } from "./openai";
import { createHash } from "crypto";

export async function embed(text: string): Promise<number[]> {
  const cacheKey = createHash("sha256").update(text).digest("hex");
  const cached = EMBEDDING_CACHE.get(cacheKey);
  if (cached) return cached;

  return embeddingQueue.add(async () => {
    // Double-check cache inside queue to prevent duplicate requests
    const cachedInQueue = EMBEDDING_CACHE.get(cacheKey);
    if (cachedInQueue) return cachedInQueue;

    try {
      const { data } = await openai.embeddings.create({
        model: OPENAI_EMBEDDING_MODEL,
        input: await clean(text),
        dimensions: EMBEDDING_DIM,
      });

      const v = data?.[0]?.embedding;
      if (!v || !Array.isArray(v)) {
        throw new Error("Failed to get valid embedding from OpenAI");
      }

      EMBEDDING_CACHE.set(cacheKey, v);
      return v;
    } catch (error) {
      console.error("Embedding error:", error);
      throw new Error(
        `Embedding failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }) as Promise<number[]>;
}

export async function retrieve(
  query: string,
  topK: number,
  namespace?: string
): Promise<Match[]> {
  const qv = await embed(query);
  const index = pinecone.Index(PINECONE_INDEX || "large-3072");

  const maxRetries = 2;
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      if (namespace && typeof (index as any).namespace === "function") {
        const namespacedIndex = (index as any).namespace(namespace);
        const res = await namespacedIndex.query({
          vector: qv,
          topK,
          includeMetadata: true,
        });
        return (res.matches || []) as Match[];
      }

      const filter = namespace ? { userId: { $eq: namespace } } : undefined;
      const res = await index.query({
        vector: qv,
        topK,
        includeMetadata: true,
        filter,
      });
      return (res.matches || []) as Match[];
    } catch (error) {
      lastError = error as Error;
      console.warn(`Retrieve attempt ${attempt + 1} failed:`, error);
      if (attempt < maxRetries - 1) {
        await new Promise((resolve) =>
          setTimeout(resolve, Math.pow(2, attempt) * 100)
        );
      }
    }
  }

  throw lastError || new Error("All retrieve attempts failed");
}
