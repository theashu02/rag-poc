import { performance } from "perf_hooks";
import { PINECONE_INDEX, OPENAI_EMBEDDING_MODEL, EMBEDDING_DIM, embeddingQueue, EMBEDDING_CACHE } from "./config";
import type { Match } from "./config";
import { pinecone } from "./pinecone";
import { cleanSync } from "./cleanQuery";
import { openai } from "./openai";
import { createHash } from "crypto";

type EmbedOptions = {
  skipClean?: boolean;
};

type RetrieveOptions = {
  alreadyClean?: boolean;
};

export async function embed(
  text: string,
  options: EmbedOptions = {}
): Promise<number[]> {
  const startTime = performance.now();
  try {
    const preparedText = options.skipClean ? text : cleanSync(text);

    if (!preparedText) {
      throw new Error("Embedding text cannot be empty");
    }

    const cacheKey = createHash("sha256").update(preparedText).digest("hex");
    const cached = EMBEDDING_CACHE.get(cacheKey);
    if (cached) return cached;

    const result = await embeddingQueue.add(async () => {
      const cachedInQueue = EMBEDDING_CACHE.get(cacheKey);
      if (cachedInQueue) return cachedInQueue;

      try {
        const { data } = await openai.embeddings.create({
          model: OPENAI_EMBEDDING_MODEL,
          input: preparedText,
          dimensions: EMBEDDING_DIM,
        });

        const embedding = data?.[0]?.embedding;
        if (!Array.isArray(embedding)) {
          throw new Error("Failed to get valid embedding from OpenAI");
        }

        EMBEDDING_CACHE.set(cacheKey, embedding);
        return embedding;
      } catch (error) {
        console.error("Embedding error:", error);
        throw new Error(
          `Embedding failed: ${
            error instanceof Error ? error.message : "Unknown error"
          }`
        );
      }
    });

    return result as number[];
  } finally {
    const duration = performance.now() - startTime;
    // Monitor embedding latency, including cache checks and network calls.
    console.log(`[Timing] embed completed in ${duration.toFixed(2)}ms`);
  }
}

export async function retrieve(
  query: string,
  topK: number,
  namespace?: string,
  options: RetrieveOptions = {}
): Promise<Match[]> {
  const startTime = performance.now();
  try {
    const preparedQuery = options.alreadyClean ? query : cleanSync(query);
    if (!preparedQuery) {
      return [];
    }

    const qv = await embed(preparedQuery, { skipClean: true });
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
  } finally {
    const duration = performance.now() - startTime;
    // Track end-to-end vector retrieval latency, including Pinecone retries.
    console.log(`[Timing] retrieve completed in ${duration.toFixed(2)}ms`);
  }
}
