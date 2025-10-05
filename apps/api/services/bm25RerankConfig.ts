import { performance } from "perf_hooks";
import bm25Search from "wink-bm25-text-search";
import { createHash } from "crypto";
import { BM25_INSTANCES_CACHE } from "./config";
import bm25PrepTasks from "./BM25_config";

export function bm25Rerank(
  query: string,
  docs: Array<{ id: string; text: string }>,
  namespace?: string
): Map<string, number> {
  const startTime = performance.now();
  try {
    const scores = new Map<string, number>();
    if (!docs.length) return scores;

    // Create cache key for BM25 instance
    const docIds = docs
      .map((d) => d.id)
      .sort()
      .join(",");
    const bm25CacheKey = createHash("sha256")
      .update(`${docIds}:${namespace || "default"}`)
      .digest("hex");

    let bm25 = BM25_INSTANCES_CACHE.get(bm25CacheKey);

    if (!bm25) {
      bm25 = bm25Search();
      bm25.defineConfig({ fldWeights: { text: 1 } });
      bm25.definePrepTasks(bm25PrepTasks);

      docs.forEach((d) => bm25.addDoc({ text: d.text }, d.id));
      bm25.consolidate();

      BM25_INSTANCES_CACHE.set(bm25CacheKey, bm25);
    }

    try {
      bm25
        .search(query, docs.length)
        .forEach(([id, score]: [string, number]) => scores.set(id, score));
    } catch (error) {
      console.error("BM25 search error:", error);
      // Fallback: equal scores for all docs
      docs.forEach((d) => scores.set(d.id, 1.0));
      return scores;
    }

    docs.forEach((d) => {
      if (!scores.has(d.id)) scores.set(d.id, 0);
    });

    return scores;
  } finally {
    const duration = performance.now() - startTime;
    // Quantify BM25 reranking time to evaluate sparse scoring overhead.
    console.log(`[Timing] bm25Rerank completed in ${duration.toFixed(2)}ms`);
  }
}
