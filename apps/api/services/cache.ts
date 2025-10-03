import { LRUCache } from "lru-cache";
import { createHash } from "crypto";

export const SEMANTIC_CACHE = new LRUCache<
  string,
  {
    answer: string;
    sources: Array<{ id: string; source?: string; score: number }>;
    latencySec: number;
  }
>({
  max: 5000,
  ttl: 1000 * 60 * 30, // 30 minutes
});

export function generateSemanticCacheKey(
  query: string,
  namespace?: string
): string {
  const normalized = query
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .trim();
  return createHash("sha256")
    .update(`${normalized}:${namespace || "default"}`)
    .digest("hex");
}
