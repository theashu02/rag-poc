import { performance } from "perf_hooks";
import { LRUCache } from "lru-cache";

const CLEAN_CACHE = new LRUCache<string, string>({
  max: 2000,
  ttl: 1000 * 60 * 10,
});

function sanitize(text: string): string {
  if (typeof text !== "string") {
    throw new TypeError("Input must be a string");
  }

  if (text.length === 0) {
    return "";
  }

  let s = text.normalize("NFD");
  s = s.replace(/[\u0300-\u036f]/g, "");

  try {
    s = s.replace(/[^\p{L}\p{N}\s]/gu, " ");
  } catch {
    s = s.replace(/[^\w\s]/g, " ");
  }

  s = s.replace(/\s+/g, " ").trim();
  return s.toLowerCase();
}

export function cleanSync(text: string): string {
  const startTime = performance.now();
  try {
    const cached = CLEAN_CACHE.get(text);
    if (cached) return cached;

    const cleaned = sanitize(text);
    CLEAN_CACHE.set(text, cleaned);
    return cleaned;
  } finally {
    const duration = performance.now() - startTime;
    // Capture synchronous cleaning latency for quick visibility into preprocessing cost.
    console.log(`[Timing] cleanSync completed in ${duration.toFixed(2)}ms`);
  }
}

export async function clean(text: string): Promise<string> {
  const startTime = performance.now();
  try {
    return cleanSync(text);
  } finally {
    const duration = performance.now() - startTime;
    // Observe async wrapper cost for query sanitation when awaiting.
    console.log(`[Timing] clean completed in ${duration.toFixed(2)}ms`);
  }
}
