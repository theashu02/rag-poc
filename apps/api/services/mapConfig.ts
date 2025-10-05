import { performance } from "perf_hooks";

export function normalizeMap(scores: Map<string, number>): Map<string, number> {
  const startTime = performance.now();
  try {
    const vals = [...scores.values()];
    if (vals.length === 0) return scores;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    if (max === min) {
      const mid = 0.5;
      return new Map([...scores.keys()].map((k) => [k, mid]));
    }
    return new Map(
      [...scores.entries()].map(([k, v]) => [k, (v - min) / (max - min)])
    );
  } finally {
    const duration = performance.now() - startTime;
    // Track normalization cost to spot unnecessary overhead in scoring.
    console.log(`[Timing] normalizeMap completed in ${duration.toFixed(2)}ms`);
  }
}
