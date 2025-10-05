import { performance } from "perf_hooks";
import { normalizeMap } from "./mapConfig";

export function blendScores(
  dense: Map<string, number>,
  text: Map<string, number>,
  alpha = 0.65
): Map<string, number> {
  const startTime = performance.now();
  try {
    const dn = normalizeMap(dense);
    const tn = normalizeMap(text);

    const ids = new Set([...dn.keys(), ...tn.keys()]);
    const out = new Map<string, number>();
    for (const id of ids) {
      const d = dn.get(id) ?? 0;
      const t = tn.get(id) ?? 0;
      out.set(id, alpha * d + (1 - alpha) * t);
    }
    return out;
  } finally {
    const duration = performance.now() - startTime;
    // Highlight time spent blending dense and sparse scores.
    console.log(`[Timing] blendScores completed in ${duration.toFixed(2)}ms`);
  }
}
