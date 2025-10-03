export function normalizeMap(scores: Map<string, number>): Map<string, number> {
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
}