import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { ScoredPineconeRecord } from "@pinecone-database/pinecone";
import { performance } from "perf_hooks";

// High-performance, lightweight NLP for BM25 re-ranking
// Uses CommonJS under the hood; works with TS via default import
// If your TS config lacks esModuleInterop, switch to require() calls.
import bm25Factory from "wink-bm25-text-search";
import * as nlp from "wink-nlp-utils";

// ------- Env & clients -------
const PORT = Number(process.env.PORT) || 5000;
const OPENAI_KEY = process.env.OPENAI_API_KEY;
const PINECONE_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;

// sensible, fast default embedding model
const OPENAI_EMBEDDING_MODEL =
  process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small";
const EMBEDDING_DIM = Number(process.env.DIMENSIONS) || 1536;

// generation model
const OPENAI_CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini";

if (!OPENAI_KEY) throw new Error("Missing env: OPENAI_API_KEY");
if (!PINECONE_KEY || !PINECONE_INDEX)
  throw new Error("Missing env: PINECONE_API_KEY or PINECONE_INDEX");

const openai = new OpenAI({
  apiKey: OPENAI_KEY,
  timeout: 30_000, // ms
  maxRetries: 2,
});

const pinecone = new Pinecone({ apiKey: PINECONE_KEY });
const index = pinecone.Index(PINECONE_INDEX);

// ------- Express -------
const app = express();
app.use(cors({ origin: "*" }));
app.use(express.json({ limit: "2mb" }));

// ------- Utilities -------
type Meta = Record<string, any>;
type Match = ScoredPineconeRecord<Meta>;

function clean(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

async function embed(text: string): Promise<number[]> {
  const { data } = await openai.embeddings.create({
    model: OPENAI_EMBEDDING_MODEL,
    input: clean(text),
    dimensions: EMBEDDING_DIM,
  });
  const v = data?.[0]?.embedding;
  if (!v) throw new Error("Failed to get embedding");
  return v;
}

function extractText(md: Meta): string {
  return (
    (md?.text as string) ??
    (md?.content as string) ??
    (md?.chunk as string) ??
    ""
  );
}

// Min-max normalize to [0,1]; handles constant arrays.
function normalizeMap(scores: Map<string, number>): Map<string, number> {
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

/**
 * Lightweight BM25 re-ranker.
 * Builds an index only over Pinecone’s candidate set (fast, per-request).
 */
function bm25Rerank(
  query: string,
  docs: Array<{ id: string; text: string }>
): Map<string, number> {
  const bm25 = bm25Factory();
  bm25.defineConfig({ fldWeights: { text: 1 } });
  bm25.definePrepTasks([
    nlp.string.normalize,
    nlp.string.removeExtraSpaces,
    nlp.string.removePunctuations,
    nlp.string.toLowerCase,
    nlp.tokens.stem,
    nlp.tokens.propagateNegations,
    nlp.tokens.removeStopWords,
  ]);

  for (const d of docs) {
    bm25.addDoc({ text: d.text }, d.id);
  }
  bm25.consolidate();

  const res = bm25.search(query);
  // wink returns array sorted by BM25 desc with { id, score }
  const map = new Map<string, number>();
  for (const r of res) map.set(String(r.id), r.score);
  return map;
}

/**
 * Simple hybrid score: blend normalized dense score and normalized BM25.
 * alpha controls weight of semantic (dense) vs lexical (BM25).
 */
function blendScores(
  dense: Map<string, number>,
  bm25: Map<string, number>,
  alpha = 0.6
): Map<string, number> {
  const dn = normalizeMap(dense);
  const bn = normalizeMap(bm25);

  const ids = new Set([...dn.keys(), ...bn.keys()]);
  const out = new Map<string, number>();
  for (const id of ids) {
    const d = dn.get(id) ?? 0;
    const b = bn.get(id) ?? 0;
    out.set(id, alpha * d + (1 - alpha) * b);
  }
  return out;
}

// ------- Core RAG -------
async function retrieve(
  query: string,
  topK: number,
  namespace?: string
): Promise<Match[]> {
  const qv = await embed(query);

  // Dense vector search only in Pinecone (fast & reliable).
  // We keep hybrid “feel” by re-ranking locally with BM25.
  const res = await index.query({
    vector: qv,
    topK,
    includeMetadata: true,
    namespace,
    // Optional metadata filter example:
    // filter: namespace ? { userId: { $eq: namespace } } : undefined,
  });

  const matches = (res.matches || []) as Match[];
  return matches;
}

async function generateAnswer(
  question: string,
  context: string
): Promise<string> {
  const completion = await openai.chat.completions.create({
    model: OPENAI_CHAT_MODEL,
    temperature: 0.2,
    max_tokens: 600,
    messages: [
      {
        role: "system",
        content:
          "You are a precise assistant. Answer ONLY from the provided context. If the answer is not present, say you don't have enough information. Keep it concise.",
      },
      {
        role: "user",
        content: `Context:\n${context}\n\nQuestion: ${question}\nAnswer:`,
      },
    ],
  });
  return completion.choices?.[0]?.message?.content?.trim() || "";
}

async function rag(
  question: string,
  opts?: { namespace?: string; topK?: number; maxContextChars?: number; maxDocs?: number }
) {
  const namespace = opts?.namespace;
  const topK = Math.min(Math.max(opts?.topK ?? 25, 5), 100);
  const maxContextChars = Math.min(Math.max(opts?.maxContextChars ?? 6000, 1000), 12000);
  const maxDocs = Math.min(Math.max(opts?.maxDocs ?? 6, 2), 12);

  const t0 = performance.now();
  const denseMatches = await retrieve(question, topK, namespace);

  if (!denseMatches.length) {
    return {
      answer:
        "I couldn't find relevant information to answer your question from the current knowledge base.",
      sources: [],
      latencySec: (performance.now() - t0) / 1000,
    };
  }

  // Prepare docs for BM25 over candidate set
  const docs = denseMatches.map((m) => ({
    id: String(m.id),
    text: extractText(m.metadata || {}),
  }));

  const bm25Scores = bm25Rerank(question, docs);

  // Dense scores from Pinecone
  const denseScoreMap = new Map<string, number>();
  for (const m of denseMatches) {
    denseScoreMap.set(String(m.id), m.score ?? 0);
  }

  // Blend
  const blended = blendScores(denseScoreMap, bm25Scores, 0.65);

  // Sort by blended score (desc)
  const byId = new Map(denseMatches.map((m) => [String(m.id), m]));
  const ranked = [...blended.entries()]
    .map(([id, score]) => {
      const m = byId.get(id)!;
      return { ...m, score, id };
    })
    .filter((m) => extractText(m.metadata || {}).length > 0)
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

  // Build concise context
  const chunks: string[] = [];
  const sources: Array<{ id: string; source?: string; score: number }> = [];
  let used = 0;

  for (const m of ranked.slice(0, maxDocs)) {
    const md = (m.metadata || {}) as Meta;
    const chunk = clean(extractText(md));
    if (!chunk) continue;
    if (used + chunk.length > maxContextChars && chunks.length >= 2) break;

    chunks.push(chunk);
    sources.push({
      id: String(m.id),
      score: Number(m.score ?? 0),
      source: (md.source as string) || (md.url as string) || md.file || "unknown",
    });
    used += chunk.length;
    if (used >= maxContextChars) break;
  }

  const context = chunks.join("\n\n---\n\n");
  const answer = await generateAnswer(question, context);

  return {
    answer: answer || "I don't have enough information in the provided context.",
    sources,
    latencySec: (performance.now() - t0) / 1000,
  };
}

// ------- Routes -------
app.get("/api/v1/health", (_req, res) => {
  res.status(200).json({ status: "ok" });
});

app.post("/api/v1/query", async (req, res) => {
  try {
    const { query, namespace, topK, maxContextChars, maxDocs } = req.body ?? {};
    if (!query || typeof query !== "string" || query.length > 1000) {
      return res.status(400).json({ message: "Invalid 'query'." });
    }
    const cleanQuery = clean(query);
    const result = await Promise.race([
      rag(cleanQuery, { namespace, topK, maxContextChars, maxDocs }),
      new Promise((_r, rej) => setTimeout(() => rej(new Error("timeout")), 30_000)),
    ]);
    res.json(result);
  } catch (err) {
    console.error("query error:", err);
    res
      .status(500)
      .json({ message: "Internal error.", error: err instanceof Error ? err.message : "Unknown" });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 RAG server listening on http://localhost:${PORT}`);
});
