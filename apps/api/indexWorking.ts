import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import type { ScoredPineconeRecord } from "@pinecone-database/pinecone";
import { performance } from "perf_hooks";

const PORT = Number(process.env.PORT) || 5000;
const OPENAI_KEY = process.env.OPENAI_API_KEY;
const PINECONE_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;

const OPENAI_EMBEDDING_MODEL =
  process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-large";
const EMBEDDING_DIM = Number(process.env.DIMENSIONS) || 3072;
const OPENAI_CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini";

if (!OPENAI_KEY) throw new Error("Missing env: OPENAI_API_KEY");
if (!PINECONE_KEY || !PINECONE_INDEX)
  throw new Error("Missing env: PINECONE_API_KEY or PINECONE_INDEX");

const openai = new OpenAI({
  apiKey: OPENAI_KEY,
  timeout: 30_000,
  maxRetries: 2,
});

const pinecone = new Pinecone({ apiKey: PINECONE_KEY });

const app = express();
app.use(cors({ origin: "*" }));
app.use(express.json({ limit: "2mb" }));

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

// Simple keyword matching score as fallback
function simpleTextRerank(
  query: string,
  docs: Array<{ id: string; text: string }>
): Map<string, number> {
  const map = new Map<string, number>();
  
  // Simple keyword matching score
  const queryTerms = query.toLowerCase().split(/\s+/).filter(term => term.length > 2);
  
  for (const doc of docs) {
    let score = 0;
    const text = doc.text.toLowerCase();
    
    for (const term of queryTerms) {
      // Count occurrences of each term
      const occurrences = (text.match(new RegExp(term, 'g')) || []).length;
      score += occurrences;
    }
    
    // Normalize score by document length
    const wordCount = doc.text.split(/\s+/).length;
    const normalizedScore = score / Math.max(1, wordCount / 100);
    map.set(doc.id, normalizedScore);
  }
  
  return map;
}

function blendScores(dense: Map<string, number>, text: Map<string, number>, alpha = 0.6): Map<string, number> {
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
}


async function retrieve(query: string, topK: number, namespace?: string): Promise<Match[]> {
  const qv = await embed(query);

  const index = pinecone.Index(PINECONE_INDEX || "large-3072");
  
  try {
    if (namespace && typeof (index as any).namespace === 'function') {
      const namespacedIndex = (index as any).namespace(namespace);
      const res = await namespacedIndex.query({
        vector: qv,
        topK,
        includeMetadata: true,
      });
      return (res.matches || []) as Match[];
    }
    
    // If namespace method is not available, fall back to filter
    const filter = namespace ? { userId: { $eq: namespace } } : undefined;
    const res = await index.query({
      vector: qv,
      topK,
      includeMetadata: true,
      filter,
    });
    return (res.matches || []) as Match[];
  } catch (error) {
    console.error("Error in retrieve function:", error);
    throw error;
  }
}

async function generateAnswer(question: string, context: string): Promise<string> {
  const completion = await openai.chat.completions.create({
    model: OPENAI_CHAT_MODEL,
    temperature: 0.2,
    max_tokens: 600,
    messages: [
      {
        role: "system",
        content: `You are a helpful and precise assistant. Follow these rules:
          1. Answer primarily from the provided context
          2. If the answer isn't in the context, say so politely
          3. Be conversational but professional
          4. If relevant, suggest related topics from the context
          5. Structure complex answers clearly
          6. Always maintain a helpful tone`
      },
      {
        role: "user",
        content: `Context information: ${context}
        Question: ${question}
        Provide a helpful answer based on the context above:`
      }
    ],
  });
  return completion.choices?.[0]?.message?.content?.trim() || "";
}

async function rag(
  question: string,
  namespace?: string
) {
  // Set default values for parameters not provided in the payload
  const topK = 25;
  const maxContextChars = 6000;
  const maxDocs = 5;

  const t0 = performance.now();
  
  try {
    const denseMatches = await retrieve(question, topK, namespace);

    if (!denseMatches.length) {
      return {
        answer:
          "I couldn't find relevant information to answer your question from the current knowledge base.",
        sources: [],
        latencySec: (performance.now() - t0) / 1000,
      };
    }

    const docs = denseMatches.map((m) => ({
      id: String(m.id),
      text: extractText(m.metadata || {}),
    }));

    const textScores = simpleTextRerank(question, docs);

    const denseScoreMap = new Map<string, number>();
    for (const m of denseMatches) {
      denseScoreMap.set(String(m.id), m.score ?? 0);
    }

    const blended = blendScores(denseScoreMap, textScores, 0.65);

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
  } catch (error) {
    console.error("Error in RAG process:", error);
    throw error;
  }
}

// ------- Routes -------
app.get("/api/v1/health", async (req, res) => {
  res.status(200).json({ status: "ok" });
  try {
    const { userID } = req.body ?? "";
    
    if(!userID){
      return res.status(400).json({  message: "invalid userID hit the api." })
    }

    res.json(userID);
  } catch (error) {
    console.error("Error userID", error);
    res.status(500).json({ message: "Internal error.", error: error instanceof Error ? error.message : "Unknown" });
  }
});

app.post("/api/v1/query", async (req, res) => {
  try {
    const { query, namespace } = req.body ?? {};

    console.log("namespace", namespace);
    console.log("query", query);
    
    if (!query || typeof query !== "string" || query.length > 1000) {
      return res.status(400).json({ message: "Invalid 'query'." });
    }
    
    const cleanQuery = clean(query);
    const result = await Promise.race([
      rag(cleanQuery, namespace),
      new Promise((_r, rej) => setTimeout(() => rej(new Error("timeout")), 30_000)),
    ]);
    
    res.json(result);
  } catch (err) {
    console.error("Query error:", err);
    res
      .status(500)
      .json({ 
        message: "Internal error.", 
        error: err instanceof Error ? err.message : "Unknown",
      });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 RAG server listening on http://localhost:${PORT}`);
});