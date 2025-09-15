import express from "express";
import cors from "cors";
import { OpenAI } from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import type { ScoredPineconeRecord } from "@pinecone-database/pinecone";
import { performance } from "perf_hooks";
import dotenv from "dotenv";
import rateLimit from "express-rate-limit";
import pRetry from "p-retry";

dotenv.config();

const PORT = Number(process.env.PORT) || 5000;
const OPENAI_KEY = process.env.OPENAI_API_KEY;
const PINECONE_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;

const OPENAI_EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-large";
const EMBEDDING_DIM = Number(process.env.DIMENSIONS) || 3072;
const OPENAI_CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini";

if (!OPENAI_KEY) throw new Error("Missing env: OPENAI_API_KEY");
if (!PINECONE_KEY || !PINECONE_INDEX)
  throw new Error("Missing env: PINECONE_API_KEY or PINECONE_INDEX");

const openai = new OpenAI({
  apiKey: OPENAI_KEY,
  timeout: 30_000,
  maxRetries: 0, // We'll handle retries manually
});

const pinecone = new Pinecone({ apiKey: PINECONE_KEY });

const app = express();

// Security: restrict CORS in production
app.use(
  cors({
    origin: process.env.FRONTEND_URL
      ? process.env.FRONTEND_URL.split(",")
      : ["http://localhost:3000", "http://127.0.0.1:3000"],
    methods: ["POST", "GET"],
  })
);

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,
  message: { error: "Too many requests from this IP. Try again later." },
  standardHeaders: true,
  legacyHeaders: false,
});
app.use("/api/v1/query", limiter);

app.use(express.json({ limit: "2mb" }));

type Meta = Record<string, any>;
type Match = ScoredPineconeRecord<Meta>;

function clean(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

function isGreeting(query: string): boolean {
  const greetings = [
    "hello", "hi", "hey", "greetings", "good morning",
    "good afternoon", "good evening", "howdy", "hola",
    "what's up", "yo", "sup", "hi there", "hello there"
  ];
  const cleanQuery = query.toLowerCase().trim();
  return greetings.some(g =>
    cleanQuery === g ||
    cleanQuery.startsWith(g + "?") ||
    cleanQuery.startsWith(g + "!") ||
    new RegExp(`\\b${g}\\b`).test(cleanQuery)
  );
}

function generateGreetingResponse(query: string): string {
  const hour = new Date().getHours();
  let timeGreeting = "Hello";
  if (hour < 12) timeGreeting = "Good morning";
  else if (hour < 18) timeGreeting = "Good afternoon";
  else timeGreeting = "Good evening";

  const cleanQuery = query.toLowerCase().trim();

  if (cleanQuery.includes("how are you")) {
    return `${timeGreeting}! I'm doing well, thank you. How can I help you today?`;
  }

  if (cleanQuery.includes("what's up") || cleanQuery.includes("sup")) {
    return `${timeGreeting}! Not much, just here to help. What do you need?`;
  }

  const responses = [
    `${timeGreeting}! How can I assist you?`,
    `${timeGreeting}! What would you like to know?`,
    `${timeGreeting}! I'm ready to help.`,
  ];
  return responses[Math.floor(Math.random() * responses.length)];
}

function isSmallTalk(query: string): boolean {
  const patterns = [
    "how are you", "what's up", "how's it going", "who are you",
    "what can you do", "tell me about yourself"
  ];
  const cleanQuery = query.toLowerCase().trim();
  return patterns.some(p => cleanQuery.includes(p));
}

function generateSmallTalkResponse(query: string): string {
  const cleanQuery = query.toLowerCase().trim();

  if (cleanQuery.includes("how are you") || cleanQuery.includes("how's it going")) {
    return "I'm doing well, thanks! I'm here to help you find answers. What can I do for you?";
  }

  if (cleanQuery.includes("what's up")) {
    return "Just ready to help you out! What would you like to know?";
  }

  if (cleanQuery.includes("who are you") || cleanQuery.includes("what can you do")) {
    return "I'm an AI assistant that helps you find information from your knowledge base. Ask me anything!";
  }

  return "I'm here to help. What would you like to know about?";
}

// Embedding with retry
async function embed(text: string): Promise<number[]> {
  return await pRetry(
    async () => {
      const { data } = await openai.embeddings.create({
        model: OPENAI_EMBEDDING_MODEL,
        input: clean(text),
        dimensions: EMBEDDING_DIM,
      });
      const embedding = data[0]?.embedding;
      if (!embedding) throw new Error("OpenAI returned empty embedding");
      return embedding;
    },
    {
      retries: 2,
      factor: 2,
      onFailedAttempt: (err) => {
        console.warn(`Embedding retry (${err.attemptNumber}):`, err.message);
      },
    }
  );
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
    return new Map([...scores.keys()].map(k => [k, 0.5]));
  }
  return new Map(
    [...scores.entries()].map(([k, v]) => [k, (v - min) / (max - min)])
  );
}

function simpleTextRerank(
  query: string,
  docs: Array<{ id: string; text: string }>
): Map<string, number> {
  const map = new Map<string, number>();
  const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);

  for (const doc of docs) {
    let score = 0;
    const text = doc.text.toLowerCase();
    for (const term of queryTerms) {
      const occurrences = (text.match(new RegExp(term, 'g')) || []).length;
      score += occurrences;
    }
    const wordCount = doc.text.split(/\s+/).length;
    map.set(doc.id, score / Math.max(1, wordCount / 100));
  }
  return map;
}

function blendScores(dense: Map<string, number>, text: Map<string, number>, alpha = 0.65): Map<string, number> {
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
  const index = pinecone.Index(PINECONE_INDEX);

  try {
    if (namespace) {
      const ns = (index as any).namespace?.(namespace);
      if (ns) {
        const res = await ns.query({
          vector: qv,
          topK,
          includeMetadata: true,
        });
        return res.matches || [];
      }
    }

    const filter = namespace ? { userId: { $eq: namespace } } : undefined;
    const res = await index.query({
      vector: qv,
      topK,
      includeMetadata: true,
      filter,
    });
    return res.matches || [];
  } catch (error) {
    console.error("Retrieve error:", error);
    throw new Error("Failed to query vector database");
  }
}

async function generateAnswer(question: string, context: string): Promise<string> {
  try {
    const completion = await pRetry(
      async () => {
        return await openai.chat.completions.create({
          model: OPENAI_CHAT_MODEL,
          temperature: 0.2,
          max_tokens: 800,
          messages: [
            {
              role: "system",
              content: `You are a helpful assistant. Answer clearly from the context. If unsure, say "I don't have enough information." Be concise and friendly.`,
            },
            {
              role: "user",
              content: `Context:\n${context}\n\nQuestion: ${question}\nAnswer:`,
            },
          ],
        });
      },
      {
        retries: 2,
        onFailedAttempt: (err) => {
          console.warn(`LLM generation retry (${err.attemptNumber}):`, err.message);
        },
      }
    );

    return completion.choices[0]?.message?.content?.trim() || "No response generated.";
  } catch (err) {
    console.error("LLM generation failed:", err);
    return "Sorry, I couldn't generate a response right now.";
  }
}

function generateFollowupSuggestions(question: string): string {
  const suggestions = [
    "\n\nWould you like more details on this?",
    "\n\nI can help with related topics if you're interested.",
    "\n\nFeel free to ask for clarification!",
    "\n\nWant to dive deeper into any part?"
  ];
  return suggestions[Math.floor(Math.random() * suggestions.length)];
}

async function rag(question: string, namespace?: string) {
  const t0 = performance.now();

  // ✅ Short-circuit for greetings and small talk
  if (isGreeting(question)) {
    return {
      answer: generateGreetingResponse(question),
      sources: [],
      latencySec: (performance.now() - t0) / 1000,
    };
  }

  if (isSmallTalk(question)) {
    return {
      answer: generateSmallTalkResponse(question),
      sources: [],
      latencySec: (performance.now() - t0) / 1000,
    };
  }

  const topK = 25;
  const maxContextChars = 6000;
  const maxDocs = 6;

  try {
    const denseMatches = await retrieve(question, topK, namespace);

    if (denseMatches.length === 0) {
      return {
        answer: "I couldn't find relevant information to answer your question. Please try rephrasing it.",
        sources: [],
        latencySec: (performance.now() - t0) / 1000,
      };
    }

    const docs = denseMatches.map(m => ({
      id: String(m.id),
      text: extractText(m.metadata || {}),
    })).filter(d => d.text.length > 0);

    if (docs.length === 0) {
      return {
        answer: "No valid content found in retrieved documents.",
        sources: [],
        latencySec: (performance.now() - t0) / 1000,
      };
    }

    const textScores = simpleTextRerank(question, docs);
    const denseScoreMap = new Map(denseMatches.map(m => [String(m.id), m.score ?? 0]));
    const blended = blendScores(denseScoreMap, textScores, 0.65);

    const byId = new Map(denseMatches.map(m => [String(m.id), m]));
    const ranked = [...blended.entries()]
      .map(([id, score]) => ({ ...byId.get(id)!, score }))
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
      .slice(0, maxDocs);

    const chunks: string[] = ["Relevant information:"];
    const sources: Array<{ id: string; source?: string; score: number }> = [];
    let usedChars = 0;

    for (const match of ranked) {
      const md = match.metadata || {};
      const text = clean(extractText(md));
      if (!text) continue;
      if (usedChars + text.length > maxContextChars && chunks.length > 2) break;

      const source = (md.source as string) || (md.url as string) || (md.file as string) || "unknown";
      chunks.push(`${text} [Source: ${source}]`);
      sources.push({ id: String(match.id), score: Number(match.score ?? 0), source });
      usedChars += text.length;
    }

    const context = chunks.join("\n\n---\n\n");
    const answer = await generateAnswer(question, context);

    return {
      answer: answer + generateFollowupSuggestions(question),
      sources,
      latencySec: (performance.now() - t0) / 1000,
    };
  } catch (err) {
    console.error("RAG error:", err);
    return {
      answer: "Sorry, an error occurred while processing your request.",
      sources: [],
      latencySec: (performance.now() - t0) / 1000,
    };
  }
}

// ------- Routes -------
app.get("/api/v1/health", async (_req, res) => {
  try {
    await pinecone.describeIndex(PINECONE_INDEX!);
    res.status(200).json({ status: "ok", db: "connected" });
  } catch (err) {
    console.error("Health check failed:", err);
    res.status(500).json({ status: "error", db: "disconnected" });
  }
});

app.post("/api/v1/query", async (req, res) => {
  try {
    const { query, namespace } = req.body ?? {};

    if (!query || typeof query !== "string" || query.trim().length === 0 || query.length > 1000) {
      return res.status(400).json({ message: "Invalid or missing 'query'. Must be a string (1-1000 chars)." });
    }

    const cleanQuery = clean(query);
    console.log(`[Query] ${cleanQuery} | Namespace: ${namespace || "global"}`);

    const result = await Promise.race([
      rag(cleanQuery, namespace),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("Request timeout")), 30000)
      ),
    ]);

    res.json(result);
  } catch (err) {
    console.error("API Error:", err);
    res.status(500).json({
      message: "Internal server error.",
      error: err instanceof Error ? err.message : "Unknown error",
    });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 RAG server running on http://localhost:${PORT}`);
  console.log(`📌 Index: ${PINECONE_INDEX}`);
  console.log(`🧠 Model: ${OPENAI_CHAT_MODEL}`);
});