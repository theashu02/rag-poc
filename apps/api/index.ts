import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import type { QueryResponse } from "@pinecone-database/pinecone";
import { performance } from "perf_hooks";
import nlp from "compromise";
import { WordNet } from "natural";
import cluster from "cluster";
import os from "os";
import { LRUCache as LRU } from "lru-cache";

/* cluster – use all CPU cores */
const cpuCount = os.cpus().length;
if (cluster.isPrimary) {
  console.log(`Primary ${process.pid} - forking ${cpuCount} workers`);
  for (let i = 0; i < cpuCount; i++) cluster.fork();

  cluster.on("exit", worker => {
    console.log(`Worker ${worker.process.pid} died – starting replacement`);
    cluster.fork();
  });
}

const app = express();
const PORT = Number(process.env.PORT) || 5000;
const OPENAI_KEY = process.env.OPENAI_API_KEY;
const PINECONE_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;

if (!OPENAI_KEY) throw new Error("Missing env: OPENAI_API_KEY");
if (!PINECONE_KEY || !PINECONE_INDEX)
  throw new Error("Missing env: PINECONE_API_KEY or PINECONE_INDEX");

const openai = new OpenAI({
  apiKey: OPENAI_KEY,
  timeout: 15_000,
  maxRetries: 2,
});

const pinecone = new Pinecone({ apiKey: PINECONE_KEY });
const index = pinecone.Index(PINECONE_INDEX);

app.use(cors({ origin: "http://localhost:3000" }));
app.use(express.json({ limit: "3mb" }));

// High-performance LRU caches
class OptimizedCacheManager {
  private embeddingCache = new LRU<string, any[]>({ max: 2000, ttl: 60 * 60 * 1000, updateAgeOnGet: true });
  private paraphraseCache = new LRU<string, any[]>({ max: 1000, ttl: 30 * 60 * 1000, updateAgeOnGet: true });
  private synonymCache  = new LRU<string, string>({ max: 1500, ttl: 2 * 60 * 60 * 1000, updateAgeOnGet: true });
  private responseCache = new LRU<string, any[]>({ max: 500,  ttl: 10 * 60 * 1000, updateAgeOnGet: true });

  getEmbedding(t: string)                { return this.embeddingCache.get(t)  || null; }
  setEmbedding(t: string, e: number[])   { this.embeddingCache.set(t, e); }
  getParaphrases(t: string)              { return this.paraphraseCache.get(t) || null; }
  setParaphrases(t: string, p: string[]) { this.paraphraseCache.set(t, p); }
  getSynonyms(t: string)                 { return this.synonymCache.get(t)    || null; }
  setSynonyms(t: string, s: string)      { this.synonymCache.set(t, s); }
  getResponse(k: string)                 { return this.responseCache.get(k)   || null; }
  setResponse(k: string, r: any)         { this.responseCache.set(k, r); }

  // quick fuzzy match for near-duplicate queries
  findSimilarResponse(q: string) {
    const qw = new Set(q.toLowerCase().split(/\s+/));
    for (const [cachedQ, resp] of this.responseCache.entries()) {
      const cw = new Set(cachedQ.toLowerCase().split(/\s+/));
      const inter = [...qw].filter(x => cw.has(x));
      if (inter.length / Math.max(qw.size, cw.size) > 0.85) return resp;
    }
    return null;
  }
}
const cache = new OptimizedCacheManager();

const embeddingQueue: Array<{ text: string; resolve: (e: number[]) => void; reject: (err: any) => void; }> = [];
let   isProcessingEmbeddings = false;

async function processEmbeddingQueue() {
  if (isProcessingEmbeddings || embeddingQueue.length === 0) return;
  isProcessingEmbeddings = true;

  const BATCH_SIZE = 100;
  while (embeddingQueue.length) {
    const batch = embeddingQueue.splice(0, BATCH_SIZE);
    try {
      const { data } = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: batch.map(b => b.text),
        dimensions: 1536,
      });
      batch.forEach((item, i) => {
        const emb = data[i]?.embedding || [];
        cache.setEmbedding(item.text, emb);
        item.resolve(emb);
      });
    } catch (err) {
      batch.forEach(it => it.reject(err));
    }
  }
  isProcessingEmbeddings = false;
}
function embed(text: string): Promise<number[]> {
  const cached = cache.getEmbedding(text);
  if (cached) return Promise.resolve(cached);

  return new Promise((resolve, reject) => {
    embeddingQueue.push({ text, resolve, reject });
    (embeddingQueue.length <= 10 ? setImmediate : setTimeout)(processEmbeddingQueue, 50);
  });
}

class FastQueryProcessor {
  private wn = new WordNet();
  private static readonly WORD_REGEX = /\b\w{3,}\b/g;

  async expandWithSynonyms(query: string): Promise<string> {
    const cached = cache.getSynonyms(query);
    if (cached) return cached;

    if (query.length < 10) {
      cache.setSynonyms(query, query);
      return query;
    }

    const words = query.toLowerCase().match(FastQueryProcessor.WORD_REGEX) || [];
    if (words.length === 0) return query;

    const important = words.slice(0, 5);
    const result = await Promise.race([
      this.getSynonymsForWords(important),
      new Promise<string>(res => setTimeout(() => res(query), 500)),
    ]);

    const final = typeof result === "string" ? result : query;
    cache.setSynonyms(query, final);
    return final;
  }

  private async getSynonymsForWords(words: string[]): Promise<string> {
    const promises = words.map(word =>
      new Promise<string>(resolve => {
        const to = setTimeout(() => resolve(word), 100);
        this.wn.lookup(word, results => {
          clearTimeout(to);
          const syns = results?.[0]?.synonyms?.slice(0, 1) ?? [];
          resolve([word, ...syns].join(" "));
        });
      }),
    );
    const expanded = await Promise.all(promises);
    return expanded.join(" ");
  }

  async paraphrases(query: string): Promise<string[]> {
    const cached = cache.getParaphrases(query);
    if (cached) return cached;

    if (query.length < 15 || query.split(/\s+/).length <= 2) {
      cache.setParaphrases(query, []);
      return [];
    }

    try {
      const completion = await Promise.race([
        openai.chat.completions.create({
          model: "gpt-4o-mini",
          temperature: 0.4,
          max_tokens: 50,
          messages: [
            { role: "system", content: "Rephrase the query once. Be concise." },
            { role: "user", content: query },
          ],
        }),
        new Promise((_, reject) => setTimeout(() => reject(new Error("Timeout")), 2000)),
      ]) as any;

      const paraphrase = completion.choices[0]?.message.content?.trim();
      const out = paraphrase ? [paraphrase] : [];
      cache.setParaphrases(query, out);
      return out;
    } catch {
      cache.setParaphrases(query, []);
      return [];
    }
  }

  entities(text: string): string[] {
    const key = `entities:${text}`;
    const cached = cache.getResponse(key);
    if (cached) return cached;

    const entities = Array.from(
      new Set([
        ...nlp(text).topics().out("array"),
        ...nlp(text).people().out("array"),
        ...nlp(text).places().out("array"),
      ]),
    ).slice(0, 5);

    cache.setResponse(key, entities);
    return entities;
  }

  async intent(query: string) {
    const len = query.length;
    const words = query.split(/\s+/).length;

    if (len < 15 && words <= 2) {
      return { expanded: query, alternatives: [query], entities: [], keywords: query.split(/\s+/) };
    }

    if (len < 30 && words <= 4) {
      const expanded = await this.expandWithSynonyms(query);
      return { expanded, alternatives: [query], entities: this.entities(query), keywords: query.split(/\s+/) };
    }

    try {
      const [expanded, alts] = await Promise.race([
        Promise.all([this.expandWithSynonyms(query), this.paraphrases(query)]),
        new Promise((_, rej) => setTimeout(() => rej(new Error("Intent timeout")), 3000)),
      ]) as [string, string[]];

      return {
        expanded,
        alternatives: [query, ...alts].slice(0, 2),
        entities: this.entities(query),
        keywords: query.split(/\s+/).slice(0, 8),
      };
    } catch {
      return {
        expanded: query,
        alternatives: [query],
        entities: this.entities(query),
        keywords: query.split(/\s+/).slice(0, 5),
      };
    }
  }
}

/* ---------- High-Performance Search Engine ---------- */
class HighPerformanceSearchEngine {
  private qp = new FastQueryProcessor();
  private searchCache = new LRU<string, QueryResponse>({
    max: 200,
    ttl: 5 * 60 * 1000,
  });

  // Wrap Pinecone query with caching and the correct return type
  private async optimizedPineconeSearch(
    embedding: number[],
    topK: number
  ): Promise<{ matches?: any[] }> {
    const key = `search:${embedding.slice(0, 10).join(",")}:${topK}`;
    const cached = this.searchCache.get(key);
    if (cached) return cached;

    const res = await index.query({
      vector: embedding,
      topK,
      includeMetadata: true,
    });
    this.searchCache.set(key, res as any);
    return res as any;
  }

  // Main multi-query search with deduplication and scoring
  async search(query: string, topK = 20) {
    const key = `full_search:${query}:${topK}`;
    const cached = cache.getResponse(key);
    if (cached) return cached;

    const { alternatives } = await this.qp.intent(query);
    const limited = alternatives.slice(0, 2);
    const embeddings = await Promise.all(limited.map(embed));

    const results = await Promise.all(
      embeddings.map((e) => this.optimizedPineconeSearch(e, topK))
    );
    const matches = results.map((r) => r.matches ?? []);

    const scoreMap = new Map<string, number>();
    const matchMap = new Map<string, any>();
    matches.flat().forEach((m) => {
      const s = scoreMap.get(m.id) || 0;
      if ((m.score ?? 0) > s) {
        scoreMap.set(m.id, m.score ?? 0);
        matchMap.set(m.id, m);
      }
    });

    const finalMatches = Array.from(matchMap.values()).sort(
      (a, b) => (b.score ?? 0) - (a.score ?? 0)
    );
    cache.setResponse(key, finalMatches);
    return finalMatches;
  }
}

class UltraFastRAGEngine {
  private searcher = new HighPerformanceSearchEngine();
  private CONTEXT_CHARS = 8000;
  private MAX_DOCS = 5;

  async query(question: string) {
    const qKey = `rag:${question}`;
    const exact = cache.getResponse(qKey);
    if (exact) return { ...exact, cached: true, searchTime: 0 };

    const similar = cache.findSimilarResponse(question);
    if (similar) return { ...similar, cached: true, searchTime: 0 };

    const t0 = performance.now();
    const matches = await this.searcher.search(question, 15);

    if (!matches.length) {
      const res = { answer: "I couldn't find relevant information to answer your question.", sources: [], searchTime: (performance.now() - t0) / 1000 };
      cache.setResponse(qKey, res);
      return res;
    }

    let chars = 0;
    const ctx: string[] = [];
    const sources: any[] = [];

    for (let i = 0; i < Math.min(matches.length, this.MAX_DOCS); i++) {
      const m = matches[i];
      const md = (m.metadata ?? {}) as Record<string, any>;
      const chunk = md.text ?? md.content ?? md.pageContent ?? md.page_content ?? "";
      if (!chunk) continue;

      if (chars + chunk.length > this.CONTEXT_CHARS && ctx.length >= 2) break;
      chars += chunk.length;
      ctx.push(`[Doc ${i + 1}] ${chunk}`);
      sources.push({ id: m.id, score: m.score ?? 0, source: md.source ?? "Unknown" });
      if (chars >= this.CONTEXT_CHARS) break;
    }

    const context = ctx.join("\n\n");

    const completion = await Promise.race([
      openai.chat.completions.create({
        model: "gpt-4o-mini",
        temperature: 0.2,
        max_tokens: 400,
        messages: [
            {
              role: "system",
              content: `You are a helpful assistant that answers questions based on the provided context.
                Follow these guidelines:
                1. Answer based ONLY on the information in the context
                2. If the context doesn't contain enough information, say so
                3. Be concise but comprehensive
                4. Cite document numbers when referencing specific information`,
            },
            { role: "user", content: `Context:\n${context}\n\nQ: ${question}\nA:` },
          ],
      }),
      new Promise((_, rej) => setTimeout(() => rej(new Error("OpenAI timeout")), 8000)),
    ]) as any;

    const result = {
      answer: completion.choices[0]?.message.content?.trim() || "Unable to generate answer.",
      sources,
      searchTime: (performance.now() - t0) / 1000,
      documentsUsed: ctx.length,
      totalRetrieved: matches.length,
      contextChars: chars,
    };

    cache.setResponse(qKey, result);
    return result;
  }
}

// request duplication
const pendingRequests = new Map<string, Promise<any>>();
function deduplicate<T>(key: string, fn: () => Promise<T>): Promise<T> {
  if (pendingRequests.has(key)) return pendingRequests.get(key) as Promise<T>;
  const p = fn().finally(() => pendingRequests.delete(key));
  pendingRequests.set(key, p);
  return p;
}

const ragEngine = new UltraFastRAGEngine();

app.get("/api/v1/health", (_, res) => {
  res.setHeader("Cache-Control", "public, max-age=30");
  res.status(200).json({ status: "API up & healthy", ts: Date.now() });
});

app.post("/api/v1/query", async (req, res) => {
  try {
    const { query } = req.body ?? {};
    if (!query || typeof query !== "string" || query.length > 500)
      return res.status(400).json({ message: "Invalid 'query' field." });

    const clean = query.trim();
    res.setHeader("Content-Type", "application/json");

    const result = await Promise.race([
      deduplicate(`query:${clean}`, () => ragEngine.query(clean)),
      new Promise((_, rej) => setTimeout(() => rej(new Error("Request timeout")), 15000)),
    ]);

    res.json(result);
  } catch (err) {
    console.error("Query error:", err);
    res.status(500).json({ message: "Internal error.", error: err instanceof Error ? err.message : "Unknown" });
  }
});

app.get("/api/v1/stats", (_, res) => {
  res.setHeader("Cache-Control", "public, max-age=10");
  res.json({
    caches: {
      embeddings: cache["embeddingCache"].size,
      responses: cache["responseCache"].size,
      paraphrases: cache["paraphraseCache"].size,
    },
    performance: {
      pendingRequests: pendingRequests.size,
      embeddingQueue: embeddingQueue.length,
      mem: process.memoryUsage(),
    },
    uptime: process.uptime(),
  });
});

process.on("SIGTERM", () => {
  console.log("Shutting down gracefully...");
  pendingRequests.clear();
  process.exit(0);
});

setInterval(() => {
  const mem = process.memoryUsage();
  if (mem.heapUsed > 500 * 1024 * 1024) console.warn("High memory usage:", mem);
}, 30000);

if (!cluster.isPrimary) {
  const start = Date.now();
  app.listen(PORT, () => {
    console.log(`🚀 Worker ${process.pid} running on http://localhost:${PORT}`);
    console.log(`🎯 Startup time: ${Date.now() - start}ms`);
  });
}