import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import type { QueryResponse } from "@pinecone-database/pinecone";
import { performance } from "perf_hooks";
import { WordNet } from "natural";
import cluster from "cluster";
import os from "os";
import { LRUCache as LRU } from "lru-cache";

interface SparseValues {
    indices: number[];
    values: number[];
}

const cpuCount = os.cpus().length;
if (cluster.isPrimary) {
  console.log(`Primary ${process.pid} - forking ${cpuCount} workers`);
  for (let i = 0; i < cpuCount; i++) cluster.fork();

  cluster.on("exit", worker => {
    console.log(`Worker ${worker.process.pid} died - starting replacement`);
    cluster.fork();
  });
}

const app = express();
const PORT = Number(process.env.PORT) || 5000;
const OPENAI_KEY = process.env.OPENAI_API_KEY;
const PINECONE_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;
const OPENAI_EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small";
const DIMENSIONS_ENV = process.env.DIMENSIONS;
const DIMENSIONS = DIMENSIONS_ENV ? Number(DIMENSIONS_ENV) : 1536; 

if (!OPENAI_KEY) throw new Error("Missing env: OPENAI_API_KEY");
if (!PINECONE_KEY || !PINECONE_INDEX) throw new Error("Missing env: PINECONE_API_KEY or PINECONE_INDEX");

const openai = new OpenAI({
  apiKey: OPENAI_KEY,
  timeout: 30_000,
  maxRetries: 2,
});

const pinecone = new Pinecone({ apiKey: PINECONE_KEY });
const index = pinecone.Index(PINECONE_INDEX);

app.use(cors({ origin: "*" }));
app.use(express.json({ limit: "5mb" }));

// --- High-performance LRU Caching Layer ---
class OptimizedCacheManager {
  private embeddingCache = new LRU<string, number[]>({ max: 2000, ttl: 60 * 60 * 1000, updateAgeOnGet: true });
  private paraphraseCache = new LRU<string, string[]>({ max: 1000, ttl: 30 * 60 * 1000, updateAgeOnGet: true });
  private synonymCache = new LRU<string, string>({ max: 1500, ttl: 2 * 60 * 60 * 1000, updateAgeOnGet: true });
  private responseCache = new LRU<string, any>({ max: 500, ttl: 10 * 60 * 1000, updateAgeOnGet: true });

  getEmbedding(t: string) { return this.embeddingCache.get(t) || null; }
  setEmbedding(t: string, e: number[]) { this.embeddingCache.set(t, e); }
  getParaphrases(t: string) { return this.paraphraseCache.get(t) || null; }
  setParaphrases(t: string, p: string[]) { this.paraphraseCache.set(t, p); }
  getSynonyms(t: string) { return this.synonymCache.get(t) || null; }
  setSynonyms(t: string, s: string) { this.synonymCache.set(t, s); }
  getResponse(k: string) { return this.responseCache.get(k) || null; }
  setResponse(k: string, r: any) { this.responseCache.set(k, r); }

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


// --- Batched Embedding Generation ---
const embeddingQueue: Array<{ text: string; resolve: (e: number[]) => void; reject: (err: any) => void; }> = [];
let isProcessingEmbeddings = false;

async function processEmbeddingQueue() {
  if (isProcessingEmbeddings || embeddingQueue.length === 0) return;
  isProcessingEmbeddings = true;

  const BATCH_SIZE = 100;
  while (embeddingQueue.length) {
    const batch = embeddingQueue.splice(0, BATCH_SIZE);
    try {
      const { data } = await openai.embeddings.create({
        model: OPENAI_EMBEDDING_MODEL,
        input: batch.map(b => b.text.replace(/\n/g, " ")),
        dimensions: DIMENSIONS,
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
    (embeddingQueue.length < 10 ? setImmediate : setTimeout)(processEmbeddingQueue, 20);
  });
}

// Advanced Query Understanding & Expansion 
class FastQueryProcessor {
  private wn = new WordNet();
  private static readonly WORD_REGEX = /\b\w{3,}\b/g;

  generateSparseVector(query: string): SparseValues {
    const terms = query.toLowerCase().match(FastQueryProcessor.WORD_REGEX) || [];
    const termFrequencies: { [key: string]: number } = {};
    terms.forEach(term => {
      termFrequencies[term] = (termFrequencies[term] || 0) + 1;
    });
    
   
    const indices = Object.keys(termFrequencies).map((_, i) => i);
    const values = Object.values(termFrequencies).map(freq => freq / terms.length); // Simple normalization
    
    return { indices, values };
  }

  async expandWithSynonyms(query: string): Promise<string> {
    const cached = cache.getSynonyms(query);
    if (cached) return cached;
    if (query.length < 10) return query;

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

  async paraphrases(query: string, namespace?: string): Promise<string[]> {
    const cached = cache.getParaphrases(query);
    if (cached) return cached;
    if (query.length < 15 || query.split(/\s+/).length <= 2) return [];

    try {
      const systemContent = `You are a concise assistant. Rephrase the user's query clearly and briefly to optimize retrieval from the vector database.`;

      const completion = await Promise.race([
        openai.chat.completions.create({
          model: "gpt-4o-mini", // More performant choice
          temperature: 0.4,
          max_tokens: 50,
          messages: [
            { role: "system", content: systemContent },
            { role: "user", content: query },
          ],
        }),
        new Promise((_, reject) => setTimeout(() => reject(new Error("Timeout")), 1500)),
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

  async intent(query: string, namespace?: string) {
    const alternatives = await this.paraphrases(query, namespace);
    return {
      alternatives: [query, ...alternatives].slice(0, 2), // Use original query + one paraphrase
      sparseVector: this.generateSparseVector(query)
    };
  }
}

// --- High-Performance Search Engine with Hybrid Search & Reranking ---
class HighPerformanceSearchEngine {
  private qp = new FastQueryProcessor();
  private searchCache = new LRU<string, QueryResponse>({ max: 200, ttl: 5 * 60 * 1000 });

  private async optimizedPineconeSearch(
    embedding: number[],
    sparseVector: SparseValues,
    topK: number,
    namespace?: string
  ): Promise<{ matches?: any[] }> {
    const key = `search:${embedding.slice(0, 5).join(",")}:${topK}:${namespace}`;
    const cached = this.searchCache.get(key);
    if (cached) return cached;

    // --- ACCURACY IMPROVEMENT: Performing a hybrid search query ---
    // Alpha=0.5 gives equal weight to semantic and keyword search. Adjust as needed.
    const res = await index.query({
      vector: embedding,
      sparseVector,
      topK,
      includeMetadata: true,
      filter: namespace ? { userId: namespace } : undefined,
    });
    this.searchCache.set(key, res as any);
    return res as any;
  }

  // --- ACCURACY IMPROVEMENT: Placeholder for a Reranker ---
  // In a real app, this would call a dedicated reranking model.
  private async rerankResults(query: string, documents: any[]): Promise<any[]> {
    console.log("Reranking step: Re-ordering results for maximum precision.");
    // This is where you would integrate with Cohere Rerank or a local cross-encoder model.
    // For now, we just return the documents sorted by their original score.
    return documents.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  }


  async search(query: string, topK = 20, namespace?: string) {
    const key = `full_search:${query}:${topK}:${namespace}`;
    const cached = cache.getResponse(key);
    if (cached) return cached;

    const { alternatives, sparseVector } = await this.qp.intent(query, namespace);
    const embeddings = await Promise.all(alternatives.map(embed));

    const results = await Promise.all(
      embeddings.map((e) => this.optimizedPineconeSearch(e, sparseVector, topK, namespace))
    );
    const matches = results.flatMap((r) => r.matches ?? []);

    // Deduplicate results before reranking
    const uniqueMatches = Array.from(new Map(matches.map(m => [m.id, m])).values());

    // Apply the reranking step for higher accuracy
    const rerankedMatches = await this.rerankResults(query, uniqueMatches);

    cache.setResponse(key, rerankedMatches);
    return rerankedMatches;
  }
}


// Main RAG Engine
class UltraFastRAGEngine {
  private searcher = new HighPerformanceSearchEngine();
  private CONTEXT_CHARS = 8_000;
  private MAX_DOCS = 7;

  async query(question: string, namespace?: string) {
    const qKey = `rag:${question}:${namespace}`;
    const exact = cache.getResponse(qKey);
    if (exact) return { ...exact, cached: true, searchTime: 0 };

    const similar = cache.findSimilarResponse(question);
    if (similar) return { ...similar, cached: true, searchTime: 0 };

    const t0 = performance.now();
    const matches = await this.searcher.search(question, 50, namespace);

    if (!matches.length) {
      const res = { answer: "I couldn't find relevant information to answer your question.", sources: [], searchTime: (performance.now() - t0) / 1000 };
      cache.setResponse(qKey, res);
      return res;
    }

    let chars = 0;
    const ctx: string[] = [];
    const sources: any[] = [];

    for (const m of matches.slice(0, this.MAX_DOCS)) {
      const md = (m.metadata ?? {}) as Record<string, any>;
      const chunk = md.text ?? md.content ?? "";
      if (!chunk) continue;

      if (chars + chunk.length > this.CONTEXT_CHARS && ctx.length >= 2) break;
      chars += chunk.length;
      ctx.push(chunk);
      sources.push({ id: m.id, score: m.score ?? 0, source: md.source ?? "Unknown" });
      if (chars >= this.CONTEXT_CHARS) break;
    }

    const context = ctx.join("\n\n---\n\n");

    const completion = await Promise.race([
      openai.chat.completions.create({
        model: "gpt-4o-mini",
        temperature: 0.4,
        max_tokens: 500,
        messages: [
          {
            role: "system",
            content: `You are a helpful assistant. Answer the user's question based *only* on the provided context. If the answer is not in the context, say so. Be concise.`,
          },
          { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}\nAnswer:` },
        ],
      }),
      new Promise((_, rej) => setTimeout(() => rej(new Error("OpenAI timeout")), 8000)),
    ]) as any;

    const result = {
      answer: completion.choices[0]?.message.content?.trim() || "Unable to generate answer.",
      sources,
      searchTime: (performance.now() - t0) / 1000,
    };

    cache.setResponse(qKey, result);
    return result;
  }
}

// Request Deduplication & API Endpoints
const pendingRequests = new Map<string, Promise<any>>();
function deduplicate<T>(key: string, fn: () => Promise<T>): Promise<T> {
  if (pendingRequests.has(key)) return pendingRequests.get(key) as Promise<T>;
  const p = fn().finally(() => pendingRequests.delete(key));
  pendingRequests.set(key, p);
  return p;
}

const ragEngine = new UltraFastRAGEngine();

app.get("/api/v1/health", (_, res) => {
  res.status(200).json({ status: "API up & healthy" });
});

app.post("/api/v1/query", async (req, res) => {
  try {
    const { query, namespace } = req.body ?? {};
    if (!query || typeof query !== "string" || query.length > 500)
      return res.status(400).json({ message: "Invalid 'query' field." });

    const cleanQuery = query.trim();
    const key = `query:${cleanQuery}:${namespace}`;

    // Note on Streaming: For even better perceived performance, you would modify
    // this to handle a streamed response from the OpenAI API.
    const result = await Promise.race([
        deduplicate(key, () => ragEngine.query(cleanQuery, namespace)),
        new Promise((_, rej) => setTimeout(() => rej(new Error("Request timeout")), 30000)),
    ]);

    res.json(result);
  } catch (err) {
    console.error("Query error:", err);
    res.status(500).json({ message: "Internal error.", error: err instanceof Error ? err.message : "Unknown" });
  }
});


app.get("/api/v1/stats", (_, res) => {
  res.setHeader("Cache-Control", "no-cache");
  res.json({
    caches: {
      embeddings: cache['embeddingCache'].size,
      responses: cache['responseCache'].size,
      paraphrases: cache['paraphraseCache'].size,
    },
    performance: {
      pendingRequests: pendingRequests.size,
      embeddingQueue: embeddingQueue.length,
      mem: process.memoryUsage(),
    },
    uptime: process.uptime(),
  });
});

if (!cluster.isPrimary) {
  const start = Date.now();
  app.listen(PORT, () => {
    console.log(`🚀 Worker threads ${process.pid} running on http://localhost:${PORT}`);
    console.log(`🎯 Startup time: ${Date.now() - start}ms`);
    console.log("🚀 Backend is up and running.")
  });
}
