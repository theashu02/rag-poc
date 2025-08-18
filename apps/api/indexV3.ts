import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { performance } from "perf_hooks";
import keywordExtractor from "keyword-extractor";
import nlp from "compromise";
import { WordNet } from "natural";

const app = express();
const PORT = Number(process.env.PORT) || 5000;
const OPENAI_KEY = process.env.OPENAI_API_KEY;
const PINECONE_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;

if (!OPENAI_KEY) throw new Error("Missing env: OPENAI_API_KEY");
if (!PINECONE_KEY || !PINECONE_INDEX)
  throw new Error("Missing env: PINECONE_API_KEY or PINECONE_INDEX");

const openai = new OpenAI({ apiKey: OPENAI_KEY });
const pinecone = new Pinecone({ apiKey: PINECONE_KEY });
const index = pinecone.Index(PINECONE_INDEX);

app.use(cors({ origin: "http://localhost:3000" }));
app.use(express.json());

// In-memory caches with TTL
class CacheManager {
  private embeddingCache = new Map<string, { embedding: number[]; timestamp: number }>();
  private paraphraseCache = new Map<string, { paraphrases: string[]; timestamp: number }>();
  private synonymCache = new Map<string, { synonyms: string; timestamp: number }>();
  
  private readonly CACHE_TTL = 30 * 60 * 1000; // 30 minutes
  private readonly MAX_CACHE_SIZE = 1000;

  private isExpired(timestamp: number): boolean {
    return Date.now() - timestamp > this.CACHE_TTL;
  }

  private cleanupCache(cache: Map<string, any>) {
    if (cache.size > this.MAX_CACHE_SIZE) {
      const entries = Array.from(cache.entries());
        const toRemove = Math.min(entries.length, Math.floor(this.MAX_CACHE_SIZE * 0.2));
        entries.sort((a,b) => a[1].timestamp - b[1].timestamp);
        for(let i = 0; i < toRemove; i++){
            const entry = entries[i];
            if(!entry) continue;
            cache.delete(entry[0]);
        }
    }
  }

  getEmbedding(text: string): number[] | null {
    const cached = this.embeddingCache.get(text);
    if (cached && !this.isExpired(cached.timestamp)) {
      return cached.embedding;
    }
    if (cached) this.embeddingCache.delete(text);
    return null;
  }

  setEmbedding(text: string, embedding: number[]) {
    this.cleanupCache(this.embeddingCache);
    this.embeddingCache.set(text, { embedding, timestamp: Date.now() });
  }

  getParaphrases(text: string): string[] | null {
    const cached = this.paraphraseCache.get(text);
    if (cached && !this.isExpired(cached.timestamp)) {
      return cached.paraphrases;
    }
    if (cached) this.paraphraseCache.delete(text);
    return null;
  }

  setParaphrases(text: string, paraphrases: string[]) {
    this.cleanupCache(this.paraphraseCache);
    this.paraphraseCache.set(text, { paraphrases, timestamp: Date.now() });
  }

  getSynonyms(text: string): string | null {
    const cached = this.synonymCache.get(text);
    if (cached && !this.isExpired(cached.timestamp)) {
      return cached.synonyms;
    }
    if (cached) this.synonymCache.delete(text);
    return null;
  }

  setSynonyms(text: string, synonyms: string) {
    this.cleanupCache(this.synonymCache);
    this.synonymCache.set(text, { synonyms, timestamp: Date.now() });
  }
}

const cache = new CacheManager();

// Optimized embedding helper with caching and batching
async function embed(text: string): Promise<number[]> {
  // Check cache first
  const cached = cache.getEmbedding(text);
  if (cached) return cached;

  const { data } = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  
  const embedding = data[0]?.embedding || [];
  cache.setEmbedding(text, embedding);
  return embedding;
}

// Batch embedding function for multiple texts
async function batchEmbed(texts: string[]): Promise<number[][]> {
  // Filter out cached embeddings
  const uncachedTexts: string[] = [];
  const cachedResults: (number[] | null)[] = [];
  
  for (const text of texts) {
    const cached = cache.getEmbedding(text);
    if (cached) {
      cachedResults.push(cached);
    } else {
      cachedResults.push(null);
      uncachedTexts.push(text);
    }
  }

  // Batch process uncached embeddings
  let uncachedEmbeddings: number[][] = [];
  if (uncachedTexts.length > 0) {
    const { data } = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: uncachedTexts,
    });
    uncachedEmbeddings = data.map(d => d.embedding);
    
    // Cache the new embeddings
    uncachedTexts.forEach((text, idx) => {
      cache.setEmbedding(text, uncachedEmbeddings[idx]!);
    });
  }

  // Merge cached and new results
  const results: number[][] = [];
  let uncachedIndex = 0;
  
  for (let i = 0; i < cachedResults.length; i++) {
    if (cachedResults[i] !== null) {
      results.push(cachedResults[i] as number[]);
    } else {
      results.push(uncachedEmbeddings[uncachedIndex++]!);
    }
  }
  
  return results;
}

// Optimized QueryProcessor
class QueryProcessor {
  private wn = new WordNet();

  /* synonyms via WordNet with caching */
  async expandWithSynonyms(q: string): Promise<string> {
    // Check cache first
    const cached = cache.getSynonyms(q);
    if (cached) return cached;

    const toks = q.toLowerCase().split(/\s+/);
    const out: string[] = [];

    // Process all tokens in parallel
    const synonymPromises = toks.map(t => 
      new Promise<string[]>((resolve) =>
        this.wn.lookup(t, (rows) => {
          const s = rows
            .flatMap((r: any) => r.synonyms)
            .filter((w: string) => w !== t)
            .slice(0, 2);
          resolve(s);
        })
      )
    );

    const synonymResults = await Promise.all(synonymPromises);
    
    for (let i = 0; i < toks.length; i++) {
      out.push(toks[i]!);
      out.push(...synonymResults[i]!);
    }

    const result = out.join(" ");
    cache.setSynonyms(q, result);
    return result;
  }

  /* GPT-4 paraphrases with caching and reduced calls */
  async paraphrases(q: string): Promise<string[]> {
    // Check cache first
    const cached = cache.getParaphrases(q);
    if (cached) return cached;

    try {
      const { choices } = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        temperature: 0.7,
        max_tokens: 80, // Reduced from 100
        messages: [
          {
            role: "system",
            content: "Generate 2 alternative phrasings of the query. Return one per line.",
          },
          { role: "user", content: q },
        ],
      });
      
      const paraphrases = choices[0]?.message.content
        ?.trim()
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean)
        .slice(0, 2) || []; // Reduced from 3 to 2
      
      cache.setParaphrases(q, paraphrases);
      return paraphrases;
    } catch {
      return [];
    }
  }

  entities(text: string): string[] {
    return Array.from(
      new Set(
        nlp(text)
          .topics()
          .out("array")
          .concat(
            nlp(text).people().out("array"),
            nlp(text).places().out("array")
          )
      )
    );
  }

  keywords(text: string, top = 10): string[] {
    return keywordExtractor.extract(text, {
      language: "english",
      remove_duplicates: true,
    }) as string[];
  }

  /* Optimized intent processing */
  async intent(query: string) {
    // For simple queries, skip expensive operations
    const isSimpleQuery = query.length < 20 && query.split(/\s+/).length <= 3;
    
    if (isSimpleQuery) {
      // Fast path for simple queries
      return {
        expanded: query,
        alternatives: [query],
        entities: this.entities(query),
        keywords: this.keywords(query, 10),
      };
    }

    // Parallel processing for complex queries
    const [syn, alts] = await Promise.all([
      this.expandWithSynonyms(query),
      this.paraphrases(query),
    ]);
    
    return {
      expanded: syn,
      alternatives: [query, ...alts].slice(0, 3),
      entities: this.entities(query),
      keywords: this.keywords(query, 10),
    };
  }
}

// Optimized HybridSearchEngine
class HybridSearchEngine {
  private qp = new QueryProcessor();

  private async denseSearch(queries: string[], topK: number) {
    // Batch embed all queries at once
    const embeddings = await batchEmbed(queries);
    
    // Parallel Pinecone searches
    const searchPromises = embeddings.map(embedding =>
      index.query({
        vector: embedding,
        topK,
        includeMetadata: true,
      })
    );

    const results = await Promise.all(searchPromises);
    return results.map(res => res.matches ?? []);
  }

  /* Optimized multi-query search with parallel processing */
  async search(query: string, topK = 25) {
    const { alternatives } = await this.qp.intent(query);
    
    // Remove duplicates to avoid unnecessary API calls
    const uniqueQueries = Array.from(new Set(alternatives));
    
    // Single batch call for all embeddings and searches
    const allMatches = await this.denseSearch(uniqueQueries, topK);
    
    // Fusion: keep best score for each document
    const best = new Map<
      string,
      { match: any; queryIndex: number }
    >();

    allMatches.forEach((matches, queryIndex) => {
      matches.forEach(m => {
        const prev = best.get(m.id);
        if (!prev || (m.score ?? 0) > (prev.match.score ?? 0)) {
          best.set(m.id, { match: m, queryIndex });
        }
      });
    });

    return Array.from(best.values())
      .map(item => item.match)
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  }
}

// Optimized RAGQueryEngine
class RAGQueryEngine {
  private searcher = new HybridSearchEngine();
  private CONTEXT_CHARS = 12_000; // Reduced from 16k for faster processing
  private responseCache = new Map<string, { result: any; timestamp: number }>();
  private readonly RESPONSE_CACHE_TTL = 5 * 60 * 1000; // 5 minutes

  private getCachedResponse(question: string) {
    const cached = this.responseCache.get(question);
    if (cached && Date.now() - cached.timestamp < this.RESPONSE_CACHE_TTL) {
      return cached.result;
    }
    if (cached) this.responseCache.delete(question);
    return null;
  }

  private setCachedResponse(question: string, result: any) {
    // Clean up old cache entries
    if (this.responseCache.size > 100) {
      const entries = Array.from(this.responseCache.entries());
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
      for (let i = 0; i < 20; i++) {
        // this.responseCache.delete(entries[i][0]);
        const entry = entries[i];
        if(!entry) continue;
        this.responseCache.delete(entry[0]);
      }
    }
    this.responseCache.set(question, { result, timestamp: Date.now() });
  }

  async query(question: string) {
    // Check response cache first
    const cachedResponse = this.getCachedResponse(question);
    if (cachedResponse) {
      return {
        ...cachedResponse,
        cached: true,
        searchTime: 0,
      };
    }

    const t0 = performance.now();
    
    // Early similarity check to avoid unnecessary processing
    const questionLower = question.toLowerCase();
    for (const [cachedQ, cached] of this.responseCache.entries()) {
      if (this.calculateSimilarity(questionLower, cachedQ.toLowerCase()) > 0.85) {
        return {
          ...cached.result,
          cached: true,
          searchTime: (performance.now() - t0) / 1000,
        };
      }
    }

    const matches = await this.searcher.search(question, 40); // Reduced from 60

    if (!matches.length) {
      const result = {
        answer: "I couldn't find relevant information to answer your question.",
        sources: [],
        searchTime: (performance.now() - t0) / 1000,
      };
      this.setCachedResponse(question, result);
      return result;
    }

    // Optimized context building with early termination
    let chars = 0;
    const ctxParts: string[] = [];
    const sources: any[] = [];
    const maxDocuments = 8; // Limit documents for faster processing

    for (let i = 0; i < Math.min(matches.length, maxDocuments); i++) {
      const m = matches[i];
      if (!m) continue;
      
      const md = (m.metadata ?? {}) as Record<string, any>;
      const chunk = md.text ?? md.content ?? md.pageContent ?? md.page_content ?? "";
      if (!chunk) continue;
      
      // Early termination if we have enough context
      if (chars + chunk.length > this.CONTEXT_CHARS && ctxParts.length >= 3) break;
      
      chars += chunk.length;
      ctxParts.push(`[Document ${i + 1}]\n${chunk}`);
      sources.push({
        id: m.id,
        score: m.score ?? 0,
        source: md.source ?? "Unknown",
        chunkIndex: md.chunk_index ?? 0,
      });

      if (chars >= this.CONTEXT_CHARS) break;
    }

    const context = ctxParts.join("\n\n");
    
    // Optimized completion call with reduced tokens
    const { choices } = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.3,
      max_tokens: 800, // Reduced from 1000
      messages: [
        {
          role: "system",
          content: "Answer concisely based ONLY on the provided context. Use [Document X] for citations.",
        },
        {
          role: "user",
          content: `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer:`,
        },
      ],
    });

    const result = {
      answer: choices[0]?.message.content?.trim() || "",
      sources,
      searchTime: (performance.now() - t0) / 1000,
      documentsUsed: ctxParts.length,
      totalRetrieved: matches.length,
      contextChars: chars,
    };

    // Cache the result
    this.setCachedResponse(question, result);
    return result;
  }

  // Simple similarity calculation for cache lookup optimization
  private calculateSimilarity(str1: string, str2: string): number {
    const words1 = new Set(str1.split(/\s+/));
    const words2 = new Set(str2.split(/\s+/));
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);
    return intersection.size / union.size;
  }
}

// Batch processing optimized RAGEvaluator
class RAGEvaluator {
  private engine = new RAGQueryEngine();

  async evalRetrieval(
    tests: { question: string; expected: string[] }[]
  ): Promise<{ avg_precision: number; details: any[] }> {
    // Process evaluations with limited concurrency to avoid rate limits
    const BATCH_SIZE = 3;
    const results: any[] = [];

    for (let i = 0; i < tests.length; i += BATCH_SIZE) {
      const batch = tests.slice(i, i + BATCH_SIZE);
      const batchPromises = batch.map(async (t) => {
        const { sources } = await this.engine.query(t.question);
        const ret = new Set(sources.map((s: any) => s.source));
        const exp = new Set(t.expected);
        const precision = exp.size === 0 
          ? 0 
          : [...exp].filter((x) => ret.has(x)).length / ret.size;
        
        return {
          question: t.question,
          precision,
          retrieved: Array.from(ret),
          expected: t.expected,
        };
      });

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
      
      // Small delay between batches to avoid overwhelming APIs
      if (i + BATCH_SIZE < tests.length) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    const avgPrecision = results.reduce((sum, r) => sum + r.precision, 0) / (results.length || 1);
    
    return {
      avg_precision: avgPrecision,
      details: results,
    };
  }
}

// API routes with request deduplication
const ragEngine = new RAGQueryEngine();
const evaluator = new RAGEvaluator();

// Request deduplication middleware
const pendingRequests = new Map<string, Promise<any>>();

function deduplicate<T>(key: string, fn: () => Promise<T>): Promise<T> {
  if (pendingRequests.has(key)) {
    return pendingRequests.get(key) as Promise<T>;
  }
  
  const promise = fn().finally(() => {
    pendingRequests.delete(key);
  });
  
  pendingRequests.set(key, promise);
  return promise;
}

/* health */
app.get("/api/v1/health", (_, res) =>
  res.status(200).json({ status: "API up & healthy" })
);

/* optimized query with deduplication */
app.post("/api/v1/query", async (req, res) => {
  try {
    const { query } = req.body ?? {};
    if (!query || typeof query !== "string")
      return res.status(400).json({ message: "Invalid 'query' field." });

    // Deduplicate identical concurrent requests
    const result = await deduplicate(`query:${query}`, () => ragEngine.query(query));
    return res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Internal error." });
  }
});

/* optimized retrieval evaluation */
app.post("/api/v1/evaluate/retrieval", async (req, res) => {
  try {
    const { tests = [] } = req.body ?? {};
    
    if (tests.length > 50) {
      return res.status(400).json({ 
        message: "Too many tests. Maximum 50 tests allowed per evaluation." 
      });
    }
    
    const metrics = await evaluator.evalRetrieval(tests);
    res.json(metrics);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Evaluation failed" });
  }
});

// Cache statistics endpoint (optional)
app.get("/api/v1/cache/stats", (_, res) => {
  res.json({
    embeddingCacheSize: cache['embeddingCache'].size,
    paraphraseCacheSize: cache['paraphraseCache'].size,
    synonymCacheSize: cache['synonymCache'].size,
    pendingRequests: pendingRequests.size,
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Received SIGTERM, shutting down gracefully...');
  process.exit(0);
});

app.listen(PORT, () =>
  console.log(`🚀  Optimized RAG API listening on http://localhost:${PORT}`)
);