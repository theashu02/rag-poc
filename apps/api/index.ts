import type { ScoredPineconeRecord } from "@pinecone-database/pinecone";
import { performance } from "perf_hooks";
import { generateGreetingResponse, generateSmallTalkResponse, isGreeting, isSmallTalk } from "./services/greeting";
import { PORT, PINECONE_INDEX, OPENAI_EMBEDDING_MODEL, OPENAI_CHAT_MODEL, EMBEDDING_DIM, PromptForGenerateAnswer } from "./services/config";
import { clean } from "./services/cleanQuery"
import bm25PrepTasks from "./services/BM25_config";
import { corsHeaders } from "./services/config";
import { pinecone } from "./services/pinecone";
import { openai } from "./services/openai";
import { generateAnswerOpenRouter, streamAnswer } from "./services/ThirdPartyOpenAI";
import bm25Search from "wink-bm25-text-search";
import { createHash } from "crypto";

// Production imports for optimization
import { LRUCache } from "lru-cache";
import pQueue from "p-queue";

// Global caches and optimization structures
const EMBEDDING_CACHE = new LRUCache<string, number[]>({
  max: 10000,
  ttl: 1000 * 60 * 60 * 2, // 2 hours
});

const SEMANTIC_CACHE = new LRUCache<
  string,
  {
    answer: string;
    sources: Array<{ id: string; source?: string; score: number }>;
    latencySec: number;
  }
>({
  max: 5000,
  ttl: 1000 * 60 * 30, // 30 minutes
});

const BM25_INSTANCES_CACHE = new LRUCache<string, any>({
  max: 100,
  ttl: 1000 * 60 * 60, // 1 hour
});

// Connection pools and queues for rate limiting
const embeddingQueue = new pQueue({
  concurrency: 10,
  interval: 1000,
  intervalCap: 50,
});

const chatQueue = new pQueue({
  concurrency: 5,
  interval: 1000,
  intervalCap: 20,
});

type Meta = Record<string, any>;
type Match = ScoredPineconeRecord<Meta>;

function generateCacheKey(query: string, namespace?: string): string {
  return createHash("sha256")
    .update(`${query}:${namespace || "default"}`)
    .digest("hex");
}

function generateSemanticCacheKey(query: string, namespace?: string): string {
  const normalized = query
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .trim();
  return createHash("sha256")
    .update(`${normalized}:${namespace || "default"}`)
    .digest("hex");
}

async function embed(text: string): Promise<number[]> {
  const cacheKey = createHash("sha256").update(text).digest("hex");
  const cached = EMBEDDING_CACHE.get(cacheKey);
  if (cached) return cached;

  return embeddingQueue.add(async () => {
    // Double-check cache inside queue to prevent duplicate requests
    const cachedInQueue = EMBEDDING_CACHE.get(cacheKey);
    if (cachedInQueue) return cachedInQueue;

    try {
      const { data } = await openai.embeddings.create({
        model: OPENAI_EMBEDDING_MODEL,
        input: await clean(text),
        dimensions: EMBEDDING_DIM,
      });

      const v = data?.[0]?.embedding;
      if (!v || !Array.isArray(v)) {
        throw new Error("Failed to get valid embedding from OpenAI");
      }

      EMBEDDING_CACHE.set(cacheKey, v);
      return v;
    } catch (error) {
      console.error("Embedding error:", error);
      throw new Error(
        `Embedding failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }) as Promise<number[]>;
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

// Optimized BM25 with caching
function bm25Rerank(
  query: string,
  docs: Array<{ id: string; text: string }>,
  namespace?: string
): Map<string, number> {
  const scores = new Map<string, number>();
  if (!docs.length) return scores;

  // Create cache key for BM25 instance
  const docIds = docs
    .map((d) => d.id)
    .sort()
    .join(",");
  const bm25CacheKey = createHash("sha256")
    .update(`${docIds}:${namespace || "default"}`)
    .digest("hex");

  let bm25 = BM25_INSTANCES_CACHE.get(bm25CacheKey);

  if (!bm25) {
    bm25 = bm25Search();
    bm25.defineConfig({ fldWeights: { text: 1 } });
    bm25.definePrepTasks(bm25PrepTasks);

    docs.forEach((d) => bm25.addDoc({ text: d.text }, d.id));
    bm25.consolidate();

    BM25_INSTANCES_CACHE.set(bm25CacheKey, bm25);
  }

  try {
    bm25
      .search(query, docs.length)
      .forEach(([id, score]: [string, number]) => scores.set(id, score));
  } catch (error) {
    console.error("BM25 search error:", error);
    // Fallback: equal scores for all docs
    docs.forEach((d) => scores.set(d.id, 1.0));
    return scores;
  }

  docs.forEach((d) => {
    if (!scores.has(d.id)) scores.set(d.id, 0);
  });

  return scores;
}

function blendScores(
  dense: Map<string, number>,
  text: Map<string, number>,
  alpha = 0.65
): Map<string, number> {
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

// Optimized retrieval with better error handling
async function retrieve(
  query: string,
  topK: number,
  namespace?: string
): Promise<Match[]> {
  const qv = await embed(query);
  const index = pinecone.Index(PINECONE_INDEX || "large-3072");

  const maxRetries = 2;
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      if (namespace && typeof (index as any).namespace === "function") {
        const namespacedIndex = (index as any).namespace(namespace);
        const res = await namespacedIndex.query({
          vector: qv,
          topK,
          includeMetadata: true,
        });
        return (res.matches || []) as Match[];
      }

      const filter = namespace ? { userId: { $eq: namespace } } : undefined;
      const res = await index.query({
        vector: qv,
        topK,
        includeMetadata: true,
        filter,
      });
      return (res.matches || []) as Match[];
    } catch (error) {
      lastError = error as Error;
      console.warn(`Retrieve attempt ${attempt + 1} failed:`, error);
      if (attempt < maxRetries - 1) {
        await new Promise((resolve) =>
          setTimeout(resolve, Math.pow(2, attempt) * 100)
        );
      }
    }
  }

  throw lastError || new Error("All retrieve attempts failed");
}

async function retrieveContext(
  question: string,
  namespace?: string
): Promise<{
  context: string;
  sources: Array<{ id: string; source?: string; score: number }>;
}> {
  const topK = 25;
  const maxContextChars = 5000;
  const maxDocs = 4;

  const denseMatches = await retrieve(question, topK, namespace);

  if (!denseMatches.length) {
    return { context: "", sources: [] };
  }

  const docs = denseMatches
    .map((m) => ({
      id: String(m.id),
      text: extractText(m.metadata || {}),
    }))
    .filter((d) => d.text.length > 10);

  const bm25Scores = bm25Rerank(question, docs, namespace);

  const denseScoreMap = new Map<string, number>();
  for (const m of denseMatches) {
    denseScoreMap.set(String(m.id), m.score ?? 0);
  }

  const blended = blendScores(denseScoreMap, bm25Scores, 0.65);

  const byId = new Map(denseMatches.map((m) => [String(m.id), m]));
  const ranked = [...blended.entries()]
    .map(([id, score]) => {
      const m = byId.get(id)!;
      return { ...m, score, id };
    })
    .filter((m) => {
      const text = extractText(m.metadata || {});
      return text.length > 20;
    })
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

  const chunks: string[] = [];
  const sources: Array<{ id: string; source?: string; score: number }> = [];
  let used = 0;

  for (const m of ranked.slice(0, maxDocs)) {
    const md = (m.metadata || {}) as Meta;
    const chunk = await clean(extractText(md));
    if (!chunk || chunk.length < 30) continue;

    if (used + chunk.length > maxContextChars && chunks.length >= 2) break;

    chunks.push(chunk);
    sources.push({
      id: String(m.id),
      score: Number(m.score ?? 0),
      source:
        (md.source as string) || (md.url as string) || md.file || "unknown",
    });
    used += chunk.length;

    if (used >= maxContextChars) break;
  }

  return { context: chunks.join("\n\n---\n\n"), sources };
}

// Main RAG function with comprehensive optimization
async function rag(question: string, namespace?: string) {
  if (isGreeting(question)) {
    return {
      answer: generateGreetingResponse(question),
      sources: [],
      latencySec: 0,
    };
  }

  if (isSmallTalk(question)) {
    return {
      answer: generateSmallTalkResponse(question),
      sources: [],
      latencySec: 0,
    };
  }

  const t0 = performance.now();

  // Check semantic cache first
  const semanticCacheKey = generateSemanticCacheKey(question, namespace);
  const cached = SEMANTIC_CACHE.get(semanticCacheKey);
  if (cached) {
    console.log("Semantic cache hit");
    return {
      ...cached,
      latencySec: (performance.now() - t0) / 1000,
    };
  }

  // Optimized parameters
  const topK = Math.min(25, 40); // Reduced from potential higher values
  const maxContextChars = 5000; // Reduced from 6000
  const maxDocs = 4; // Reduced from 5

  try {
    // Parallel execution where possible
    const [denseMatches] = await Promise.all([
      retrieve(question, topK, namespace),
    ]);

    if (!denseMatches.length) {
      const result = {
        answer:
          "I couldn't find relevant information to answer your question from the current knowledge base.",
        sources: [],
        latencySec: (performance.now() - t0) / 1000,
      };

      SEMANTIC_CACHE.set(semanticCacheKey, result);
      return result;
    }

    const docs = denseMatches
      .map((m) => ({
        id: String(m.id),
        text: extractText(m.metadata || {}),
      }))
      .filter((d) => d.text.length > 10); // Filter out very short texts early

    // Parallel BM25 reranking
    const [bm25Scores] = await Promise.all([
      Promise.resolve(bm25Rerank(question, docs, namespace)),
    ]);

    const denseScoreMap = new Map<string, number>();
    for (const m of denseMatches) {
      denseScoreMap.set(String(m.id), m.score ?? 0);
    }

    const blended = blendScores(denseScoreMap, bm25Scores, 0.65);

    // Optimized ranking and context building
    const byId = new Map(denseMatches.map((m) => [String(m.id), m]));
    const ranked = [...blended.entries()]
      .map(([id, score]) => {
        const m = byId.get(id)!;
        return { ...m, score, id };
      })
      .filter((m) => {
        const text = extractText(m.metadata || {});
        return text.length > 20; // More aggressive filtering
      })
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

    // Optimized context building with early termination
    const chunks: string[] = [];
    const sources: Array<{ id: string; source?: string; score: number }> = [];
    let used = 0;

    for (const m of ranked.slice(0, maxDocs)) {
      const md = (m.metadata || {}) as Meta;
      const chunk = await clean(extractText(md));
      if (!chunk || chunk.length < 30) continue;

      if (used + chunk.length > maxContextChars && chunks.length >= 2) break;

      chunks.push(chunk);
      sources.push({
        id: String(m.id),
        score: Number(m.score ?? 0),
        source:
          (md.source as string) || (md.url as string) || md.file || "unknown",
      });
      used += chunk.length;

      if (used >= maxContextChars) break;
    }

    const context = chunks.join("\n\n---\n\n");
    const answer = await generateAnswerOpenRouter(question, context);
    // const answer = await generateAnswer(question, context);

    const result = {
      answer:
        answer || "I don't have enough information in the provided context.",
      sources,
      latencySec: (performance.now() - t0) / 1000,
    };

    // Cache the result
    SEMANTIC_CACHE.set(semanticCacheKey, result);

    return result;
  } catch (error) {
    console.error("Error in RAG process:", error);
    const errorResult = {
      answer:
        "I encountered an error while processing your request. Please try again.",
      sources: [],
      latencySec: (performance.now() - t0) / 1000,
    };

    return errorResult;
  }
}

// streming rag
async function ragStreaming(
  question: string,
  namespace: string | undefined,
  onToken: (token: string) => void,
  onSources: (sources: Array<{ id: string; source?: string; score: number }>) => void
): Promise<string> {
  // Quick responses for greetings/small talk
  if (isGreeting(question)) {
    const response = generateGreetingResponse(question);
    onToken(response);
    onSources([]);
    return response;
  }

  if (isSmallTalk(question)) {
    const response = generateSmallTalkResponse(question);
    onToken(response);
    onSources([]);
    return response;
  }

  try {
    const { context, sources } = await retrieveContext(question, namespace);

    // Send sources first
    onSources(sources);

    if (!context) {
      const fallback = "I couldn't find relevant information to answer your question from the current knowledge base.";
      onToken(fallback);
      return fallback;
    }

    // Stream the answer
    return await streamAnswer(question, context, {
      onToken,
      onError: (error) => {
        console.error("Streaming error:", error);
      },
    });
  } catch (error) {
    console.error("Error in streaming RAG:", error);
    const errorMsg = "I encountered an error while processing your request.";
    onToken(errorMsg);
    return errorMsg;
  }
}


// Enhanced server with better error handling and monitoring
Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    const startTime = performance.now();

    // Enhanced CORS handling
    if (req.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    // Health check with cache stats
    if (url.pathname === "/api/v1/health" && req.method === "GET") {
      return new Response(
        JSON.stringify({
          status: "ok",
          cacheStats: {
            embeddingCache: {
              size: EMBEDDING_CACHE.size,
              max: EMBEDDING_CACHE.max,
            },
            semanticCache: {
              size: SEMANTIC_CACHE.size,
              max: SEMANTIC_CACHE.max,
            },
            bm25Cache: {
              size: BM25_INSTANCES_CACHE.size,
              max: BM25_INSTANCES_CACHE.max,
            },
          },
          queueStats: {
            embeddingQueue: embeddingQueue.size,
            chatQueue: chatQueue.size,
          },
        }),
        {
          headers: { "Content-Type": "application/json", ...corsHeaders },
          status: 200,
        }
      );
    }

    // Optimized query endpoint with request validation
    if (url.pathname === "/api/v1/query" && req.method === "POST") {
      try {
        const body = (await req.json()) as {
          query?: string;
          namespace?: string;
        };
        const { query, namespace } = body ?? {};

        // Enhanced validation
        if (!query || typeof query !== "string") {
          return new Response(
            JSON.stringify({
              message: "Missing or invalid 'query' parameter.",
            }),
            {
              headers: { "Content-Type": "application/json", ...corsHeaders },
              status: 400,
            }
          );
        }

        if (query.length > 1000) {
          return new Response(
            JSON.stringify({
              message: "Query too long. Maximum 1000 characters.",
            }),
            {
              headers: { "Content-Type": "application/json", ...corsHeaders },
              status: 400,
            }
          );
        }

        if (query.length < 3) {
          return new Response(
            JSON.stringify({
              message: "Query too short. Minimum 3 characters.",
            }),
            {
              headers: { "Content-Type": "application/json", ...corsHeaders },
              status: 400,
            }
          );
        }

        const cleanQuery = await clean(query);
        console.log("--- This is cleaned query ---",cleanQuery);

        // Optimized timeout with Promise.race
        const result = await Promise.race([
          rag(cleanQuery, namespace),
          new Promise(
            (_, reject) =>
              setTimeout(() => reject(new Error("Request timeout")), 25_000) // Reduced from 30s
          ),
        ]);

        const responseTime = performance.now() - startTime;
        console.log(`Query processed in ${responseTime.toFixed(2)}ms`);

        return new Response(JSON.stringify(result), {
          headers: {
            "Content-Type": "application/json",
            "X-Response-Time": responseTime.toFixed(2) + "ms",
            ...corsHeaders,
          },
          status: 200,
        });
      } catch (err) {
        const responseTime = performance.now() - startTime;
        console.error("Query error:", err);

        const errorMessage =
          err instanceof Error ? err.message : "Unknown error";
        const isTimeout = errorMessage.includes("timeout");

        return new Response(
          JSON.stringify({
            message: isTimeout
              ? "Request timed out. Please try a shorter query."
              : "Internal error occurred.",
            error: errorMessage,
            responseTime: responseTime.toFixed(2) + "ms",
          }),
          {
            headers: {
              "Content-Type": "application/json",
              "X-Response-Time": responseTime.toFixed(2) + "ms",
              ...corsHeaders,
            },
            status: isTimeout ? 408 : 500,
          }
        );
      }
    }

    // for the streaming
    if (url.pathname === "/api/v1/query/stream" && req.method === "POST") {
      try {
        const body = (await req.json()) as {
          query?: string;
          namespace?: string;
        };
        const { query, namespace } = body ?? {};

        if (!query || typeof query !== "string" || query.length < 3) {
          return new Response(
            JSON.stringify({ message: "Invalid query parameter" }),
            {
              headers: { "Content-Type": "application/json", ...corsHeaders },
              status: 400,
            }
          );
        }

        const cleanQuery = await clean(query);

        const stream = new ReadableStream({
          async start(controller) {
            const encoder = new TextEncoder();

            try {
              // Send initial event
              // controller.enqueue(
              //   encoder.encode(
              //     `data: ${JSON.stringify({ type: "start" })}\n\n`
              //   )
              // );

              let sourceSent = false;

              await ragStreaming(
                cleanQuery,
                namespace,
                (token) => {
                  // Stream each token
                  controller.enqueue(
                    encoder.encode(token)
                    // `data: ${JSON.stringify({ type: "token", content: token })}\n\n`
                  );
                },
                (sources) => {
                  // Send sources once
                  if (!sourceSent) {
                    controller.enqueue(
                      encoder.encode(
                        // `data: ${({ type: "sources", sources })}\n\n`
                      )
                    );
                    sourceSent = true;
                  }
                }
              );

              // Send completion event
              // controller.enqueue(
              //   encoder.encode(
              //     `data: ${JSON.stringify({ type: "done" })}\n\n`
              //   )
              // );
            } catch (error) {
              controller.enqueue(
                encoder.encode(
                  `data: ${JSON.stringify({ type: "error", message: error instanceof Error ? error.message : "Unknown error" })}\n\n`
                )
              );
            } finally {
              controller.close();
            }
          },
        });

        return new Response(stream, {
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform",
            Connection: "keep-alive",
            "X-Accel-Buffering": "no",
            ...corsHeaders,
          },
        });
      } catch (err) {
        console.error("Streaming error:", err);
        return new Response(
          JSON.stringify({
            message: "Streaming failed",
            error: err instanceof Error ? err.message : "Unknown error",
          }),
          {
            headers: { "Content-Type": "application/json", ...corsHeaders },
            status: 500,
          }
        );
      }
    }


    // Cache management endpoint (optional)
    if (url.pathname === "/api/v1/cache" && req.method === "DELETE") {
      EMBEDDING_CACHE.clear();
      SEMANTIC_CACHE.clear();
      BM25_INSTANCES_CACHE.clear();

      return new Response(
        JSON.stringify({ message: "Caches cleared successfully" }),
        {
          headers: { "Content-Type": "application/json", ...corsHeaders },
          status: 200,
        }
      );
    }

    return new Response(JSON.stringify({ message: "Not found" }), {
      headers: { "Content-Type": "application/json", ...corsHeaders },
      status: 404,
    });
  },
});

console.log(`🚀 Production RAG server listening on http://localhost:${PORT}`);
console.log(
  `✅ Caching enabled: Embedding(${EMBEDDING_CACHE.max}), Semantic(${SEMANTIC_CACHE.max}), BM25(${BM25_INSTANCES_CACHE.max})`
);
console.log(`⚡ Rate limiting: Embeddings(50/s), Chat(20/s)`);
