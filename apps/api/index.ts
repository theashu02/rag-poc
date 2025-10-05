import { performance } from "perf_hooks";
import { PORT } from "./services/config";
import { clean } from "./services/cleanQuery"
import { corsHeaders } from "./services/config";
import { EMBEDDING_CACHE, BM25_INSTANCES_CACHE, embeddingQueue, chatQueue } from "./services/config";
import { rag } from "./services/rag";
import { SEMANTIC_CACHE } from "./services/cache";

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

