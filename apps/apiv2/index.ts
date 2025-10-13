import { clean } from "./services/clean";
import { corsHeaders, PORT } from "./services/config";
import { searchWithVectors } from "./services/hybridRetrieve/hybridQuery";
import { generateVectors } from "./services/vectorGenerator";



Bun.serve({
    port: PORT,
    async fetch(req) {
      const url = new URL(req.url);
        
      if (req.method === 'OPTIONS') {
        return new Response(null, { headers: corsHeaders });
      }

      if (url.pathname === "/api/v1/health" && req.method === "GET") {
        return new Response(
          JSON.stringify({ message: "Api is up and running" }),
          {
            headers: { "Content-Type": "application/json", ...corsHeaders },
            status: 200,
          }
        );
      }

      if (url.pathname === "/api/v1/cache" && req.method === "DELETE") {
        return new Response(
          JSON.stringify({ message: "Caches cleared successfully" }),
          {
            headers: { "Content-Type": "application/json", ...corsHeaders },
            status: 200,
          }
        );
      }

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
          console.log(cleanQuery, "----1");
  
          const vectors = await generateVectors(cleanQuery, 'query', 3);

          const HybridQuery = await searchWithVectors(vectors, namespace as string, 10);
          
          console.log("==== Hybrid query result ====",HybridQuery);
          return new Response(JSON.stringify("result"), {
            headers: {
              "Content-Type": "application/json",
              ...corsHeaders,
            },
            status: 200,
          });
        } catch (err) {
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
            }),
            {
              headers: {
                "Content-Type": "application/json",
                ...corsHeaders,
              },
              status: isTimeout ? 408 : 500,
            }
          );
        }
      }
      
      return new Response(JSON.stringify({ message: "Not found" }), {
        headers: { "Content-Type": "application/json", ...corsHeaders },
        status: 404,
      });
    }
})

console.log(`🚀 Production RAG server listening on http://localhost:${PORT}`);