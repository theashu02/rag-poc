import { corsHeaders, PORT } from "./services/config";



Bun.serve({
    port: PORT,
    async fetch(req) {
        const url = new URL(req.url);
        
        if (req.method === 'OPTIONS') {
            return new Response(null, { headers: corsHeaders });
        }

        if (url.pathname === "/api/v2/health" && req.method === "GET") {
            return new Response(
              JSON.stringify({ message: "Api is up and running" }),
              {
                headers: { "Content-Type": "application/json", ...corsHeaders },
                status: 200,
              }
            );
        }

        if (url.pathname === "/api/v2/cache" && req.method === "DELETE") {
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
    }
})

console.log(`🚀 Production RAG server listening on http://localhost:${PORT}`);