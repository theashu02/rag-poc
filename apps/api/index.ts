import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import type { QueryResponse } from "@pinecone-database/pinecone";
import nlp from "compromise";
import { WordNet } from "natural";
import cluster from "cluster";
import os from "os";
import { LRUCache as LRU } from "lru-cache";

const cpuCount = os.cpus().length;
if (cluster.isPrimary) {
  console.log(`Primary ${process.pid} - forking ${cpuCount} workers`);
  for (let i = 0; i < cpuCount; i++) cluster.fork();
  cluster.on("exit", w => {
    console.log(`Worker ${w.process.pid} died – restarting`);
    cluster.fork();
  });
}

const app             = express();
const PORT            = Number(process.env.PORT) || 5000;
const OPENAI_KEY      = process.env.OPENAI_API_KEY;
const PINECONE_KEY    = process.env.PINECONE_API_KEY;
const PINECONE_INDEX  = process.env.PINECONE_INDEX;

if (!OPENAI_KEY)              throw new Error("Missing env OPENAI_API_KEY");
if (!PINECONE_KEY || !PINECONE_INDEX)
  throw new Error("Missing env PINECONE_API_KEY or PINECONE_INDEX");

const openai   = new OpenAI({ apiKey: OPENAI_KEY, timeout: 15_000, maxRetries: 2 });
const pinecone = new Pinecone({ apiKey: PINECONE_KEY });
const index    = pinecone.Index(PINECONE_INDEX);

app.use(cors({ origin: "http://localhost:3000" }));
app.use(express.json({ limit: "3mb" }));

/* -------------------- caches -------------------- */
class OptimizedCacheManager {
  private embeddingCache = new LRU<string, any[]>({ max: 2000, ttl: 60 * 60 * 1000 });
  private paraphraseCache= new LRU<string, any[]>({ max: 1000, ttl: 30 * 60 * 1000 });
  private synonymCache   = new LRU<string, string>({ max: 1500, ttl: 2 * 60 * 60 * 1000 });
  private responseCache  = new LRU<string, any[]>({ max: 500,  ttl: 10 * 60 * 1000 });

  getEmbedding(t: string)                { return this.embeddingCache.get(t)  || null; }
  setEmbedding(t: string, e: number[])   { this.embeddingCache.set(t, e); }
  getParaphrases(t: string)              { return this.paraphraseCache.get(t) || null; }
  setParaphrases(t: string, p: string[]) { this.paraphraseCache.set(t, p); }
  getSynonyms(t: string)                 { return this.synonymCache.get(t)    || null; }
  setSynonyms(t: string, s: string)      { this.synonymCache.set(t, s); }
  getResponse(k: string)                 { return this.responseCache.get(k)   || null; }
  setResponse(k: string, r: any)         { this.responseCache.set(k, r); }
}
const cache = new OptimizedCacheManager();

/* -------------------- batching for embeddings -------------------- */
const embeddingQueue: Array<{ text:string; resolve:(e:number[])=>void; reject:(e:any)=>void }> = [];
let   processingEmbeddings = false;

async function processEmbeddingQueue() {
  if (processingEmbeddings || !embeddingQueue.length) return;
  processingEmbeddings = true;
  const BATCH = 100;
  while (embeddingQueue.length) {
    const batch = embeddingQueue.splice(0, BATCH);
    try {
      const { data } = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: batch.map(b => b.text),
        dimensions: 1536,
      });
      batch.forEach((item,i) => {
        const emb = data[i]?.embedding || [];
        cache.setEmbedding(item.text, emb);
        item.resolve(emb);
      });
    } catch (err) {
      batch.forEach(b => b.reject(err));
    }
  }
  processingEmbeddings = false;
}

function embed(text: string): Promise<number[]> {
  const c = cache.getEmbedding(text);
  if (c) return Promise.resolve(c);
  return new Promise((resolve,reject)=>{
    embeddingQueue.push({ text, resolve, reject });
    (embeddingQueue.length <= 10 ? setImmediate : setTimeout)(processEmbeddingQueue, 50);
  });
}

/* -------------------- NLP helpers -------------------- */
class FastQueryProcessor {
  private wn = new WordNet();
  private static WORD = /\b\w{3,}\b/g;

  async expandWithSynonyms(q: string)/*…*/{ return q; } 
  async paraphrases(q: string)       /*…*/{ return []; }

  entities(t:string){ return nlp(t).topics().out("array"); }

  async intent(q:string){
    return { alternatives:[q] }; 
  }
}

/* -------------------- search + RAG -------------------- */
class HighPerformanceSearchEngine {
  private qp = new FastQueryProcessor();
  private searchCache = new LRU<string, QueryResponse>({ max:200, ttl:5*60*1000 });

  private async optimizedPineconeSearch(
    emb:number[], topK:number, ns?:string
  ){
    const key = `search:${ns}:${emb.slice(0,10).join(",")}:${topK}`;
    const c = this.searchCache.get(key); if (c) return c;
    const res = await index.query({
      vector: emb,
      topK,
      includeMetadata: true,
      namespace: ns || "default",
      filter: { userID: ns || "default" },
    } as any);
    this.searchCache.set(key,res as any);
    return res as any;
  }

  async search(q:string, topK=20, ns?:string){
    const { alternatives } = await this.qp.intent(q);
    const toEmbed = alternatives.slice(0,2);
    const embs    = await Promise.all(toEmbed.map(embed));
    const results = await Promise.all(embs.map(e=>this.optimizedPineconeSearch(e, topK, ns)));
    const matches = results.flatMap(r=>r.matches ?? []);
    const map = new Map<string, any>();
    matches.forEach(m=>{
      if (!map.has(m.id) || (m.score??0) > (map.get(m.id).score??0))
        map.set(m.id,m);
    });
    return [...map.values()].sort((a,b)=>(b.score??0)-(a.score??0));
  }
}

class UltraFastRAGEngine {
  private searcher = new HighPerformanceSearchEngine();
  private CONTEXT = 8_000;
  private MAX_DOC = 5;

  async query(question:string, ns?:string){
    const matches = await this.searcher.search(question,15,ns);
    if (!matches.length) return { answer:"No relevant info found.", sources:[] };

    let chars=0, ctx:string[]=[], sources:any[]=[];
    for (let i=0;i<Math.min(matches.length,this.MAX_DOC);i++){
      const m=matches[i]; const md=m.metadata||{} as any;
      const chunk = md.text||md.content||md.pageContent||"";
      if (!chunk) continue;
      if (chars+chunk.length>this.CONTEXT && ctx.length>=2) break;
      chars+=chunk.length;
      ctx.push(`[Doc ${i+1}] ${chunk}`);
      sources.push({ id:m.id, score:m.score, source:md.source||"unk" });
    }
    const context = ctx.join("\n\n");
    const completion = await openai.chat.completions.create({
      model:"gpt-4o-mini", temperature:0.2, max_tokens:400,
      messages:[
        { role:"system",
          content:`Answer strictly from provided context. Quote and cite [Doc #].` },
        { role:"user", content:`Context:\n${context}\n\nQ: ${question}\nA:` }
      ],
    });
    return {
      answer : completion.choices[0]?.message.content?.trim()||"Unable to answer.",
      sources
    };
  }
}
const ragEngine = new UltraFastRAGEngine();

/* -------------------- routes -------------------- */
app.get("/api/v1/health", (_req, res) => {
    res.json({ status: "API up", ts: Date.now() });
});

app.post("/api/v1/query", async (req,res)=>{
  try{
    const { query, namespace } = req.body??{};
    if (!query || typeof query!=="string" || query.length>500)
      return res.status(400).json({ message:"Invalid 'query' field." });

    const result = await ragEngine.query(query.trim(), namespace);
    res.json(result);
  }catch(e:any){
    console.error(e);
    res.status(500).json({ message:"Internal error", error:e?.message||"unknown" });
  }
});

/* -------------------- server start -------------------- */
if (!cluster.isPrimary){
  app.listen(PORT, ()=>console.log(`Worker ${process.pid} ➜ http://localhost:${PORT}`));
}