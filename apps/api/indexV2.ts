import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { performance } from "perf_hooks";
// Additional NLP helpers (tiny)
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

// OpenAI embedding helper
async function embed(text: string): Promise<number[]> {
  const { data } = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return data[0]?.embedding || [];
}

// QueryProcessor
class QueryProcessor {
  private wn = new WordNet();

  /* synonyms via WordNet (max 2 per token) */
  async expandWithSynonyms(q: string): Promise<string> {
    const toks = q.toLowerCase().split(/\s+/);
    const out: string[] = [];

    for (const t of toks) {
      out.push(t);
      const syns: string[] = await new Promise((resolve) =>
        this.wn.lookup(t, (rows) => {
          const s = rows
            .flatMap((r: any) => r.synonyms)
            .filter((w: string) => w !== t)
            .slice(0, 2);
          resolve(s);
        })
      );
      out.push(...syns);
    }
    return out.join(" ");
  }

  /* GPT-4 paraphrases (best-effort) */
  async paraphrases(q: string): Promise<string[]> {
    try {
      const { choices } = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        temperature: 0.7,
        max_tokens: 100,
        messages: [
          {
            role: "system",
            content:
              "Generate 3 alternative phrasings of the query. Return one per line.",
          },
          { role: "user", content: q },
        ],
      });
      return (
        choices[0]?.message.content
          ?.trim()
          .split("\n")
          .map((s) => s.trim())
          .filter(Boolean)
          .slice(0, 3) || []
      );
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

  /* ---- public API ---- */
  async intent(query: string) {
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
    // if api takes time so uncomment this run on simple oroginal query
    // const syn = await this.expandWithSynonyms(query);
    // return {
    //   expanded: syn,
    //   alternatives: [query],
    //   entities: this.entities(query),
    //   keywords: this.keywords(query, 10),
    // };
  }
}

// HybridSearchEngine
class HybridSearchEngine {
  private qp = new QueryProcessor();

  private async denseSearch(q: string, topK: number) {
    const v = await embed(q);
    const res = await index.query({
      vector: v,
      topK,
      includeMetadata: true,
    });
    return res.matches ?? [];
  }

  /* multi-query search + best-score fusion */
  async search(query: string, topK = 50) {
    const { alternatives } = await this.qp.intent(query);
    const best = new Map<
      string,
      Awaited<ReturnType<typeof this.denseSearch>>[0]
    >();

    for (const q of alternatives) {
      for (const m of await this.denseSearch(q, topK)) {
        const prev = best.get(m.id);
        if (!prev || (m.score ?? 0) > (prev.score ?? 0)) best.set(m.id, m);
      }
    }
    return Array.from(best.values()).sort(
      (a, b) => (b.score ?? 0) - (a.score ?? 0)
    );
  }
}

// RAGQueryEngine
class RAGQueryEngine {
  private searcher = new HybridSearchEngine();
  private CONTEXT_CHARS = 16_000; // ~8k tokens

  async query(question: string) {
    const t0 = performance.now();
    const matches = await this.searcher.search(question, 60);

    if (!matches.length)
      return {
        answer: "I couldn't find relevant information to answer your question.",
        sources: [],
        searchTime: (performance.now() - t0) / 1000,
      };

    let chars = 0;
    const ctxParts: string[] = [];
    const sources: any[] = [];

    for (let i = 0; i < matches.length; i++) {
      const m = matches[i];
      if (!m) continue;
      const md = (m.metadata ?? {}) as Record<string, any>;
      const chunk =
        md.text ?? md.content ?? md.pageContent ?? md.page_content ?? "";
      if (!chunk) continue;
      if (chars + chunk.length > this.CONTEXT_CHARS) break;
      chars += chunk.length;

      ctxParts.push(`[Document ${i + 1}]\n${chunk}`);
      sources.push({
        id: m.id,
        score: m.score ?? 0,
        source: md.source ?? "Unknown",
        chunkIndex: md.chunk_index ?? 0,
      });
    }

    const context = ctxParts.join("\n\n");
    const { choices } = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.3,
      max_tokens: 1000,
      messages: [
        {
          role: "system",
          content:
            "Answer based ONLY on the provided context. Use [Document X] for citations.",
        },
        {
          role: "user",
          content: `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer:`,
        },
      ],
    });

    return {
      answer: choices[0]?.message.content?.trim() || "",
      sources,
      searchTime: (performance.now() - t0) / 1000,
      documentsUsed: ctxParts.length,
      totalRetrieved: matches.length,
      contextChars: chars,
    };
  }
}

// RAGEvaluator (optional)
class RAGEvaluator {
  private engine = new RAGQueryEngine();

  async evalRetrieval(
    tests: { question: string; expected: string[] }[]
  ): Promise<{ avg_precision: number }> {
    const precisions: number[] = [];

    for (const t of tests) {
      const { sources } = await this.engine.query(t.question);
      const ret = new Set(sources.map((s: any) => s.source));
      const exp = new Set(t.expected);
      const prec =
        exp.size === 0
          ? 0
          : [...exp].filter((x) => ret.has(x)).length / ret.size;
      precisions.push(prec);
    }
    return {
      avg_precision:
        precisions.reduce((a, b) => a + b, 0) / (precisions.length || 1),
    };
  }
}

// API routes
   
const ragEngine = new RAGQueryEngine();
const evaluator = new RAGEvaluator();

/* health */
app.get("/api/v1/health", (_, res) =>
  res.status(200).json({ status: "API up & healthy" })
);

/* query */
app.post("/api/v1/query", async (req, res) => {
  try {
    const { query } = req.body ?? {};
    if (!query || typeof query !== "string")
      return res.status(400).json({ message: "Invalid 'query' field." });

    const result = await ragEngine.query(query);
    return res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Internal error." });
  }
});

/* retrieval evaluation (optional) */
app.post("/api/v1/evaluate/retrieval", async (req, res) => {
  try {
    const { tests = [] } = req.body ?? {};
    const metrics = await evaluator.evalRetrieval(tests);
    res.json(metrics);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Evaluation failed" });
  }
});

app.listen(PORT, () =>
  console.log(`🔥  RAG API listening on http://localhost:${PORT}`)
);
