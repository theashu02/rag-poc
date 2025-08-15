import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { performance } from "perf_hooks";

const app = express();
const PORT = process.env.PORT || 5000;

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

type SourceInfo = {
  id: string;
  score: number;
  source: string;
  chunkIndex: number;
  summary?: string;
  entities?: string[];
};

const CONTEXT_WINDOW_CHARS = 16_000;         // = 8000 tokens

async function embed(text: string): Promise<number[]> {
  const { data } = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return data[0]?.embedding || [];
}

/* =========================================================
   1)  RAGQueryEngine  — single function version
   ========================================================= */
async function queryEngine(
  question: string,
  topK = 50
): Promise<{
  answer: string;
  sources: SourceInfo[];
  documentsUsed: number;
  totalRetrieved: number;
  contextChars: number;
  searchTime: number;
}> {
  const t0 = performance.now();

  /* dense retrieval */
  const queryVector = await embed(question);
  const results = await index.query({
    vector: queryVector,
    topK: Math.min(Math.max(1, topK), 100),
    includeMetadata: true,
  });

  const matches = results.matches ?? [];
  if (!matches.length) {
    return {
      answer: "I couldn't find relevant information to answer your question.",
      sources: [],
      documentsUsed: 0,
      totalRetrieved: 0,
      contextChars: 0,
      searchTime: (performance.now() - t0) / 1000,
    };
  }

  /* context assembly */
  const contextParts: string[] = [];
  const sources: SourceInfo[] = [];
  let charCount = 0;

  for (let i = 0; i < matches.length; i++) {
    const m = matches[i];
    if(!m) continue;
    const md = (m.metadata ?? {}) as Record<string, any>;
    const chunk =
      md.text ?? md.content ?? md.pageContent ?? md.page_content ?? "";
    if (!chunk) continue;

    if (charCount + chunk.length > CONTEXT_WINDOW_CHARS) break;

    contextParts.push(`[Document ${i + 1}]\n${chunk}`);
    charCount += chunk.length;

    sources.push({
      id: m.id,
      score: m.score ?? 0,
      source: md.source ?? "Unknown",
      chunkIndex: md.chunk_index ?? 0,
      summary: md.chunk_summary ?? "",
      entities: md.chunk_entities ?? [],
    });
  }
  const context = contextParts.join("\n\n");

  /* final LLM answer */
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    temperature: 0.3,
    max_tokens: 1000,
    messages: [
      {
        role: "system",
        content:
          `You are a helpful assistant that answers questions based on the provided context.\n` +
          `Guidelines:\n` +
          `1. Answer ONLY with the information in the context.\n` +
          `2. If the context is insufficient, say so.\n` +
          `3. Be concise but comprehensive.\n` +
          `4. Cite with [Document X] references where relevant.`,
      },
      {
        role: "user",
        content: `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer:`,
      },
    ],
  });

  const answer = completion.choices[0]?.message?.content?.trim() ?? "";

  return {
    answer,
    sources,
    documentsUsed: contextParts.length,
    totalRetrieved: matches.length,
    contextChars: charCount,
    searchTime: (performance.now() - t0) / 1000,
  };
}

/* =========================================================
   2)  RAGEvaluator helpers
   ========================================================= */
function precisionRecall(
  expected: Set<string>,
  retrieved: Set<string>
): { precision: number; recall: number } {
  const intersection = new Set([...expected].filter((x) => retrieved.has(x)));
  const precision = retrieved.size ? intersection.size / retrieved.size : 0;
  const recall = expected.size ? intersection.size / expected.size : 0;
  return { precision, recall };
}

async function evaluateRetrievalQuality(testQueries: Array<any>) {
  const precisions: number[] = [];
  const recalls: number[] = [];
  const mrrs: number[] = [];
  const times: number[] = [];

  for (const t of testQueries) {
    const expected = new Set<string>(t.expected_sources || []);
    const { sources, searchTime } = await queryEngine(t.question);
    times.push(searchTime);
    const retrievedIds = new Set(sources.map((s) => s.source));
    const { precision, recall } = precisionRecall(expected, retrievedIds);
    precisions.push(precision);
    recalls.push(recall);

    /* MRR */
    let rr = 0;
    for (let i = 0; i < sources.length; i++) {
      const src = sources[i]?.source;
      if(src && expected.has(src)){
        rr = 1 / (i + 1);
        break;
      }
    }
    mrrs.push(rr);
  }

  const avg = (arr: number[]) =>
    arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

  return {
    avg_precision: avg(precisions),
    avg_recall: avg(recalls),
    mrr: avg(mrrs),
    avg_response_time: avg(times),
    total_queries: testQueries.length,
  };
}

async function evaluateAnswerQuality(testCases: Array<any>) {
  const scores: number[] = [];

  for (const t of testCases) {
    const { answer } = await queryEngine(t.question);

    /* LLM-as-judge */
    const resp = await openai.chat.completions.create({
      model: "gpt-4o",
      temperature: 0,
      max_tokens: 10,
      messages: [
        {
          role: "system",
          content:
            `Evaluate the quality of the generated answer compared to the expected answer.\n` +
            `Score 1-5 (1 = wrong, 5 = perfect).  Return ONLY the number.`,
        },
        {
          role: "user",
          content:
            `Question: ${t.question}\n\n` +
            `Expected: ${t.expected_answer}\n\n` +
            `Generated: ${answer}\n\n` +
            `Score:`,
        },
      ],
    });

    const score = Number(resp.choices[0]?.message?.content?.trim() || 0);
    scores.push(isNaN(score) ? 0 : score);
  }

  const avg =
    scores.length > 0
      ? scores.reduce((a, b) => a + b, 0) / scores.length
      : 0;

  const distribution = scores.reduce<Record<number, number>>((acc, s) => {
    acc[s] = (acc[s] || 0) + 1;
    return acc;
  }, {});

  return {
    avg_quality_score: avg,
    score_distribution: distribution,
    total_evaluated: scores.length,
  };
}

/* =========================================================
   3)  ROUTES
   ========================================================= */

/* Health-check */
app.get("/api/v1/health", (_, res) =>
  res.status(200).json({ status: "API up & healthy" })
);

/* RAGQueryEngine endpoint */
app.post("/api/v1/query", async (req, res) => {
  try {
    const { query, topK } = req.body ?? {};
    if (!query || typeof query !== "string")
      return res.status(400).json({ message: "Invalid 'query' field." });

    const result = await queryEngine(query, topK);
    return res.json(result);
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: "Internal Error" });
  }
});

/* =========================================================
   -- For offline testing purpose only
   ========================================================= */

/* Retrieval-quality evaluation */
app.post("/api/v1/evaluate/retrieval", async (req, res) => {
  try {
    const { testQueries = [] } = req.body ?? {};
    const metrics = await evaluateRetrievalQuality(testQueries);
    return res.json(metrics);
  } catch (e) {
    console.error(e);
    return res.status(500).json({ message: "Evaluation failed" });
  }
});

/* Answer-quality evaluation */
app.post("/api/v1/evaluate/answer", async (req, res) => {
  try {
    const { testCases = [] } = req.body ?? {};
    const metrics = await evaluateAnswerQuality(testCases);
    return res.json(metrics);
  } catch (e) {
    console.error(e);
    return res.status(500).json({ message: "Evaluation failed" });
  }
});

app.listen(PORT);
console.log(`API server listening on port ${PORT}`);