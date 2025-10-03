import { bm25Rerank } from "./bm25RerankConfig";
import { generateSemanticCacheKey, SEMANTIC_CACHE } from "./cache";
import { clean } from "./cleanQuery";
import { extractText } from "./config";
import { generateGreetingResponse, generateSmallTalkResponse, isGreeting, isSmallTalk } from "./greeting";
import { retrieve } from "./retrieveService";
import { blendScores } from "./scoreBlending";
import { generateAnswerOpenRouter } from "./ThirdPartyOpenAI";
import type { Meta } from "./config";



export async function rag(question: string, namespace?: string) {
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
