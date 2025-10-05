import { performance } from "perf_hooks";
import { generateSemanticCacheKey, SEMANTIC_CACHE } from "./cache";
import { generateGreetingResponse, generateSmallTalkResponse, isGreeting, isSmallTalk } from "./greeting";
import { buildContextForQuestion } from "./retrievalPipeline";
import { generateAnswer } from "./ThirdPartyOpenAI";

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

  const semanticCacheKey = generateSemanticCacheKey(question, namespace);
  const cached = SEMANTIC_CACHE.get(semanticCacheKey);
  if (cached) {
    console.log("Semantic cache hit");
    return {
      ...cached,
      latencySec: (performance.now() - t0) / 1000,
    };
  }

  try {
    const { context, sources } = await buildContextForQuestion(question, namespace, {
      alreadyClean: true,
    });

    if (!context) {
      const result = {
        answer:
          "I couldn't find relevant information to answer your question from the current knowledge base.",
        sources: [],
        latencySec: (performance.now() - t0) / 1000,
      };

      SEMANTIC_CACHE.set(semanticCacheKey, result);
      return result;
    }

    const answer = await generateAnswer(question, context);

    const result = {
      answer:
        answer || "I don't have enough information in the provided context.",
      sources,
      latencySec: (performance.now() - t0) / 1000,
    };

    SEMANTIC_CACHE.set(semanticCacheKey, result);

    return result;
  } catch (error) {
    console.error("Error in RAG process:", error);
    return {
      answer:
        "I encountered an error while processing your request. Please try again.",
      sources: [],
      latencySec: (performance.now() - t0) / 1000,
    };
  }
}