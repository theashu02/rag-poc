import { performance } from "perf_hooks";
import { bm25Rerank } from "./bm25RerankConfig";
import { blendScores } from "./scoreBlending";
import { retrieve } from "./retrieveService";
import { cleanSync } from "./cleanQuery";
import { extractText } from "./config";
import type { Meta, Match } from "./config";

interface PreparedDoc {
  id: string;
  metadata: Meta;
  rawText: string;
  cleanedText: string;
  vectorScore: number;
}

export interface RankedDocument extends PreparedDoc {
  score: number;
}

export interface ContextBuildOptions {
  topK?: number;
  maxDocs?: number;
  maxContextChars?: number;
  alreadyClean?: boolean;
}

export interface ContextBuildResult {
  context: string;
  sources: Array<{ id: string; source?: string; score: number }>;
  rankedDocuments: RankedDocument[];
  denseMatches: Match[];
}

export async function buildContextForQuestion(
  question: string,
  namespace?: string,
  options: ContextBuildOptions = {}
): Promise<ContextBuildResult> {
  const startTime = performance.now();
  try {
    const topK = Math.min(options.topK ?? 25, 50);
    const maxDocs = Math.max(1, options.maxDocs ?? 4);
    const maxContextChars = Math.max(500, options.maxContextChars ?? 5000);
    const alreadyClean = options.alreadyClean ?? false;

    const denseMatches = await retrieve(question, topK, namespace, {
      alreadyClean,
    });

    if (!denseMatches.length) {
      return { context: "", sources: [], rankedDocuments: [], denseMatches: [] };
    }

    const preparedDocs: PreparedDoc[] = [];

    for (const match of denseMatches) {
      const metadata = (match.metadata || {}) as Meta;
      const rawText = extractText(metadata);
      if (!rawText || rawText.length < 10) continue;

      const cleanedText = cleanSync(rawText);
      if (cleanedText.length < 20) continue;

      preparedDocs.push({
        id: String(match.id),
        metadata,
        rawText,
        cleanedText,
        vectorScore: match.score ?? 0,
      });
    }

    if (!preparedDocs.length) {
      return { context: "", sources: [], rankedDocuments: [], denseMatches };
    }

    const bm25Scores = bm25Rerank(
      question,
      preparedDocs.map((doc) => ({ id: doc.id, text: doc.rawText })),
      namespace
    );

    const denseScoreMap = new Map<string, number>();
    for (const doc of preparedDocs) {
      denseScoreMap.set(doc.id, doc.vectorScore);
    }

    const blendedScores = blendScores(denseScoreMap, bm25Scores, 0.65);

    const rankedDocuments: RankedDocument[] = preparedDocs
      .map((doc) => ({
        ...doc,
        score: blendedScores.get(doc.id) ?? doc.vectorScore,
      }))
      .sort((a, b) => b.score - a.score);

    const contextChunks: string[] = [];
    const sources: Array<{ id: string; source?: string; score: number }> = [];
    let used = 0;

    for (const doc of rankedDocuments) {
      if (contextChunks.length >= maxDocs) break;

      const chunk = doc.cleanedText;
      if (!chunk || chunk.length < 30) continue;

      if (used + chunk.length > maxContextChars && contextChunks.length >= 2) {
        break;
      }

      contextChunks.push(chunk);
      used += chunk.length;

      const rawSource =
        (doc.metadata?.source as string) ??
        (doc.metadata?.url as string) ??
        (doc.metadata?.file as string);

      const source =
        typeof rawSource === "string" && rawSource.trim().length > 0
          ? rawSource.trim()
          : undefined;

      sources.push({
        id: doc.id,
        score: Number.isFinite(doc.score) ? Number(doc.score) : 0,
        source,
      });

      if (used >= maxContextChars) {
        break;
      }
    }

    const context = contextChunks.join("\n\n---\n\n");

    return { context, sources, rankedDocuments, denseMatches };
  } finally {
    const duration = performance.now() - startTime;
    // Report overall context-building latency, including retrieval and reranking.
    console.log(`[Timing] buildContextForQuestion completed in ${duration.toFixed(2)}ms`);
  }
}
