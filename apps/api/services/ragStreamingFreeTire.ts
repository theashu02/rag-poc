import { bm25Rerank } from "./bm25RerankConfig";
import type { Meta } from "./config";
import { isGreeting, generateGreetingResponse, generateSmallTalkResponse, isSmallTalk } from "./greeting";
import { streamAnswer } from "./ThirdPartyOpenAI";
import { clean } from "./cleanQuery";
import { extractText } from "./config";
import { retrieve } from "./retrieveService";
import { blendScores } from "./scoreBlending";

export async function retrieveContext(
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

export async function ragStreaming(
  question: string,
  namespace: string | undefined,
  onToken: (token: string) => void,
  onSources: (
    sources: Array<{ id: string; source?: string; score: number }>
  ) => void
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
      const fallback =
        "I couldn't find relevant information to answer your question from the current knowledge base.";
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
