import PQueue from "p-queue";
import { LRUCache } from "lru-cache";
import type { ScoredPineconeRecord } from "@pinecone-database/pinecone";

export type Meta = Record<string, any>;
export type Match = ScoredPineconeRecord<Meta>;

export const EMBEDDING_CACHE = new LRUCache<string, number[]>({ max: 10000, ttl: 1000 * 60 * 60 * 2 });
export const chatQueue = new PQueue({ concurrency: 5, interval: 1000, intervalCap: 20 });
export const embeddingQueue = new PQueue({ concurrency: 10, interval: 1000, intervalCap: 50 });
export const BM25_INSTANCES_CACHE = new LRUCache<string, any>({ max: 100, ttl: 1000 * 60 * 60 });

export const PORT = Number(process.env.PORT) || 5000;
export const PINECONE_INDEX = process.env.PINECONE_INDEX;
export const OPENAI_EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small";
export const EMBEDDING_DIM = Number(process.env.DIMENSIONS) || 3072;
export const OPENAI_CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini";
export const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
export const OPENROUTER_BASE_URL = process.env.OPENROUTER_BASE_URL;

// cors configration
export const corsHeaders = {
  "Access-Control-Allow-Origin": "*", // or restrict to a specific domain
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

export const PromptForGenerateAnswer = `You are a helpful and precise assistant. Follow these rules:
              1. Answer primarily from the provided context
              2. If the answer isn't in the context, say so politely
              3. Be conversational but professional
              4. If relevant, suggest related topics from the context
              5. Structure complex answers clearly
              6. Always maintain a helpful tone
              7. Does not specifically mention what is there in the context and what is not just prepare the answer and reply based on the context.`;


export function normalizeMap(scores: Map<string, number>): Map<string, number> {
  const vals = [...scores.values()];
  if (vals.length === 0) return scores;
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  if (max === min) {
    const mid = 0.5;
    return new Map([...scores.keys()].map((k) => [k, mid]));
  }
  return new Map(
    [...scores.entries()].map(([k, v]) => [k, (v - min) / (max - min)])
  );
}

export function extractText(md: Meta): string {
  return (
    (md?.text as string) ??
    (md?.content as string) ??
    (md?.chunk as string) ??
    ""
  );
}