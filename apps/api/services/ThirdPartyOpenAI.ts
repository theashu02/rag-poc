import OpenAI from "openai";
import { OPENAI_CHAT_MODEL, PromptForGenerateAnswer, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, chatQueue } from "./config";

if (!OPENROUTER_BASE_URL || !OPENROUTER_API_KEY) {
  console.error("---- Missing env: OPENROUTER_BASE_URL or OPENROUTER_API_KEY ----");
  throw new Error("---- Missing env: OPENROUTER_BASE_URL or OPENROUTER_API_KEY ----");
} else {
  console.log("---- OPENROUTER_BASE_URL and OPENROUTER_API_KEY is present ----");
}

export const openRouterAI = new OpenAI({
  baseURL: OPENROUTER_BASE_URL,
  apiKey: OPENROUTER_API_KEY,
});

export type ChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
};

export interface StreamHandlers {
  onToken?: (token: string) => void;
  onComplete?: (full: string) => void;
  onError?: (error: Error) => void;
}

export interface StreamOptions {
  signal?: AbortSignal;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

function buildMessages(question: string, context: string): ChatMessage[] {
  return [
    { role: "system", content: PromptForGenerateAnswer },
    {
      role: "user",
      content: `Context information: ${context}\nQuestion: ${question}\nProvide a helpful answer based on the context above:`,
    },
  ];
}

// STREAMING VERSION - No queue to allow real-time chunks
export async function streamChatCompletion(
  messages: ChatMessage[],
  handlers: StreamHandlers = {},
  options: StreamOptions = {}
): Promise<string> {
  try {
    const stream = await openRouterAI.chat.completions.create({
      model: options.model ?? OPENAI_CHAT_MODEL,
      messages: messages as any[],
      stream: true,
      temperature: options.temperature ?? 0.7,
      max_tokens: options.maxTokens,
    });

    let fullText = "";

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      if (content) {
        fullText += content;
        handlers.onToken?.(content);
      }
    }

    handlers.onComplete?.(fullText);
    console.log("this is full text: ",fullText);
    return fullText;
  } catch (error) {
    handlers.onError?.(error as Error);
    throw error;
  }
}

// Wrapper for streaming answer
export async function streamAnswer(
  question: string,
  context: string,
  handlers?: StreamHandlers,
  options?: StreamOptions
): Promise<string> {
  return streamChatCompletion(
    buildMessages(question, context),
    handlers,
    options
  );
}

// NON-STREAMING VERSION - Uses queue for rate limiting
export async function generateAnswerOpenRouter(
  question: string,
  context: string
): Promise<string> {
  const task: () => Promise<string> = async () => {
    const completion = await openRouterAI.chat.completions.create({
      model: OPENAI_CHAT_MODEL,
      messages: buildMessages(question, context) as any[],
      temperature: 0.7,
    });

    return (
      completion.choices?.[0]?.message?.content?.trim() ||
      "I apologize, but I couldn't generate a response."
    );
  };

  return chatQueue.add<any>(task);
}



// import OpenAI from "openai";
// import { chatQueue, OPENAI_CHAT_MODEL, PromptForGenerateAnswer, OPENROUTER_API_KEY, OPENROUTER_BASE_URL } from "./config";
// import type { ReadableStreamDefaultReader as NodeStreamReader } from "stream/web";

// type StreamReader = NodeStreamReader<Uint8Array>;

// if (!OPENROUTER_BASE_URL || !OPENROUTER_API_KEY) {
//   console.error("---- Error open router key or base url not present or incorrect. ----",);
//   throw new Error("---- Missing env: OPENROUTER_BASE_URL or OPENROUTER_API_KEY ----");
// } else {
//   console.log("---- Open router key and url are present. ----");
// }

// const CHAT_COMPLETIONS_URL = (() => {
//   const base = OPENROUTER_BASE_URL.endsWith("/") ? OPENROUTER_BASE_URL : `${OPENROUTER_BASE_URL}/`;
//   try {
//     return new URL("chat/completions", base).toString();
//   } catch (error) {
//     console.error("---- Invalid OPENROUTER_BASE_URL ----", error);
//     throw error;
//   }
// })();

// export const openRouterAI = new OpenAI({
//   baseURL: OPENROUTER_BASE_URL,
//   apiKey: OPENROUTER_API_KEY,
// });

// export type ChatMessage = {
//   role: "system" | "user" | "assistant" | "tool";
//   content: string;
// };

// export interface StreamHandlers {
//   onToken?: (token: string) => void;
//   onComplete?: (full: string) => void;
//   onError?: (error: Error) => void;
// }

// export interface StreamOptions {
//   signal?: AbortSignal;
//   model?: string;
//   temperature?: number;
//   maxTokens?: number;
//   headers?: Record<string, string>;
// }

// type OpenRouterStreamChoice = {
//   delta?: { content?: string };
//   finish_reason?: string | null;
// };

// type OpenRouterStreamChunk = {
//   choices?: OpenRouterStreamChoice[];
//   error?: { message?: string; code?: string };
// };

// type ErrorPayload = {
//   error?: { message?: string };
//   message?: string;
// };

// const baseHeaders: Record<string, string> = {
//   Authorization: `Bearer ${OPENROUTER_API_KEY}`,
//   "Content-Type": "application/json",
// };

// function mergeHeaders(extra?: Record<string, string>): Record<string, string> {
//   return extra ? { ...baseHeaders, ...extra } : { ...baseHeaders };
// }

// function buildMessages(question: string, context: string): ChatMessage[] {
//   return [
//     { role: "system", content: PromptForGenerateAnswer },
//     {
//       role: "user",
//       content: `Context information: ${context}
//       Question: ${question}
//       Provide a helpful answer based on the context above:`,
//     },
//   ];
// }

// async function consumeEventStream(
//   reader: StreamReader,
//   handlers: StreamHandlers,
// ): Promise<string> {
//   const decoder = new TextDecoder();
//   let buffer = "";
//   let aggregated = "";
//   let closed = false;

//   const processLine = (rawLine: string) => {
//     const line = rawLine.trim();
//     if (!line || line.startsWith(":")) return;
//     if (!line.startsWith("data:")) return;

//     const payload = line.slice(5).trim();
//     if (!payload) return;

//     if (payload === "[DONE]") {
//       closed = true;
//       return;
//     }

//     let chunk: OpenRouterStreamChunk;
//     try {
//       chunk = JSON.parse(payload) as OpenRouterStreamChunk;
//     } catch {
//       return;
//     }

//     if (chunk.error) {
//       const err = new Error(chunk.error.message ?? "OpenRouter stream error");
//       (err as Error & { code?: string }).code = chunk.error.code;
//       throw err;
//     }

//     const choice = chunk.choices?.[0];
//     const delta = choice?.delta?.content ?? "";
//     if (delta) {
//       aggregated += delta;
//       handlers.onToken?.(delta);
//     }

//     if (choice?.finish_reason === "error") {
//       throw new Error("OpenRouter terminated stream with an error finish_reason");
//     }

//     if (choice?.finish_reason && choice.finish_reason !== "stop") {
//       closed = true;
//     }
//   };

//   const flushLines = (force = false) => {
//     let newlineIdx = buffer.indexOf("\n");
//     while (newlineIdx !== -1) {
//       const line = buffer.slice(0, newlineIdx);
//       buffer = buffer.slice(newlineIdx + 1);
//       processLine(line);
//       if (closed) {
//         buffer = "";
//         return;
//       }
//       newlineIdx = buffer.indexOf("\n");
//     }

//     if (force && buffer.length) {
//       processLine(buffer);
//       buffer = "";
//     }
//   };

//   while (!closed) {
//     const { value, done } = await reader.read();
//     if (done) break;

//     buffer += decoder.decode(value, { stream: true });
//     flushLines();
//   }

//   buffer += decoder.decode();
//   flushLines(true);

//   handlers.onComplete?.(aggregated);
//   return aggregated;
// }

// const streamChatCompletionInternal = async (
//   messages: ChatMessage[],
//   handlers: StreamHandlers = {},
//   options: StreamOptions = {},
// ): Promise<string> => {
//   let reader: StreamReader | null = null;

//   try {
//     const payload: Record<string, unknown> = {
//       model: options.model ?? OPENAI_CHAT_MODEL,
//       messages,
//       stream: true,
//     };

//     if (typeof options.temperature === "number") {
//       payload.temperature = options.temperature;
//     }

//     if (typeof options.maxTokens === "number") {
//       payload.max_tokens = options.maxTokens;
//     }

//     const response = await fetch(CHAT_COMPLETIONS_URL, {
//       method: "POST",
//       headers: mergeHeaders(options.headers),
//       body: JSON.stringify(payload),
//       signal: options.signal,
//     });

//     if (!response.ok) {
//       let message = `OpenRouter request failed with status ${response.status}`;
//       try {
//         const body = (await response.json()) as ErrorPayload;
//         message = body.error?.message ?? body.message ?? message;
//       } catch {
//         const text = await response.text().catch(() => "");
//         if (text) message = text;
//       }
//       throw new Error(message);
//     }

//     const stream = response.body as ReadableStream<Uint8Array> | null;
    
//     reader = (stream?.getReader() as StreamReader) ?? null;
//     if (!reader) {
//       throw new Error("OpenRouter returned an empty response body");
//     }

//     return await consumeEventStream(reader, handlers);
//   } catch (error) {
//     handlers.onError?.(error as Error);
//     throw error;
//   } finally {
//     if (reader?.cancel) {
//       try {
//         await reader.cancel();
//       } catch {
//         // ignore cancel errors
//       }
//     }
//   }
// };

// export async function streamChatCompletion(messages: ChatMessage[], handlers?: StreamHandlers, options?: StreamOptions): Promise<string> {
//   const task: () => Promise<string> = () => streamChatCompletionInternal(messages, handlers, options);
//   console.log("task: ===> ",task)
//   return chatQueue.add<any>(task);
// }

// export async function streamAnswer(question: string, context: string, handlers?: StreamHandlers, options?: StreamOptions): Promise<string> {
//   return streamChatCompletion(buildMessages(question, context), handlers, options);
// }

// export async function generateAnswerOpenRouter(
//   question: string,
//   context: string,
// ): Promise<string> {
//   const task: () => Promise<string> = async () => {
//     const completion = await openRouterAI.chat.completions.create({
//       model: OPENAI_CHAT_MODEL,
//       messages: buildMessages(question, context) as any[],
//     });

//     return (
//       completion.choices?.[0]?.message?.content?.trim() ??
//       "I apologize, but I couldn't generate a response."
//     );
//   };

//   return chatQueue.add<any>(task);
// }
