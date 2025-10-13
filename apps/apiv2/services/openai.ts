import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_MODEL } from './config';

if (!OPENAI_API_KEY || !OPENAI_EMBEDDING_MODEL) {
    console.error("---- Missing env: OPENAI_API_KEY or OPENAI_EMBEDDING_MODEL ----");
    throw new Error("---- Missing env: OPENAI_API_KEY or OPENAI_EMBEDDING_MODEL ----");
} else {
    console.log("---- OPENAI_API_KEY is present ----");
    console.log("---- OPENAI_EMBEDDING_MODEL is present ----");
}

export const chatModel = new ChatOpenAI({
    apiKey: OPENAI_API_KEY,
    modelName: OPENAI_CHAT_MODEL,
    temperature: 0.2,
});
  

export const embeddingsModel = new OpenAIEmbeddings({
    apiKey: OPENAI_API_KEY,
    modelName: OPENAI_EMBEDDING_MODEL,
    maxRetries: 2,
})
