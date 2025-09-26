export const PORT = Number(process.env.PORT) || 5000;
export const PINECONE_INDEX = process.env.PINECONE_INDEX;
export const OPENAI_EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small";
export const EMBEDDING_DIM = Number(process.env.DIMENSIONS) || 3072;
export const OPENAI_CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini";

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


