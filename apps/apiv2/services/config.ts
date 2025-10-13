export const OPENAI_EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL
export const OPENAI_API_KEY = process.env.OPENAI_API_KEY
export const OPENAI_CHAT_MODEL = process.env.OPENAI_CHAT_MODEL
export const PORT = process.env.PORT || 5000
export const PINECONE_INDEX = process.env.PINECONE_INDEX
export const PINECONE_API_KEY = process.env.PINECONE_API_KEY

export const corsHeaders = {
  "Access-Control-Allow-Origin": "*", // or restrict to a specific domain
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};
