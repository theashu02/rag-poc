import { Pinecone } from "@pinecone-database/pinecone";

const PINECONE_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;

if (!PINECONE_KEY || !PINECONE_INDEX) throw new Error("Missing env: PINECONE_API_KEY or PINECONE_INDEX");

export const pinecone = new Pinecone({ apiKey: PINECONE_KEY });