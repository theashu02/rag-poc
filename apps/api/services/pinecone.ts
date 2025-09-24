import { Pinecone } from "@pinecone-database/pinecone";

const PINECONE_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;

if(!PINECONE_INDEX || !PINECONE_KEY) {
    console.log("---- Missing env: Pinecone key or Pinecone Index is not present. ----")
    throw new Error("---- Missing env: pinecone key or pinecone Index is not present. ----")
} else {
    console.log("---- Pinecone key and Index is Present ----")
}

export const pinecone = new Pinecone({ apiKey: PINECONE_KEY });