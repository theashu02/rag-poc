import { Pinecone } from "@pinecone-database/pinecone";
import { PINECONE_INDEX,PINECONE_API_KEY } from "./config";

if(!PINECONE_INDEX || !PINECONE_API_KEY) {
    console.log("---- Missing env: Pinecone key or Pinecone Index is not present. ----")
    throw new Error("---- Missing env: pinecone key or pinecone Index is not present. ----")
} else {
    console.log("---- Pinecone key and Index is Present ----")
}

export const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });

export const pineconeIndex = pinecone.index(PINECONE_INDEX);



// const models = await pinecone.inference.listModels();
// console.log(models);