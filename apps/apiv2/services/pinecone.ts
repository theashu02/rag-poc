import { Pinecone } from "@pinecone-database/pinecone";
import { PINECONE_INDEX,PINECONE_API_KEY } from "./config";

// if(!PINECONE_INDEX || !PINECONE_API_KEY) {
//     console.log("---- Missing env: Pinecone key or Pinecone Index is not present. ----")
//     throw new Error("---- Missing env: pinecone key or pinecone Index is not present. ----")
// } else {
//     console.log("---- Pinecone key and Index is Present ----")
// }

export const pinecone = new Pinecone({ apiKey: 'pcsk_9TP3e_H8veAGvy46pxVHowVev7z5NaGSUhc27qhtvDFq21Zfhuz2LtY8g3wrujfknNVoj' });

export const pineconeIndex = pinecone.index('rag-poc');
const idx = pinecone.describeIndex('rag-poc');

const ans = await idx;
console.log(ans);



// const models = await pinecone.inference.listModels();
// console.log(models);