import { pinecone } from "../pinecone";
import type { Document } from "./hybridQuery";

const RERANKER_MODEL = 'bge-reranker-v2-m3';

export async function rerankDocuments(query: string, documents: Document[], topK: number): Promise<Document[]> {

  if (!query || typeof query !== 'string' || query.trim() === '') {
    throw new Error("Reranker Error: Input query cannot be empty.");
  }

  if (!documents || !Array.isArray(documents) || documents.length === 0) {
    console.log("No documents provided to rerank.");
    return [];
  }

  try {

    const compliantDocs = documents.map(doc => ({
      id: doc.id,
      text: doc.text,
      // We intentionally leave out the 'score' number property.
    }));

    const rerankResponse = await pinecone.inference.rerank(
      RERANKER_MODEL,
      query,
      compliantDocs,
      { topN: topK }
    );
    
    if (!rerankResponse.data || rerankResponse.data.length === 0) {
      console.log("Reranker returned no results.");
      return [];
    }
    
    // const rerankedDocs: Document[] = rerankResponse.data.map(result=> ({
    //   id: result.document?.id,
    //   text: result.document?.text,
    //   score: result.score,
    // }));

    const rerankedDocs: Document[] = rerankResponse.data.map(
      (result: { document: { id: string, text: string }; score: number }) => ({
        id: result.document?.id,
        text: result.document?.text,
        score: result.score,
      })
    );

    return rerankedDocs;

  } catch (error) {
    console.error("Error during reranking process:", error);
    throw new Error("Failed to rerank documents.");
  }
}