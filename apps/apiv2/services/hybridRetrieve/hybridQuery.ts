import { pineconeIndex } from "../pinecone";
import type { HybridVectors } from "../vectorGenerator";
import type { QueryOptions } from "@pinecone-database/pinecone";

// Define the Document interface for the return type
export interface Document {
  id: string;
  score: number;
  text: string;
}

export async function searchWithVectors(vectors: HybridVectors, namespace: string, topK: number = 10): Promise<Document[]> {
  try {
    const { dense, sparse } = vectors;

    const queryParams: QueryOptions = {
      vector: dense,
      sparseVector: sparse,
      topK,
      includeMetadata: true,
    };

    let results;

    if (namespace) {
      if (typeof (pineconeIndex as any).namespace !== "function") {
        queryParams.filter = { namespace: { $eq: namespace } };
        results = await pineconeIndex.query(queryParams);
      } else {
        results = await pineconeIndex.namespace(namespace).query(queryParams);
      }
    } else {
      results = await pineconeIndex.query(queryParams);
    }

    if (!results.matches) {
      return [];
    }

    return results.matches.map((match) => ({
      id: match.id,
      score: match.score || 0,
      text: (match.metadata?.text as string) || "",
    }));
    
  } catch (error) {
    console.error("Error during search with pre-computed vectors:", error);
    throw new Error("Failed to execute search on the vector store.");
  }
}