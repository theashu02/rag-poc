import { embeddingsModel } from "./openai";
import type { SparseVector } from "./type";
import { pinecone } from "./pinecone";

interface sparseVectors {
  sparseIndices?: number[]; 
  sparseValues?: number[]
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export interface HybridVectors {
  dense: number[];
  sparse: SparseVector;
}

// Generates a dense vector for semantic search
async function generateDenseVector(text: string): Promise<number[]> {
  return embeddingsModel.embedQuery(text);
}

// Generates sparse vector using Pinecone optimized model
async function generateSparseVector(text: string, inputType: 'query' | 'passage' = 'query'): Promise<SparseVector> {
  const response = await pinecone.inference.embed('pinecone-sparse-english-v0', [text],                        
    {                               
      inputType: inputType,
      truncate: 'END',
    }
  );

  const sparseData = response.data?.[0] as sparseVectors;
  
  if (!Array.isArray(sparseData?.sparseIndices) || !Array.isArray(sparseData?.sparseValues)) {
    throw new Error(`Invalid sparse embedding response: missing sparseIndices or sparseValues`);
  }

  if (sparseData.sparseIndices.length !== sparseData.sparseValues.length) {
    throw new Error(`Sparse vector length mismatch: indices=${sparseData.sparseIndices.length}, values=${sparseData.sparseValues.length}`);
  }

  return {
    indices: sparseData.sparseIndices,
    values: sparseData.sparseValues,
  };
}

export async function generateVectors(text: string, inputType: 'query' | 'passage' = 'query', retries: number = 2): Promise<HybridVectors> {

  if (!text || typeof text !== 'string' || text.trim() === '') {
    throw new Error("Input text cannot be null, empty, or just whitespace.");
  }

  let attempt = 0;
  while (attempt <= retries) {
    try {
      const [dense, sparse] = await Promise.all([
        generateDenseVector(text),
        generateSparseVector(text, inputType),
      ]);
      
      return { dense, sparse };

    } catch (error: any) {
      attempt++;
      console.error(`Attempt ${attempt} failed:`, error.message);
      
      if (attempt > retries) {
        console.error("All retry attempts failed. Could not generate vectors.");
        throw error;
      }
      
      const delay = 300 * attempt;
      console.log(`Waiting ${delay}ms before next retry...`);
      await sleep(delay);
    }
  }
  throw new Error("Exited retry loop without generating vectors.");
}

// for testing 
// const text = "Artificial intelligence helps computers learn from data.";
// const ans = await generateSparseVector(text, 'passage');
// console.log("=====", "Sparese vector embeddings", "=====", ans);
// const ans2 = await generateDenseVector(text);
// console.log("=====", "Dense vector embeddings", "=====", ans2);
// const ans = await generateVectors(text, 'query');
// console.log(ans);

// const models = await pinecone.inference.listModels();
// console.log(models);