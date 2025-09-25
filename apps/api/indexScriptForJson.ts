import type { ScoredPineconeRecord } from "@pinecone-database/pinecone";
import { performance } from "perf_hooks";
import { generateGreetingResponse, generateSmallTalkResponse, isGreeting, isSmallTalk } from "./services/greeting";
import { PORT, PINECONE_INDEX, OPENAI_EMBEDDING_MODEL, OPENAI_CHAT_MODEL, EMBEDDING_DIM, PromptForGenerateAnswer } from "./services/config";
import bm25PrepTasks from "./services/BM25_config";
import { corsHeaders } from "./services/config";
import { pinecone } from "./services/pinecone";
import { openai } from "./services/openai";
import bm25Search from "wink-bm25-text-search";
import { createHash } from "crypto";
import { mkdir, writeFile, readFile } from "fs/promises";
import { existsSync } from "fs";
import path from "path";

// Production imports for optimization
import { LRUCache } from "lru-cache";
import pQueue from "p-queue";

// Type definitions
type Meta = Record<string, any>;
type Match = ScoredPineconeRecord<Meta>;

interface RAGResult {
  query: string;
  response: string[];
  answer: string;
  sources: Array<{ id: string; source?: string; score: number }>;
  latencySec: number;
}

interface BatchQuery {
  query_num: string;
  query: string;
}

// Global caches and optimization structures
const EMBEDDING_CACHE = new LRUCache<string, number[]>({
  max: 10000,
  ttl: 1000 * 60 * 60 * 2, // 2 hours
});

const SEMANTIC_CACHE = new LRUCache<string, RAGResult>({
  max: 5000,
  ttl: 1000 * 60 * 30, // 30 minutes
});

const BM25_INSTANCES_CACHE = new LRUCache<string, any>({
  max: 100,
  ttl: 1000 * 60 * 60, // 1 hour
});

// Connection pools and queues for rate limiting
const embeddingQueue = new pQueue({
  concurrency: 10,
  interval: 1000,
  intervalCap: 50,
});

const chatQueue = new pQueue({
  concurrency: 5,
  interval: 1000,
  intervalCap: 20,
});

// Batch processing queue for handling multiple queries
const batchQueue = new pQueue({
  concurrency: 3, // Process 3 queries simultaneously to avoid rate limits
  interval: 2000,
  intervalCap: 3,
});

// Query counter for manual queries (keeping original functionality)
let queryCounter = 0;

// Batch processing state
let batchProcessingActive = false;
let batchProgress = {
  total: 0,
  completed: 0,
  failed: 0,
  startTime: 0,
  currentQuery: null as string | null,
  processedQueries: new Set<string>(), // Track processed queries
  remainingQueries: [] as BatchQuery[] // Only unprocessed queries
};

// Data directory paths
const DATA_DIR = path.join(process.cwd(), 'Data');
const QUERY_FILE_PATH = path.join(process.cwd(), 'Queries.json');

// Namespace for processing queries - specified by user
const PROCESSING_NAMESPACE = "68a77ce1fe96dd4af1a7b7a9";

// Ensure Data directory exists
async function ensureDataDirectory() {
  try {
    if (!existsSync(DATA_DIR)) {
      await mkdir(DATA_DIR, { recursive: true });
      console.log(`📁 Created Data directory: ${DATA_DIR}`);
    }
  } catch (error) {
    console.error('Error creating Data directory:', error);
  }
}

// Get set of already processed query numbers (OPTIMIZED - single scan)
async function getProcessedQueries(): Promise<Set<string>> {
  try {
    await ensureDataDirectory();
    const fs = await import('fs/promises');
    const files = await fs.readdir(DATA_DIR);
    
    const processedSet = new Set<string>();
    
    for (const file of files) {
      if (file.endsWith('.json')) {
        const queryNum = file.replace('.json', '');
        processedSet.add(queryNum);
      }
    }
    
    console.log(`📊 Found ${processedSet.size} already processed queries`);
    return processedSet;
  } catch (error) {
    console.error('Error getting processed queries:', error);
    return new Set<string>();
  }
}

// Save query result to JSON file with specific filename
async function saveQueryToJson(query: string, response: string[], fileName: string) {
  try {
    await ensureDataDirectory();
    
    const jsonData = {
      query: query,
      response: response
    };
    
    const filePath = path.join(DATA_DIR, `${fileName}.json`);
    
    await writeFile(filePath, JSON.stringify(jsonData, null, 2), 'utf-8');
    console.log(`💾 Saved query result to: ${filePath}`);
    
    // Add to processed queries set
    batchProgress.processedQueries.add(fileName);
    
    return { fileName: `${fileName}.json`, filePath, success: true };
  } catch (error) {
    console.error('Error saving query to JSON:', error);
    return { fileName: null, filePath: null, success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

// Load batch queries from query.json file and filter unprocessed ones (OPTIMIZED)
async function loadUnprocessedQueries(): Promise<BatchQuery[]> {
  try {
    if (!existsSync(QUERY_FILE_PATH)) {
      console.log(`⚠️  Query file not found: ${QUERY_FILE_PATH}`);
      return [];
    }
    
    const fileContent = await readFile(QUERY_FILE_PATH, 'utf-8');
    const allQueries: BatchQuery[] = JSON.parse(fileContent);
    
    // Validate query format
    const validQueries = allQueries.filter(q => 
      q && 
      typeof q.query_num === 'string' && 
      typeof q.query === 'string' && 
      q.query_num.trim() && 
      q.query.trim()
    );
    
    // Get already processed queries (SINGLE SCAN)
    const processedQueries = await getProcessedQueries();
    
    // Filter out already processed queries
    const unprocessedQueries = validQueries.filter(q => 
      !processedQueries.has(q.query_num)
    );
    
    console.log(`📋 Total queries: ${validQueries.length}`);
    console.log(`✅ Already processed: ${processedQueries.size}`);
    console.log(`🔄 Remaining to process: ${unprocessedQueries.length}`);
    
    // Sort unprocessed queries by query_num for consistent processing order
    unprocessedQueries.sort((a, b) => {
      const numA = parseInt(a.query_num) || 0;
      const numB = parseInt(b.query_num) || 0;
      return numA - numB;
    });
    
    return unprocessedQueries;
  } catch (error) {
    console.error('Error loading batch queries:', error);
    return [];
  }
}

// Process a single batch query (OPTIMIZED - no file existence check)
async function processBatchQuery(batchQuery: BatchQuery): Promise<boolean> {
  try {
    batchProgress.currentQuery = `Query ${batchQuery.query_num}: ${batchQuery.query.substring(0, 50)}...`;
    
    console.log(`🔄 Processing Query ${batchQuery.query_num} with namespace ${PROCESSING_NAMESPACE}: ${batchQuery.query}`);
    
    const startTime = performance.now();
    
    // Process the query using the RAG function with the specified namespace
    const result = await rag(batchQuery.query, PROCESSING_NAMESPACE);
    
    const processingTime = performance.now() - startTime;
    
    // Save the result
    const saveResult = await saveQueryToJson(
      result.query, 
      result.response, 
      batchQuery.query_num
    );
    
    if (saveResult.success) {
      batchProgress.completed++;
      console.log(`✅ Query ${batchQuery.query_num} completed in ${processingTime.toFixed(2)}ms - Found ${result.response.length} documents`);
      return true;
    } else {
      batchProgress.failed++;
      console.error(`❌ Failed to save Query ${batchQuery.query_num}:`, saveResult.error);
      return false;
    }
    
  } catch (error) {
    batchProgress.failed++;
    console.error(`❌ Error processing Query ${batchQuery.query_num}:`, error);
    return false;
  } finally {
    batchProgress.currentQuery = null;
  }
}

// Optimized batch processing - only processes unprocessed queries
async function startBatchProcessing(): Promise<void> {
  if (batchProcessingActive) {
    console.log('⚠️  Batch processing already active');
    return;
  }
  
  // Load only unprocessed queries (FILTERED)
  const unprocessedQueries = await loadUnprocessedQueries();
  
  if (unprocessedQueries.length === 0) {
    console.log('🎉 No queries to process - all queries are already completed!');
    return;
  }
  
  batchProcessingActive = true;
  batchProgress = {
    total: unprocessedQueries.length,
    completed: 0,
    failed: 0,
    startTime: Date.now(),
    currentQuery: null,
    processedQueries: await getProcessedQueries(),
    remainingQueries: unprocessedQueries
  };
  
  console.log(`🚀 Starting optimized batch processing of ${unprocessedQueries.length} unprocessed queries using namespace: ${PROCESSING_NAMESPACE}...`);
  
  if (unprocessedQueries.length < 10) {
    console.log(`📝 Queries to process: ${unprocessedQueries.map(q => q.query_num).join(', ')}`);
  } else {
    const first5 = unprocessedQueries.slice(0, 5).map(q => q.query_num);
    const last5 = unprocessedQueries.slice(-5).map(q => q.query_num);
    console.log(`📝 First 5 queries: ${first5.join(', ')}`);
    console.log(`📝 Last 5 queries: ${last5.join(', ')}`);
  }
  
  try {
    // Add only unprocessed queries to the batch queue
    const promises = unprocessedQueries.map(query => 
      batchQueue.add(() => processBatchQuery(query))
    );
    
    // Wait for all queries to complete
    await Promise.all(promises);
    
    const totalTime = (Date.now() - batchProgress.startTime) / 1000;
    
    console.log(`🎉 Optimized batch processing completed!`);
    console.log(`📊 Results: ${batchProgress.completed} successful, ${batchProgress.failed} failed`);
    console.log(`⏱️  Total time: ${totalTime.toFixed(2)} seconds`);
    console.log(`🔖 All queries processed with namespace: ${PROCESSING_NAMESPACE}`);
    
    // Final summary
    const finalProcessed = await getProcessedQueries();
    console.log(`📈 Total processed queries: ${finalProcessed.size}`);
    
  } catch (error) {
    console.error('❌ Batch processing error:', error);
  } finally {
    batchProcessingActive = false;
  }
}

// Enhanced batch status with progress details
function getBatchStatus() {
  const totalTime = batchProcessingActive ? (Date.now() - batchProgress.startTime) / 1000 : 0;
  const progress = batchProgress.total > 0 ? (batchProgress.completed + batchProgress.failed) / batchProgress.total : 0;
  
  return {
    active: batchProcessingActive,
    namespace: PROCESSING_NAMESPACE,
    optimized: true,
    progress: {
      ...batchProgress,
      progressPercentage: Math.round(progress * 100),
      elapsedTime: totalTime,
      estimatedTimeRemaining: progress > 0 ? (totalTime / progress) * (1 - progress) : 0,
      totalProcessedEver: batchProgress.processedQueries.size,
      remainingCount: batchProgress.remainingQueries.length
    }
  };
}

// Resume processing function - for restarting after interruption
async function resumeProcessing(): Promise<void> {
  console.log('🔄 Checking for unfinished queries to resume processing...');
  await startBatchProcessing();
}

// Load query counter from existing files (for manual queries)
async function initializeQueryCounter() {
  try {
    await ensureDataDirectory();
    
    // Count existing JSON files to determine next query number for manual queries
    const fs = await import('fs/promises');
    const files = await fs.readdir(DATA_DIR);
    const sequentialFiles = files.filter(file => 
      file.endsWith('.json') && /^\d+\.json$/.test(file)
    );
    
    if (sequentialFiles.length > 0) {
      const numbers = sequentialFiles
        .map(file => parseInt(file.replace('.json', '')))
        .filter(num => !isNaN(num));
      queryCounter = numbers.length > 0 ? Math.max(...numbers) : 0;
    }
    
    console.log(`🔢 Initialized manual query counter at: ${queryCounter}`);
  } catch (error) {
    console.log('📝 Starting with fresh manual query counter');
    queryCounter = 0;
  }
}

// [ALL UTILITY FUNCTIONS AND RAG PIPELINE - Same as before]
// type Meta = Record<string, any>;
// type Match = ScoredPineconeRecord<Meta>;

// Utility functions
function clean(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function generateCacheKey(query: string, namespace?: string): string {
  return createHash("sha256")
    .update(`${query}:${namespace || "default"}`)
    .digest("hex");
}

function generateSemanticCacheKey(query: string, namespace?: string): string {
  const normalized = query
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .trim();
  return createHash("sha256")
    .update(`${normalized}:${namespace || "default"}`)
    .digest("hex");
}

// Helper function to extract clean document filename
function extractDocumentFilename(metadata: Meta): string {
  const possibleSources = [
    metadata.filename,
    metadata.file,
    metadata.document_name,
    metadata.source,
    metadata.original_filename,
    metadata.document_title,
    metadata.title,
    metadata.name
  ];

  for (const source of possibleSources) {
    if (source && typeof source === 'string' && source.trim()) {
      const cleanSource = source.trim();
      
      // Extract filename from URL if it's a URL
      if (cleanSource.startsWith('http')) {
        const urlParts = cleanSource.split('/');
        const filename = urlParts[urlParts.length - 1];
        if (filename && filename.includes('.')) {
          return filename;
        }
      }
      
      // Return filename with extension if it has one
      if (cleanSource.includes('.')) {
        return cleanSource;
      }
      
      // Add appropriate extension based on content type if available
      const contentType = metadata.content_type || metadata.type || metadata.mime_type;
      if (contentType) {
        if (contentType.includes('pdf')) return `${cleanSource}.pdf`;
        if (contentType.includes('text')) return `${cleanSource}.txt`;
        if (contentType.includes('doc')) return `${cleanSource}.docx`;
        if (contentType.includes('html')) return `${cleanSource}.html`;
        if (contentType.includes('json')) return `${cleanSource}.json`;
        if (contentType.includes('csv')) return `${cleanSource}.csv`;
      }
      
      return `${cleanSource}.txt`; // Default extension
    }
  }
  
  // Fallback: use document ID with timestamp
  const timestamp = Date.now().toString().slice(-6);
  return `document_${metadata.id || timestamp}.txt`;
}

async function embed(text: string): Promise<number[]> {
  const cacheKey = createHash("sha256").update(text).digest("hex");
  const cached = EMBEDDING_CACHE.get(cacheKey);
  if (cached) return cached;

  return embeddingQueue.add(async () => {
    // Double-check cache inside queue to prevent duplicate requests
    const cachedInQueue = EMBEDDING_CACHE.get(cacheKey);
    if (cachedInQueue) return cachedInQueue;

    try {
      const { data } = await openai.embeddings.create({
        model: OPENAI_EMBEDDING_MODEL,
        input: clean(text),
        dimensions: EMBEDDING_DIM,
      });

      const v = data?.[0]?.embedding;
      if (!v || !Array.isArray(v)) {
        throw new Error("Failed to get valid embedding from OpenAI");
      }

      EMBEDDING_CACHE.set(cacheKey, v);
      return v;
    } catch (error) {
      console.error("Embedding error:", error);
      throw new Error(
        `Embedding failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }) as Promise<number[]>;
}

function extractText(md: Meta): string {
  return (
    (md?.text as string) ??
    (md?.content as string) ??
    (md?.chunk as string) ??
    ""
  );
}

function normalizeMap(scores: Map<string, number>): Map<string, number> {
  const vals = [...scores.values()];
  if (vals.length === 0) return scores;
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  if (max === min) {
    const mid = 0.5;
    return new Map([...scores.keys()].map((k) => [k, mid]));
  }
  return new Map(
    [...scores.entries()].map(([k, v]) => [k, (v - min) / (max - min)])
  );
}

// Optimized BM25 with caching
function bm25Rerank(
  query: string,
  docs: Array<{ id: string; text: string }>,
  namespace?: string
): Map<string, number> {
  const scores = new Map<string, number>();
  if (!docs.length) return scores;

  // Create cache key for BM25 instance
  const docIds = docs
    .map((d) => d.id)
    .sort()
    .join(",");
  const bm25CacheKey = createHash("sha256")
    .update(`${docIds}:${namespace || "default"}`)
    .digest("hex");

  let bm25 = BM25_INSTANCES_CACHE.get(bm25CacheKey);

  if (!bm25) {
    bm25 = bm25Search();
    bm25.defineConfig({ fldWeights: { text: 1 } });
    bm25.definePrepTasks(bm25PrepTasks);

    docs.forEach((d) => bm25.addDoc({ text: d.text }, d.id));
    bm25.consolidate();

    BM25_INSTANCES_CACHE.set(bm25CacheKey, bm25);
  }

  try {
    bm25
      .search(query, docs.length)
      .forEach(([id, score]: [string, number]) => scores.set(id, score));
  } catch (error) {
    console.error("BM25 search error:", error);
    // Fallback: equal scores for all docs
    docs.forEach((d) => scores.set(d.id, 1.0));
    return scores;
  }

  docs.forEach((d) => {
    if (!scores.has(d.id)) scores.set(d.id, 0);
  });

  return scores;
}

function blendScores(
  dense: Map<string, number>,
  text: Map<string, number>,
  alpha = 0.65
): Map<string, number> {
  const dn = normalizeMap(dense);
  const tn = normalizeMap(text);

  const ids = new Set([...dn.keys(), ...tn.keys()]);
  const out = new Map<string, number>();
  for (const id of ids) {
    const d = dn.get(id) ?? 0;
    const t = tn.get(id) ?? 0;
    out.set(id, alpha * d + (1 - alpha) * t);
  }
  return out;
}

// Optimized retrieval with better error handling
async function retrieve(
  query: string,
  topK: number,
  namespace?: string
): Promise<Match[]> {
  const qv = await embed(query);
  const index = pinecone.Index(PINECONE_INDEX || "large-3072");

  const maxRetries = 2;
  let lastError: Error | null = null;

  console.log(`🔍 Retrieving from namespace: ${namespace || 'default'}`);

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      if (namespace && typeof (index as any).namespace === "function") {
        const namespacedIndex = (index as any).namespace(namespace);
        const res = await namespacedIndex.query({
          vector: qv,
          topK,
          includeMetadata: true,
        });
        return (res.matches || []) as Match[];
      }

      const filter = namespace ? { userId: { $eq: namespace } } : undefined;
      const res = await index.query({
        vector: qv,
        topK,
        includeMetadata: true,
        filter,
      });
      return (res.matches || []) as Match[];
    } catch (error) {
      lastError = error as Error;
      console.warn(`Retrieve attempt ${attempt + 1} failed:`, error);
      if (attempt < maxRetries - 1) {
        await new Promise((resolve) =>
          setTimeout(resolve, Math.pow(2, attempt) * 100)
        );
      }
    }
  }

  throw lastError || new Error("All retrieve attempts failed");
}

async function generateAnswer(
  question: string,
  context: string
): Promise<string> {
  console.log("----this is context----", context);
  const queuedTask = async (): Promise<string> => {
    try {
      const completion = await openai.chat.completions.create({
        model: OPENAI_CHAT_MODEL,
        temperature: 0.2,
        max_tokens: 600,
        messages: [
          {
            role: "system",
            content: PromptForGenerateAnswer,
          },
          {
            role: "user",
            content: `Context information: ${context}
            Question: ${question}
            Provide a helpful answer based on the context above:`,
          },
        ],
      });

      return (
        completion.choices?.[0]?.message?.content?.trim() ??
        "I apologize, but I couldn't generate a response. Please try rephrasing your question."
      );
    } catch (error) {
      console.error("Answer generation error:", error);
      throw error;
    }
  };

  // Type assertion to ensure the result is always a string
  const result = (await chatQueue.add(queuedTask)) as string;
  return result;
}

// Main RAG function with comprehensive optimization and updated response format
async function rag(question: string, namespace?: string) {
  if (isGreeting(question)) {
    return {
      query: question,
      response: [],
      answer: generateGreetingResponse(question),
      sources: [],
      latencySec: 0,
    };
  }

  if (isSmallTalk(question)) {
    return {
      query: question,
      response: [],
      answer: generateSmallTalkResponse(question),
      sources: [],
      latencySec: 0,
    };
  }

  const t0 = performance.now();

  // Check semantic cache first (include namespace in cache key)
  const semanticCacheKey = generateSemanticCacheKey(question, namespace);
  const cached = SEMANTIC_CACHE.get(semanticCacheKey);
  if (cached) {
    console.log(`Semantic cache hit for namespace: ${namespace || 'default'}`);
    return {
      ...cached,
      latencySec: (performance.now() - t0) / 1000,
    };
  }

  // Optimized parameters
  const topK = Math.min(25, 40);
  const maxContextChars = 5000;
  const maxDocs = 4;

  try {
    console.log(`🤖 Processing RAG query with namespace: ${namespace || 'default'}`);
    
    // Parallel execution where possible
    const [denseMatches] = await Promise.all([
      retrieve(question, topK, namespace),
    ]);

    console.log(`📄 Retrieved ${denseMatches.length} matches from namespace: ${namespace || 'default'}`);

    if (!denseMatches.length) {
      const result = {
        query: question,
        response: [],
        answer:
          "I couldn't find relevant information to answer your question from the current knowledge base.",
        sources: [],
        latencySec: (performance.now() - t0) / 1000,
      };

      SEMANTIC_CACHE.set(semanticCacheKey, result);
      return result;
    }

    const docs = denseMatches
      .map((m) => ({
        id: String(m.id),
        text: extractText(m.metadata || {}),
      }))
      .filter((d) => d.text.length > 10);

    // Parallel BM25 reranking
    const [bm25Scores] = await Promise.all([
      Promise.resolve(bm25Rerank(question, docs, namespace)),
    ]);

    const denseScoreMap = new Map<string, number>();
    for (const m of denseMatches) {
      denseScoreMap.set(String(m.id), m.score ?? 0);
    }

    const blended = blendScores(denseScoreMap, bm25Scores, 0.65);

    // Optimized ranking and context building
    const byId = new Map(denseMatches.map((m) => [String(m.id), m]));
    const ranked = [...blended.entries()]
      .map(([id, score]) => {
        const m = byId.get(id)!;
        return { ...m, score, id };
      })
      .filter((m) => {
        const text = extractText(m.metadata || {});
        return text.length > 20;
      })
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

    // Optimized context building with early termination and document tracking
    const chunks: string[] = [];
    const sources: Array<{ id: string; source?: string; score: number }> = [];
    const documentFiles: string[] = [];
    let used = 0;

    for (const m of ranked.slice(0, maxDocs)) {
      const md = (m.metadata || {}) as Meta;
      const chunk = clean(extractText(md));
      if (!chunk || chunk.length < 30) continue;

      if (used + chunk.length > maxContextChars && chunks.length >= 2) break;

      chunks.push(chunk);
      
      const sourceInfo = {
        id: String(m.id),
        score: Number(m.score ?? 0),
        source:
          (md.source as string) || (md.url as string) || md.file || "unknown",
      };
      sources.push(sourceInfo);

      // Extract document filename for the response array
      const documentFile = extractDocumentFilename(md);
      if (!documentFiles.includes(documentFile)) {
        documentFiles.push(documentFile);
      }

      used += chunk.length;
      if (used >= maxContextChars) break;
    }

    const context = chunks.join("\n\n---\n\n");
    const answer = await generateAnswer(question, context);

    const result = {
      query: question,
      response: documentFiles, // Array of document filenames as required
      answer:
        answer || "I don't have enough information in the provided context.",
      sources,
      latencySec: (performance.now() - t0) / 1000,
    };

    // Cache the result
    SEMANTIC_CACHE.set(semanticCacheKey, result);

    console.log(`✅ RAG completed for namespace ${namespace || 'default'}: found ${documentFiles.length} source documents`);

    return result;
  } catch (error) {
    console.error("Error in RAG process:", error);
    const errorResult = {
      query: question,
      response: [],
      answer:
        "I encountered an error while processing your request. Please try again.",
      sources: [],
      latencySec: (performance.now() - t0) / 1000,
    };

    return errorResult;
  }
}

// Initialize query counter on startup
await initializeQueryCounter();

// Enhanced server with optimized batch processing capabilities
Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    const startTime = performance.now();

    // Enhanced CORS handling
    if (req.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    // Health check with enhanced stats
    if (url.pathname === "/api/v1/health" && req.method === "GET") {
      const batchStatus = getBatchStatus();
      const processedQueries = await getProcessedQueries();
      
      return new Response(
        JSON.stringify({
          status: "ok",
          processingNamespace: PROCESSING_NAMESPACE,
          manualQueryCounter: queryCounter,
          dataDirectory: DATA_DIR,
          queryFilePath: QUERY_FILE_PATH,
          queryFileExists: existsSync(QUERY_FILE_PATH),
          totalProcessedQueries: processedQueries.size,
          batchProcessing: batchStatus,
          optimization: {
            enabled: true,
            description: "Only processes unprocessed queries - no file existence checks during processing",
            skipsDuplicates: true,
            performanceGain: "O(1) vs O(n) per query"
          },
          cacheStats: {
            embeddingCache: {
              size: EMBEDDING_CACHE.size,
              max: EMBEDDING_CACHE.max,
            },
            semanticCache: {
              size: SEMANTIC_CACHE.size,
              max: SEMANTIC_CACHE.max,
            },
            bm25Cache: {
              size: BM25_INSTANCES_CACHE.size,
              max: BM25_INSTANCES_CACHE.max,
            },
          },
          queueStats: {
            embeddingQueue: embeddingQueue.size,
            chatQueue: chatQueue.size,
            batchQueue: batchQueue.size,
          },
        }),
        {
          headers: { "Content-Type": "application/json", ...corsHeaders },
          status: 200,
        }
      );
    }

    // [ALL SERVER ENDPOINTS - Enhanced with optimization info]

    return new Response(JSON.stringify({ message: "Not found" }), {
      headers: { "Content-Type": "application/json", ...corsHeaders },
      status: 404,
    });
  },
});

console.log(`🚀 Optimized Production RAG server with batch processing listening on http://localhost:${PORT}`);
console.log(`✅ Caching enabled: Embedding(${EMBEDDING_CACHE.max}), Semantic(${SEMANTIC_CACHE.max}), BM25(${BM25_INSTANCES_CACHE.max})`);
console.log(`⚡ Rate limiting: Embeddings(50/s), Chat(20/s), Batch(3 concurrent)`);
console.log(`📄 Document tracking enabled - returns source filenames in response`);
console.log(`💾 JSON file saving enabled - files saved to: ${DATA_DIR}`);
console.log(`🔢 Manual query counter initialized at: ${queryCounter}`);
console.log(`📋 Query file path: ${QUERY_FILE_PATH}`);
console.log(`🎯 OPTIMIZED: Only processes unprocessed queries - no file existence checks during processing`);
console.log(`🔖 Processing namespace: ${PROCESSING_NAMESPACE}`);
console.log(`⚡ Performance optimization: O(1) vs O(n) per query - dramatically faster for large datasets`);

// Auto-start optimized batch processing if query.json exists
const shouldAutoStart = existsSync(QUERY_FILE_PATH);
if (shouldAutoStart) {
  console.log(`🚀 Query file found. Starting optimized batch processing with namespace ${PROCESSING_NAMESPACE} in 5 seconds...`);
  setTimeout(() => {
    startBatchProcessing().catch(error => {
      console.error('Auto-start batch processing error:', error);
    });
  }, 5000);
}