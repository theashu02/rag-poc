import express from "express";
import cors from "cors";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

const app = express();
const PORT = process.env.PORT || 5000;
const OpenAI_Key = process.env.OPENAI_API_KEY;
const Pinecone_Key = process.env.PINECONE_API_KEY;
const Pinecone_Index = process.env.PINECONE_INDEX;

app.use(cors({ origin: "http://localhost:3000" }));
app.use(express.json());

app.get("/api/v1/health", (req, res) => {
  res.status(200).json({ status: "Endpoints is working from the api folder!" });
});

app.post("/api/v1/query", async (req, res) => {
  try {
    const { query, topK = 5, namespace } = req.body ?? {};
    if (!query || typeof query !== "string") {
      return res
        .status(400)
        .json({ message: "Invalid 'query' in request body." });
    }

    if (!OpenAI_Key) {
      return res.status(500).json({ message: "Missing env: OPENAI_API_KEY" });
    }

    if (!Pinecone_Key || !Pinecone_Index) {
      return res
        .status(500)
        .json({ message: "Missing env: PINECONE_API_KEY or PINECONE_INDEX" });
    }

    const openai = new OpenAI({ apiKey: OpenAI_Key });

    const emb = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: query,
    });

    const vector = emb.data[0]?.embedding;
    if (!vector) {
      return res.status(500).json({ message: "Failed to create embeddings" });
    }

    const pinecone = new Pinecone({ apiKey: Pinecone_Key });
    const index = pinecone.Index(Pinecone_Index!);

    const result = await index.query({
      vector,
      topK: Math.min(Math.max(1, Number(topK) || 5), 20),
      includeMetadata: true,
      namespace: namespace || undefined,
    });

    const matches = result.matches ?? [];
    const answer = matches
      .map((m) => {
        const md = (m.metadata ?? {}) as Record<string, any>;
        return md.text || md.content || md.pageContent || "";
      })
      .filter(Boolean)
      .join("\n\n");

    const sources = matches.map((m) => ({
      id: m.id,
      score: m.score,
      metadata: m.metadata,
    }));

    return res.json({ answer, sources });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ message: "Internal Error" });
  }
});

app.listen(PORT);

console.log(`The api server is listening on the port ${PORT}`);