import OpenAI from "openai";

const OPENAI_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_KEY) throw new Error("Missing env: OPENAI_API_KEY");

export const openai = new OpenAI({
    apiKey: OPENAI_KEY,
    timeout: 30_000,
    maxRetries: 2
})