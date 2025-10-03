import OpenAI from "openai";
import { OPENAI_CHAT_MODEL } from "./config";
import { PromptForGenerateAnswer, chatQueue } from "./config";

const OPENAI_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_KEY) {
  console.error("---- Missing env: OPENAI_API_KEY ----");
  throw new Error("---- Missing env: OPENAI_API_KEY ----");
} else {
  console.log("---- OPENAI_API_KEY is present ----");
}

export const openai = new OpenAI({
  apiKey: OPENAI_KEY,
  timeout: 30_000,
  maxRetries: 2
});


async function generateAnswer(question: string, context: string): Promise<string> {
  const queuedTask = async (): Promise<string> => {
    try {
      const completion = await openai.chat.completions.create({
        model: OPENAI_CHAT_MODEL,
        // if using the actual open ai model then un-comment this feature and change the model name 
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
  console.log("Answer generation result:", result);
  return result;
}