import { performance } from "perf_hooks";
import { OPENAI_CHAT_MODEL, PromptForGenerateAnswer, chatQueue } from "./config";
import { openai } from "./openai";

export async function generateAnswer(question: string, context: string): Promise<string> {
  const startTime = performance.now();
  try {
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
              content: `Context information: ${context}\n 
              Question: ${question}\n           
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

    const result = (await chatQueue.add(queuedTask)) as string;
    return result;
  } finally {
    const duration = performance.now() - startTime;
    // Track total LLM answer generation duration.
    console.log(`[Timing] generateAnswer completed in ${duration.toFixed(2)}ms`);
  }
}
