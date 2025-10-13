import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { System_Prompt, Human_Prompt } from "./prompts";
import { chatModel } from "./openai";

export async function reformulateQuery(originalQuery: string) {
  const systemPrompt = System_Prompt;
  const humanPrompt = `Please reformulate the following user query: "${originalQuery}" ${Human_Prompt} Now, reformulate this query: "${originalQuery}"`;

  try {
    
    const response = await chatModel.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(humanPrompt),
    ]);

    const content = response.content.toString();
    
    const cleanedContent = content.replace(/```json\n|```/g, '').trim();

    const parsedJson = JSON.parse(cleanedContent);
    
    console.log("✅ Query reformulation successful.");
    return parsedJson;

  } catch (error) {
    console.error("❌ Error reformulating query:", error);
    return {
      hypotheticalAnswer: `An answer to the query: ${originalQuery}`,
      subQuestions: [originalQuery],
      synonymousQuery: originalQuery,
    };
  }
}


// For testing 
const text = "Artificial intelligence helps computers learn from data.";
const ans = await reformulateQuery(text);
console.log(ans);