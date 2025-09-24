import winkTokenizer from "wink-tokenizer";

let winkUtils: any;
let stringUtils: any;
let tokensUtils: any;
let tokenizer: any;

try {
  winkUtils = require("wink-nlp-utils");
  stringUtils = winkUtils.string;
  tokensUtils = winkUtils.tokens;
  tokenizer = winkTokenizer();
  console.log("---- successfully importing the wink-nlp-utils ----");
} catch (error) {
  console.error("Error importing wink-nlp-utils: ", error);
  throw error;
}

let bm25PrepTasks: any[];

try {
  bm25PrepTasks = [
    (text: string) => stringUtils.lowerCase(text),
    (text: string) =>
      tokenizer
        .tokenize(text)
        .filter((t: any) => t.tag === "word")
        .map((t: any) => t.value),
    (tokens: string[]) => tokensUtils.removeWords(tokens),
    (tokens: string[]) => tokensUtils.stem(tokens),
  ];
} catch (error) {
  bm25PrepTasks = [
    (text: string) =>
      text
        .toLowerCase()
        .replace(/[^\w\s]/g, " ")
        .split(/\s+/)
        .filter((t) => t.length > 2),
  ];
}

export default bm25PrepTasks;