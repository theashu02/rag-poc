export function isGreeting(query: string): boolean {
  const greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "good night", "good noon", "g'day", "bonjour", "salut", "ciao", "namaste", "konnichiwa", "ni hao", "shalom", "hallo", "howdy", "hola", "what's up", "yo", "sup", "hi there", "hello there"];

  const cleanQuery = query.toLowerCase().trim();
  return greetings.some(
    (g) =>
      cleanQuery === g ||
      cleanQuery.startsWith(g + "?") ||
      cleanQuery.startsWith(g + "!") ||
      new RegExp(`\\b${g}\\b`).test(cleanQuery)
  );
}

export function generateGreetingResponse(query: string): string {
  const hour = new Date().getHours();
  let timeGreeting = "Hello";
  if (hour < 12) timeGreeting = "Good morning";
  else if (hour < 18) timeGreeting = "Good afternoon";
  else timeGreeting = "Good evening";

  const cleanQuery = query.toLowerCase().trim();

  if (cleanQuery.includes("how are you")) {
    return `${timeGreeting}! I'm doing well, thank you. How can I help you today?`;
  }

  if (cleanQuery.includes("what's up") || cleanQuery.includes("sup")) {
    return `${timeGreeting}! Not much, just here to help. What do you need?`;
  }

  const responses = [
    `${timeGreeting}! How can I assist you?`,
    `${timeGreeting}! What would you like to know?`,
    `${timeGreeting}! I'm ready to help.`,
    `${timeGreeting}! Let me know what information you're after.`,
    `${timeGreeting}! Ask away when you're ready.`,
  ];
  return responses[Math.floor(Math.random() * responses.length)]!;
}

export function isSmallTalk(query: string): boolean {
  const patterns = [
    "how are you",
    "what's up",
    "how's it going",
    "who are you",
    "what can you do",
    "tell me about yourself",
    "how are things",
    "how is everything",
    "what is going on",
    "what's going on",
    "who am i talking to",
    "what do you know",
    "introduce yourself",
  ];
  const cleanQuery = query.toLowerCase().trim();
  return patterns.some((p) => cleanQuery.includes(p));
}

export function generateSmallTalkResponse(query: string): string {
  const cleanQuery = query.toLowerCase().trim();

  if (
    cleanQuery.includes("how are you") ||
    cleanQuery.includes("how's it going") ||
    cleanQuery.includes("how are things") ||
    cleanQuery.includes("how is everything")
  ) {
    return "I'm doing well, thanks! I'm here to help you find answers. What can I do for you?";
  }

  if (cleanQuery.includes("what's up")) {
    return "Just ready to help you out! What would you like to know?";
  }

  if (
    cleanQuery.includes("who are you") ||
    cleanQuery.includes("what can you do") ||
    cleanQuery.includes("introduce yourself") ||
    cleanQuery.includes("who am i talking to")
  ) {
    return "I'm an AI assistant that helps you find information from your knowledge base. Ask me anything!";
  }

  return "I'm here to help. What would you like to know about?";
}
