function sanitize(text: string): string {
  if (typeof text !== "string") {
    throw new TypeError("Input must be a string");
  }

  if (text.length === 0) {
    return "";
  }

  let s = text.normalize("NFD");
  s = s.replace(/[\u0300-\u036f]/g, "");

  try {
    s = s.replace(/[^\p{L}\p{N}\s]/gu, " ");
  } catch {
    s = s.replace(/[^\w\s]/g, " ");
  }

  s = s.replace(/\s+/g, " ").trim();
  return s.toLowerCase();
}

function cleanSync(text: string): string {
  try {
    return sanitize(text);
  } finally {
    // something
  }
}

export async function clean(text: string): Promise<string> {
  try {
    return cleanSync(text);
  } finally {
    // something
  }
}
