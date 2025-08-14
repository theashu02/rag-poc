"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import api from "@/lib/axios";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await api.post("/api/v1/query", { query: text });
      const payload = res?.data;
      const answer = (payload?.answer ??
        payload?.response ??
        payload?.message ??
        (typeof payload === "string"
          ? payload
          : JSON.stringify(payload))) as string;

      const botMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: answer || "No response.",
      };
      setMessages((m) => [...m, botMsg]);
    } catch (e: any) {
      const errMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: e?.response?.data?.message ?? e?.message ?? "Request failed.",
      };
      setMessages((m) => [...m, errMsg]);
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="flex flex-col min-w-full bg-gray-800">
      <div className="flex-1 overflow-y-auto p-4 sm:p-6" ref={listRef}>
        <div className="max-w-3xl mx-auto space-y-3">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`rounded-2xl px-4 py-2 text-sm sm:text-base max-w-[80%] whitespace-pre-wrap ${
                  m.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-900"
                }`}
              >
                {m.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl px-4 py-2 text-sm bg-gray-100 text-gray-600">
                Thinking…
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="border-t">
        <div className="max-w-3xl mx-auto p-4 sm:p-6">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder="Send a message…"
              className="flex-1 resize-none rounded-md border p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <Button onClick={send} disabled={loading || !input.trim()}>
              {loading ? "Sending…" : "Send"}
            </Button>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Press Enter to send, Shift+Enter for a new line.
          </p>
        </div>
      </div>
    </div>
  );
}
