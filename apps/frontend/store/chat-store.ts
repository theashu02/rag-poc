'use client'
import { create } from "zustand"
import { devtools, persist } from "zustand/middleware"
import api from "@/lib/ApiStore/axios"

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: number
  isTyping?: boolean
  isComplete?: boolean
}

interface ChatState {
  messages: Message[]
  isLoading: boolean
  error: string | null
  typingMessageId: string | null
}

interface ChatActions {
  addMessage: (message: Message) => void
  updateMessageContent: (id: string, content: string) => void
  setMessageComplete: (id: string) => void
  setTypingMessageId: (id: string | null) => void
  clearError: () => void
  clearMessages: () => void
  sendMessage: (content: string) => Promise<{ messageId: string; content: string }>
}

const generateId = () => {
  // if (typeof window !== "undefined" && window.crypto?.randomUUID) {
  //   return crypto.randomUUID()
  // }
  if(typeof window !== 'undefined') {
    return window.crypto.randomUUID();
  }
  // Fallback for SSR
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

export const useChatStore = create<ChatState & ChatActions>()(
  devtools(
    persist(
      (set, get) => ({
        messages: [],
        isLoading: false,
        error: null,
        typingMessageId: null,

        addMessage: (message) =>
          set((state) => ({
            messages: [...state.messages, message],
          })),

        updateMessageContent: (id, content) =>
          set((state) => ({
            messages: state.messages.map((msg) => (msg.id === id ? { ...msg, content } : msg)),
          })),

        setMessageComplete: (id) =>
          set((state) => ({
            messages: state.messages.map((msg) =>
              msg.id === id ? { ...msg, isTyping: false, isComplete: true } : msg,
            ),
          })),

        setTypingMessageId: (id) => set({ typingMessageId: id }),

        clearError: () => set({ error: null }),

        clearMessages: () => set({ messages: [], typingMessageId: null }),

        sendMessage: async (content) => {
          if (typeof window === "undefined") {
            throw new Error("Cannot send message during SSR")
          }

          const { addMessage, setTypingMessageId } = get()

          set({ isLoading: true, error: null })

          // Add user message immediately
          const userMessage: Message = {
            id: generateId(), // Using SSR-safe ID generation
            role: "user",
            content,
            timestamp: Date.now(),
            isComplete: true,
          }
          addMessage(userMessage)

          // Create assistant message placeholder
          const assistantMessage: Message = {
            id: generateId(), // Using SSR-safe ID generation
            role: "assistant",
            content: "",
            timestamp: Date.now(),
            isTyping: true,
            isComplete: false,
          }
          addMessage(assistantMessage)
          setTypingMessageId(assistantMessage.id)

          try {
            const res = await api.post("/api/v1/query", { query: content })
            const payload = res?.data
            const answer =
              (payload?.answer ??
                payload?.response ??
                payload?.message ??
                (typeof payload === "string" ? payload : JSON.stringify(payload))) ||
              "No response."

            set({ isLoading: false })
            return { messageId: assistantMessage.id, content: answer }
          } catch (error: any) {
            const errorMessage = error?.response?.data?.message ?? error?.message ?? "Failed to send message"

            set((state) => ({
              isLoading: false,
              error: errorMessage,
              messages: state.messages.map((msg) =>
                msg.id === assistantMessage.id
                  ? {
                      ...msg,
                      content: "Sorry, I encountered an error. Please try again.",
                      isTyping: false,
                      isComplete: true,
                    }
                  : msg,
              ),
              typingMessageId: null,
            }))

            throw error
          }
        },
      }),
      {
        name: "chat-storage",
        partialize: (state) => ({ messages: state.messages }),
        skipHydration: true,
      },
    ),
    { name: "chat-store" },
  ),
)