import { create } from "zustand"
import { devtools, persist } from "zustand/middleware"

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

export const useChatStore = create<ChatState & ChatActions>()(
  devtools(
    persist(
      (set, get) => ({
        // State
        messages: [],
        isLoading: false,
        error: null,
        typingMessageId: null,

        // Actions
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
          const { addMessage, setTypingMessageId } = get()

          set({ isLoading: true, error: null })

          // Add user message immediately
          const userMessage: Message = {
            id: crypto.randomUUID(),
            role: "user",
            content,
            timestamp: Date.now(),
            isComplete: true,
          }
          addMessage(userMessage)

          // Create assistant message placeholder
          const assistantMessage: Message = {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "",
            timestamp: Date.now(),
            isTyping: true,
            isComplete: false,
          }
          addMessage(assistantMessage)
          setTypingMessageId(assistantMessage.id)

          try {
            // Simulate realistic API response with delay
            await new Promise((resolve) => setTimeout(resolve, 800))

            const mockResponses = [
              "I understand your question. Let me provide you with a comprehensive answer that demonstrates the smooth typing animation feature you requested.",
              "This is a demonstration of the modern chat interface with smooth typing animations, similar to ChatGPT, Claude, and Gemini. The response appears character by character for a natural conversation flow.",
              "Great question! I'm here to help you with any queries you might have. This interface handles large responses gracefully and maintains smooth performance even with extensive conversations.",
              "The chat interface now uses Zustand for lightweight state management, providing excellent performance and developer experience. All responses are handled smoothly with proper error handling and typing animations.",
              "This modern chat interface includes features like auto-scrolling, responsive design, dark mode support, and industry-standard UX patterns. Feel free to ask me anything!",
            ]

            const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)]

            set({ isLoading: false })
            return { messageId: assistantMessage.id, content: randomResponse }
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : "Failed to send message"

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
      },
    ),
    { name: "chat-store" },
  ),
)