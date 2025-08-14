"use client"

import { useEffect, useRef, useState } from "react"
import { useChatStore } from "@/store/chat-store"
import { MessageBubble } from "./MessageBubble"
import { ChatInput } from "./ChatInput"
import { Button } from "@/components/ui/button"
import { Trash2, MessageSquare } from "lucide-react"

export function ChatInterface() {
  const { messages, isLoading, error, typingMessageId, sendMessage, clearMessages } = useChatStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [pendingResponse, setPendingResponse] = useState<string>("")

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Handle API response for typing animation
  useEffect(() => {
    if (typingMessageId && !pendingResponse) {
      // Simulate getting response - replace with your actual API integration
      const simulateResponse = async () => {
        // This would be your actual API response
        const mockResponse =
          "This is a simulated response that will be typed out character by character to create a smooth, professional chat experience similar to ChatGPT, Claude, and Gemini. The typing animation makes the interaction feel more natural and engaging for users."
        setPendingResponse(mockResponse)
      }

      const timer = setTimeout(simulateResponse, 500)
      return () => clearTimeout(timer)
    }
  }, [typingMessageId, pendingResponse])

  const handleSendMessage = async (content: string) => {
    setPendingResponse("")
    try {
      const result = await sendMessage(content)
      setPendingResponse(result.content)
    } catch (error) {
      console.error("Failed to send message:", error)
    }
  }

  const handleClearChat = () => {
    clearMessages()
    setPendingResponse("")
  }

  return (
    <div className="flex flex-col h-screen min-w-5xl bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="border-b bg-white dark:bg-gray-800 px-4 py-3">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
              <MessageSquare className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-gray-900 dark:text-gray-100">AI Assistant</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">Always here to help</p>
            </div>
          </div>

          {messages.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearChat}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Clear
            </Button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md mx-auto px-4">
              <div className="w-16 h-16 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center mx-auto mb-4">
                <MessageSquare className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              </div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">Start a conversation</h2>
              <p className="text-gray-500 dark:text-gray-400">
                Ask me anything! I'm here to help with your questions and tasks.
              </p>
            </div>
          </div>
        ) : (
          <div className="py-4">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                responseContent={message.id === typingMessageId ? pendingResponse : undefined}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="max-w-4xl mx-auto px-4 py-2">
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3">
              <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <ChatInput onSend={handleSendMessage} disabled={isLoading} placeholder="Type your message..." />
    </div>
  )
}
