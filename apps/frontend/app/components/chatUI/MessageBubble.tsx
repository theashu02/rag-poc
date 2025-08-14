"use client"

import type { Message } from "@/store/chat-store"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { TypingAnimation } from "./TypingAnimation"
import { User, Bot } from "lucide-react"
import { cn } from "@/lib/utils"

interface MessageBubbleProps {
  message: Message
  responseContent?: string
}

export function MessageBubble({ message, responseContent }: MessageBubbleProps) {
  const isUser = message.role === "user"
  const isTyping = message.isTyping && !message.isComplete

  return (
    <div className={cn("flex gap-3 max-w-4xl mx-auto px-4 py-6", isUser ? "flex-row-reverse" : "flex-row")}>
      <Avatar className={cn("w-8 h-8 shrink-0", isUser ? "bg-blue-600" : "bg-gray-200 dark:bg-gray-700")}>
        <AvatarFallback className={cn(isUser ? "text-white" : "text-gray-600 dark:text-gray-300")}>
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </AvatarFallback>
      </Avatar>

      <div className={cn("flex-1 space-y-2", isUser ? "text-right" : "text-left")}>
        <div
          className={cn(
            "inline-block rounded-2xl px-4 py-3 max-w-[85%] text-sm leading-relaxed",
            isUser ? "bg-blue-600 text-white ml-auto" : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100",
          )}
        >
          {isTyping && responseContent ? (
            <TypingAnimation messageId={message.id} content={responseContent} speed={20} />
          ) : (
            <div className="whitespace-pre-wrap break-words">{message.content || (isTyping ? "Thinking..." : "")}</div>
          )}
        </div>

        <div className={cn("text-xs text-gray-500 dark:text-gray-400 px-2", isUser ? "text-right" : "text-left")}>
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </div>
      </div>
    </div>
  )
}