"use client"

import type { Message } from "@/store/chat-store"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { TypingAnimation } from "./TypingAnimation"
import { User, Copy, Check, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useState } from "react"

interface MessageBubbleProps {
  message: Message
  responseContent?: string
}

export function MessageBubble({ message, responseContent }: MessageBubbleProps) {
  const isUser = message.role === "user"
  const isTyping = message.isTyping && !message.isComplete
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    const textToCopy = message.content || responseContent || ""
    await navigator.clipboard.writeText(textToCopy)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={cn("group flex gap-4 max-w-4xl mx-auto px-6 py-6", isUser ? "flex-row-reverse" : "flex-row")}>
      <Avatar
        className={cn(
          "w-11 h-11 shrink-0 shadow-lg ring-2 transition-all duration-200",
          isUser
            ? "bg-gradient-to-br from-primary to-secondary ring-primary/20"
            : "bg-gradient-to-br from-card to-muted ring-border",
        )}
      >
        <AvatarFallback
          className={cn(
            "border-0 transition-colors duration-200",
            isUser ? "text-[#f4f4f4] bg-stone-900 border-2" : "text-gray-300 bg-stone-800 border-2",
          )}
        >
          {isUser ? <User className="w-5 h-5" /> : <Sparkles className="w-5 h-5" />}
        </AvatarFallback>
      </Avatar>

      <div className={cn("flex-1 space-y-3 max-w-[85%]", isUser ? "text-right" : "text-left")}>
        <div className="relative">
          <div
            className={cn(
              "inline-block rounded-2xl px-5 py-4 text-[15px] leading-relaxed shadow-sm transition-all duration-200 hover:shadow-md",
              isUser
                ? "bg-stone-900 border-2 text-amber-100 ml-auto"
                : "bg-card text-card-foreground border border-border backdrop-blur-sm",
            )}
          >
            {isTyping && responseContent ? (
              <TypingAnimation messageId={message.id} content={responseContent} speed={25} />
            ) : (
              <div className="whitespace-pre-wrap break-words">
                {message.content ||
                  (isTyping ? (
                    <div className="flex items-center gap-3">
                      <div className="flex gap-1">
                        <div
                          className="w-2 h-2 bg-primary/60 rounded-full animate-bounce"
                          style={{ animationDelay: "0ms" }}
                        />
                        <div
                          className="w-2 h-2 bg-primary/60 rounded-full animate-bounce"
                          style={{ animationDelay: "150ms" }}
                        />
                        <div
                          className="w-2 h-2 bg-primary/60 rounded-full animate-bounce"
                          style={{ animationDelay: "300ms" }}
                        />
                      </div>
                      <span className="text-muted-foreground text-sm font-medium">Thinking...</span>
                    </div>
                  ) : (
                    ""
                  ))}
              </div>
            )}
          </div>

          {/* Copy button for assistant messages */}
          {!isUser && !isTyping && message.content && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="absolute -bottom-4 right-2 opacity-0 group-hover:opacity-100 transition-all duration-200 h-8 w-8 p-0 bg-card border border-border rounded-full shadow-sm hover:shadow-md hover:scale-105"
            >
              {copied ? (
                <Check className="w-3 h-3 text-green-600" />
              ) : (
                <Copy className="w-3 h-3 text-muted-foreground" />
              )}
            </Button>
          )}
        </div>

        <div className={cn("text-xs text-muted-foreground px-3 font-medium", isUser ? "text-right" : "text-left")}>
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </div>
      </div>
    </div>
  )
}