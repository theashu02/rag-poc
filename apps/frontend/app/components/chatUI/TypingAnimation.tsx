"use client"

import { useEffect, useState } from "react"
import { useChatStore } from "@/store/chat-store"

interface TypingAnimationProps {
  messageId: string
  content: string
  speed?: number
  onComplete?: () => void
}

export function TypingAnimation({ messageId, content, speed = 30, onComplete }: TypingAnimationProps) {
  const [displayedContent, setDisplayedContent] = useState("")
  const [currentIndex, setCurrentIndex] = useState(0)
  const { updateMessageContent, setMessageComplete, setTypingMessageId } = useChatStore()

  useEffect(() => {
    if (currentIndex < content.length) {
      const timer = setTimeout(() => {
        const nextChar = content[currentIndex]
        const newContent = displayedContent + nextChar

        setDisplayedContent(newContent)
        setCurrentIndex(currentIndex + 1)

        updateMessageContent(messageId, newContent)
      }, speed)

      return () => clearTimeout(timer)
    } else if (content.length > 0) {
      // Animation complete
      setMessageComplete(messageId)
      setTypingMessageId(null)
      onComplete?.()
    }
  }, [
    currentIndex,
    content,
    displayedContent,
    messageId,
    speed,
    updateMessageContent,
    setMessageComplete,
    setTypingMessageId,
    onComplete,
  ])

  return (
    <div className="relative">
      <span className="whitespace-pre-wrap">{displayedContent}</span>
      {currentIndex < content.length && <span className="inline-block w-2 h-5 bg-current animate-pulse ml-1" />}
    </div>
  )
}