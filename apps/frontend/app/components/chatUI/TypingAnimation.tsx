"use client"

import { useEffect, useState, useRef } from "react"
import { useChatStore } from "@/store/chat-store"

interface TypingAnimationProps {
  messageId: string
  content: string
  speed?: number
  onComplete?: () => void
}

export function TypingAnimation({ messageId, content, speed = 10, onComplete }: TypingAnimationProps) {
  const [displayedContent, setDisplayedContent] = useState("")
  const [currentIndex, setCurrentIndex] = useState(0)
  const { updateMessageContent, setMessageComplete, setTypingMessageId } = useChatStore()
  const startTimeRef = useRef(0)
  const currentIndexRef = useRef(0)

  useEffect(() => {
    currentIndexRef.current = 0
    startTimeRef.current = performance.now()
    const tick = () => {
      const now = performance.now()
      const elapsed = now - startTimeRef.current
      const nextIndex = Math.min(Math.floor(elapsed / speed), content.length)

      if (nextIndex > currentIndexRef.current) {
        const newContent = content.slice(0, nextIndex)
        setDisplayedContent(newContent)
        setCurrentIndex(nextIndex)
        currentIndexRef.current = nextIndex
        updateMessageContent(messageId, newContent)
      }

      if (nextIndex < content.length) {
        requestAnimationFrame(tick)
      } else {
        setMessageComplete(messageId)
        setTypingMessageId(null)
        onComplete?.()
      }
    }

    const rAF = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rAF)
  }, [content, speed, messageId, updateMessageContent, setMessageComplete, setTypingMessageId, onComplete])

  return (
    <div className="relative">
      <span className="whitespace-pre-wrap">{displayedContent}</span>
      {currentIndex < content.length && <span className="inline-block w-2 h-5 bg-current animate-pulse ml-1" />}
    </div>
  )
}