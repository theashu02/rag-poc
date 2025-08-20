"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { CheckCircle, Sparkles } from "lucide-react"

interface WelcomeProps {
  userName?: string
  onComplete?: () => void
  duration?: number
}

function Welcome({ userName = "User", onComplete, duration = 3000 }: WelcomeProps) {
  const [stage, setStage] = useState<"entering" | "showing" | "exiting" | "hidden">("entering")

  useEffect(() => {
    const timer1 = setTimeout(() => setStage("showing"), 500)
    const timer2 = setTimeout(() => setStage("exiting"), duration - 800)
    const timer3 = setTimeout(() => {
      setStage("hidden")
      onComplete?.()
    }, duration)

    return () => {
      clearTimeout(timer1)
      clearTimeout(timer2)
      clearTimeout(timer3)
    }
  }, [duration, onComplete])

  if (stage === "hidden") return null

  return (
    <div
      className={`
      fixed inset-0 z-50 flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-purple-50
      transition-all duration-1000 ease-in-out
      ${stage === "entering" ? "opacity-0 scale-95" : ""}
      ${stage === "showing" ? "opacity-100 scale-100" : ""}
      ${stage === "exiting" ? "opacity-0 scale-150 blur-sm" : ""}
    `}
    >
      {/* Animated background particles */}
      <div className="absolute inset-0 overflow-hidden">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className={`
              absolute w-2 h-2 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full
              animate-pulse opacity-60
              ${stage === "showing" ? "animate-bounce" : ""}
            `}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 2}s`,
              animationDuration: `${2 + Math.random() * 2}s`,
            }}
          />
        ))}
      </div>

      {/* Main welcome card */}
      <Card
        className={`
        relative p-12 max-w-md mx-4 text-center shadow-2xl border-0
        bg-white/80 backdrop-blur-sm
        transition-all duration-700 ease-out
        ${stage === "entering" ? "translate-y-8 opacity-0" : ""}
        ${stage === "showing" ? "translate-y-0 opacity-100" : ""}
        ${stage === "exiting" ? "translate-y-0 opacity-100 scale-110" : ""}
      `}
      >
        {/* Success icon */}
        <div
          className={`
          flex justify-center mb-6
          transition-all duration-500 delay-300
          ${stage === "entering" ? "scale-0 rotate-180" : ""}
          ${stage === "showing" ? "scale-100 rotate-0" : ""}
        `}
        >
          <div className="relative">
            <CheckCircle className="w-16 h-16 text-green-500" />
            <div className="absolute -top-2 -right-2">
              <Sparkles
                className={`
                w-6 h-6 text-yellow-400
                transition-all duration-700 delay-500
                ${stage === "showing" ? "animate-spin" : ""}
              `}
              />
            </div>
          </div>
        </div>

        {/* Welcome text */}
        <div
          className={`
          space-y-4
          transition-all duration-500 delay-500
          ${stage === "entering" ? "translate-y-4 opacity-0" : ""}
          ${stage === "showing" ? "translate-y-0 opacity-100" : ""}
        `}
        >
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Welcome!
          </h1>
          <p className="text-xl text-gray-700 font-medium">Hello, {userName}</p>
          <p className="text-gray-500">{"You're successfully logged in"}</p>
        </div>

        {/* Animated progress bar */}
        <div
          className={`
          mt-8 w-full bg-gray-200 rounded-full h-1 overflow-hidden
          transition-all duration-300 delay-700
          ${stage === "entering" ? "opacity-0" : ""}
          ${stage === "showing" ? "opacity-100" : ""}
        `}
        >
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-1000 ease-out"
            style={{
              width: stage === "showing" ? "100%" : "0%",
              transitionDelay: "800ms",
            }}
          />
        </div>
      </Card>

      <div
        className={`
        absolute inset-0 bg-gradient-radial from-transparent via-white/20 to-white
        transition-all duration-800 ease-in-out pointer-events-none
        ${stage === "exiting" ? "opacity-100 scale-150" : "opacity-0 scale-100"}
      `}
      />
    </div>
  )
}

export default Welcome