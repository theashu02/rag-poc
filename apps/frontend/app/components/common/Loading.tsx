"use client"

import React from "react"
import { cn } from "@/lib/utils"

type Size = "sm" | "md" | "lg" | "xl"
type Color = "blue" | "purple" | "pink" | "green" | "orange" | "red"

interface LoadingProps {
  size?: Size
  color?: Color
}

export default function Loading({ size = "lg", color = "purple" }: LoadingProps) {
  const sizeClasses: Record<Size, string> = {
    sm: "h-6 w-6",
    md: "h-10 w-10",
    lg: "h-16 w-16",
    xl: "h-24 w-24",
  }

  const solidColorClasses: Record<Color, string> = {
    blue: "bg-blue-500",
    purple: "bg-purple-500",
    pink: "bg-pink-500",
    green: "bg-green-500",
    orange: "bg-orange-500",
    red: "bg-red-500",
  }

  return (
    <div className="flex items-center justify-center w-screen h-screen">
      <div className={cn("relative", sizeClasses[size])}>
        <div
          className={cn(
            "absolute inset-0 rounded-full animate-ping opacity-20",
            solidColorClasses[color]
          )}
        />
        <div
          className={cn("absolute inset-1 animate-pulse", solidColorClasses[color])}
          style={{
            borderRadius: "30% 70% 70% 30% / 30% 30% 70% 70%",
            animation: "morph 2s ease-in-out infinite",
          }}
        />
        {/* keyframes live inside the component to stay self-contained */}
        <style jsx>{`
          @keyframes morph {
            0%,
            100% {
              border-radius: 30% 70% 70% 30% / 30% 30% 70% 70%;
            }
            25% {
              border-radius: 58% 42% 75% 25% / 76% 46% 54% 24%;
            }
            50% {
              border-radius: 50% 50% 33% 67% / 55% 27% 73% 45%;
            }
            75% {
              border-radius: 33% 67% 58% 42% / 63% 68% 32% 37%;
            }
          }
        `}</style>
      </div>
    </div>
  )
}