"use client"

import { useEffect, useState } from "react"
import { useSession } from "next-auth/react"

export default function AnimatedGreeting() {
  const [isAnimating, setIsAnimating] = useState(false)
  const [animationComplete, setAnimationComplete] = useState(false)
  const { data: session } = useSession();

  useEffect(() => {
    // Start animation after component mounts
    const timer = setTimeout(() => setIsAnimating(true), 100)

    const completeTimer = setTimeout(() => setAnimationComplete(true), 3000)

    return () => {
      clearTimeout(timer)
      clearTimeout(completeTimer)
    }
  }, [])

  return (
    <div className={`text-wrapper`}>
      <h1
        className={`text-3xl md:text-3xl font-bold transition-all duration-1000 ${
          isAnimating ? "animate-gradient-sweep" : ""
        }`}
        style={{
          backgroundImage: animationComplete
            ? "linear-gradient(to right, rgb(59, 130, 246), rgb(147, 197, 253), rgb(30, 64, 175))"
            : "linear-gradient(to right, transparent 0%, transparent 30%, rgb(59, 130, 246) 35%, rgb(147, 197, 253) 45%, rgb(30, 64, 175) 55%, transparent 65%, transparent 100%)",
          backgroundSize: animationComplete ? "100% 100%" : "200% 100%",
          backgroundPosition: isAnimating ? "100% 0px" : "-100% 0px",
          WebkitBackgroundClip: "text",
          backgroundClip: "text",
          WebkitTextFillColor: "transparent",
          transition: "background-position 3s ease-in-out",
        }}
      >
        Hello, {session?.user?.name}
      </h1>
    </div>
  )
}
