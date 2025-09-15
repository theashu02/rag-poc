import React from 'react'
import { InteractiveHero } from './AnimatedHero';

export default function Hero() {
  return (
    <main className="h-screen w-screen text-foreground">
      <div className="h-screen max-w-screen">
          <InteractiveHero
            title="INTELLIGENCE&nbsp;ANALYTICS,&nbsp;FINALLY."
            subtitle="A lightweight, accessible glow-field that springs to your pointer and pauses when you leave—clean, performant, and production-ready."
          />
        </div>
    </main>
  )
}
