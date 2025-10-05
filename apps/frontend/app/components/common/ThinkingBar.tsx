"use client";

import type * as React from "react";
import { cn } from "@/lib/utils";

type AIThinkingIndicatorProps = {
  label?: string;
  className?: string;
  durationMs?: number;
  dotSize?: number;
};

export function ThinkingIndicator({
  label = "Thinking",
  className,
  durationMs = 1750,
  dotSize = 8,
}: AIThinkingIndicatorProps) {
  const sheenVar = {
    ["--sheen-duration" as any]: `${durationMs}ms`,
  } as React.CSSProperties;

  const animationDelayTime = ["0ms", "150ms", "300ms"];

  // Component now accepts animationDelay prop
  const JumpingBalls = ({ animationDelay }: { animationDelay: string }) => {
    return (
      <span
        className="shine-dot shrink-0 animate-bounce"
        aria-hidden="true"
        style={{
          ...sheenVar,
          width: dotSize,
          height: dotSize,
          backgroundColor: "currentColor",
          borderRadius: 9999,
          opacity: 0.8,
          animationDelay: animationDelay,
        }}
      />
    );
  };

  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(
        "inline-flex items-center gap-3 text-muted-foreground",
        className
      )}
    >
      <div className="flex gap-1">
        {animationDelayTime.map((delay, index) => (
          <JumpingBalls key={index} animationDelay={delay} />
        ))}
      </div>
      <span className="shine-text" style={sheenVar}>
        {label}
      </span>
    </div>
  );
}

export default ThinkingIndicator;
