"use client";

import type React from "react";
import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { SendHorizontal , Square, Paperclip, Mic, Plus } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Ask me anything...",
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || disabled) return;

    onSend(trimmedInput);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [input]);

  return (
    <div className="backdrop-blur-xl bg-card/80 border-t border-border/50 p-6 shadow-lg">
      <div className="max-w-4xl mx-auto">
        <div className="relative flex items-end gap-4">
          <div className="flex-1 relative">
            <div className="relative rounded-2xl bg-background border border-border shadow-lg hover:shadow-xl transition-all duration-200 hover:border-primary/30">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={placeholder}
                disabled={disabled}
                className={cn(
                  "min-h-[60px] max-h-[200px] resize-none border-0 bg-transparent rounded-2xl",
                  "focus:ring-0 focus:outline-none",
                  "px-6 py-5 pr-32 text-[15px] leading-relaxed placeholder:text-muted-foreground",
                  "font-medium"
                )}
                rows={1}
              />

              {/* Action buttons */}
              <div className="absolute right-3 bottom-3 flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-9 w-9 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted transition-all duration-200 hover:scale-105"
                >
                  <Plus className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-9 w-9 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted transition-all duration-200 hover:scale-105"
                >
                  <Paperclip className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-9 w-9 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted transition-all duration-200 hover:scale-105"
                >
                  <Mic className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>

          <Button
            onClick={handleSend}
            disabled={disabled || !input.trim()}
            size="sm"
            className={cn(
              "h-[60px] w-[60px] rounded-2xl shrink-0 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105",
              "bg-stone-900 border-4 text-[#f4f4f4] hover:bg-stone-800/90 hover:text-blue-100",
              "disabled:from-muted disabled:to-muted disabled:text-muted-foreground",
              "disabled:shadow-none disabled:hover:scale-100"
            )}
          >
            {disabled ? (
              <Square className="w-5 h-5" />
            ) : (
              <SendHorizontal className="w-5 h-5"/>
            )}
          </Button>
        </div>

        <p className="text-xs text-muted-foreground mt-4 px-2 text-center font-medium">
          Press{" "}
          <kbd className="px-2 py-1 bg-muted rounded text-xs font-mono font-semibold">
            Enter
          </kbd>{" "}
          to send,
          <kbd className="px-2 py-1 bg-muted rounded text-xs font-mono font-semibold ml-1">
            Shift+Enter
          </kbd>{" "}
          for new line
        </p>
      </div>
    </div>
  );
}
