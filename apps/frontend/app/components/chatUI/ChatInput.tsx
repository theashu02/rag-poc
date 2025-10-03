"use client";

import type React from "react";
import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Square, ArrowUp, Plus } from "lucide-react";
import { cn } from "@/lib/utils";
// import { FileUploader } from "../common/FileUpload";
// import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Instant insights...",
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false)

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
    <div className="backdrop-blur-xl bg-card/80 border-t border-border/50 p-6">
      <div className="max-w-2xl mx-auto">
        <div className="relative flex items-end gap-4">
          <div className="flex-1 relative">
            <div className="relative rounded-2xl bg-background border border-gray-800/90 dark:border-gray-500/90 transition-all duration-200 hover:border-primary/30">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={placeholder}
                disabled={disabled}
                className={cn(
                  "min-h-[60px] max-h-[200px] resize-none border bg-transparent rounded-2xl",
                  "focus:ring-0 focus:outline-none",
                  "px-6 py-5 pr-32 text-[15px] leading-relaxed placeholder:text-muted-foreground",
                  "font-normal"
                )}
                rows={1}
              />

              <div className="absolute right-3 bottom-3 flex items-center gap-1">
                {/* <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                  <DialogTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-9 w-9 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted transition-all duration-200 hover:scale-105"
                    >
                      <Plus className="w-4 h-4" />
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                      <DialogTitle>Upload Files</DialogTitle>
                    </DialogHeader>
                    <FileUploader />
                  </DialogContent>
                </Dialog> */}
                {/* <Button
                  variant="ghost"
                  size="sm"
                  disabled
                  className="h-9 w-9 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted transition-all duration-200 hover:scale-105"
                >
                  <Paperclip className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled
                  className="h-9 w-9 rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted transition-all duration-200 hover:scale-105"
                >
                  <Mic className="w-4 h-4" />
                </Button> */}
              </div>
            </div>
          </div>

          <Button
            onClick={handleSend}
            disabled={disabled || !input.trim()}
            size="sm"
            className={cn(
              "h-[60px] w-[60px] rounded-full shrink-0 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-105",
              "bg-stone-900 border-4 text-[#f4f4f4] hover:bg-stone-800/90 hover:text-blue-100",
              "disabled:from-muted disabled:to-muted disabled:text-muted-foreground",
              "disabled:shadow-none disabled:hover:scale-100"
            )}
          >
            {disabled ? (
              <Square className="size-6" />
            ) : (
              <ArrowUp className="size-6" />
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
