"use client";

import { useEffect, useRef, useState } from "react";
import { useChatStore } from "@/store/chat-store";
import { MessageBubble } from "./MessageBubble";
import { ChatInput } from "./ChatInput";
import { Button } from "@/components/ui/button";
import { Trash2, MessageSquare, Sparkles, Zap, Brain, Lightbulb } from "lucide-react";

export function ChatInterface() {
  const { messages, isLoading, error, typingMessageId, sendMessage, clearMessages } = useChatStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [pendingResponse, setPendingResponse] = useState<string>("");

  // Auto-scroll to bottom with smooth behavior
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    setPendingResponse("");
    try {
      const result = await sendMessage(content);
      setPendingResponse(result.content);
    } catch (error) {
      // error is already handled in the store and UI, keep UI stable
    }
  };

  const handleClearChat = () => {
    clearMessages();
    setPendingResponse("");
  };

  const suggestedPrompts = [
    {
      icon: Brain,
      text: "Explain quantum computing",
      color: "from-purple-500 to-violet-600",
    },
    {
      icon: Lightbulb,
      text: "Help me brainstorm ideas",
      color: "from-amber-500 to-orange-600",
    },
    {
      icon: Sparkles,
      text: "Write a creative story",
      color: "from-pink-500 to-rose-600",
    },
    {
      icon: Zap,
      text: "Analyze this problem",
      color: "from-blue-500 to-indigo-600",
    },
  ];

  return (
    <div className="flex flex-col h-screen w-screen bg-gradient-to-br from-background via-card to-background">
      {/* Header */}
      <div className="absolute top-16 right-6">
        {messages.length > 0 && (
          <Button
            variant="destructive"
            size="sm"
            onClick={handleClearChat}
            className="text-[#f4f4f4] hover:text-gray-300 hover:bg-muted rounded-lg transition-all duration-200"
          >
            <Trash2 className="w-4 h-4 mr-1" />
            Clear Chat
          </Button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full p-6">
            <div className="text-center max-w-2xl mx-auto">
              <div className="relative mb-8">
                <div className="w-24 h-24 rounded-3xl bg-stone-800 flex items-center justify-center mx-auto shadow-2xl">
                  <MessageSquare className="w-12 h-12" />
                </div>
              </div>

              <h2 className="text-4xl font-bold mb-4">
                Welcome to AI Assistant
              </h2>
              <p className="text-lg text-muted-foreground mb-8 leading-relaxed">
                Experience the future of communication with our advanced AI chat
                interface, designed to make every interaction seamless and
                intuitive.
              </p>

              {/* Suggested Prompts */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-2xl mx-auto">
                {suggestedPrompts.map((prompt, index) => {
                  const IconComponent = prompt.icon;
                  return (
                    <button
                      key={index}
                      onClick={() => handleSendMessage(prompt.text)}
                      className="group p-5 text-left bg-card rounded-2xl border border-border hover:border-primary/30 hover:shadow-lg transition-all duration-300 hover:scale-105 hover:-translate-y-1"
                    >
                      <div className="flex items-center gap-3 mb-2">
                        <div
                          className={`w-8 h-8 rounded-lg bg-gradient-to-r ${prompt.color} flex items-center justify-center shadow-sm`}
                        >
                          <IconComponent className="w-4 h-4 text-white" />
                        </div>
                        <div className="text-sm font-semibold text-foreground group-hover:text-primary transition-colors">
                          {prompt.text}
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Click to start this conversation
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="mt-8">
                <p className="text-sm text-muted-foreground">
                  Or type your own message below to get started
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="py-6">
            {messages.map((message) => (
              <div key={message.id} className="message-slide-in">
                <MessageBubble
                  message={message}
                  responseContent={
                    message.id === typingMessageId ? pendingResponse : undefined
                  }
                />
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="max-w-4xl mx-auto px-6 py-2">
            <div className="bg-destructive/10 border border-destructive/20 rounded-2xl p-4 backdrop-blur-sm">
              <p className="text-sm text-destructive font-medium">{error}</p>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <ChatInput
        onSend={handleSendMessage}
        disabled={isLoading}
        placeholder="Ask me anything..."
      />
    </div>
  );
}
