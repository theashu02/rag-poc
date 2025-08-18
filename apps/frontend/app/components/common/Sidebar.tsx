"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import {
  PlusIcon,
  MessageSquareIcon,
  MenuIcon,
  SettingsIcon,
  MoreHorizontalIcon,
  SparklesIcon,
  ClockIcon,
  StarIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useIsMobile } from "@/hooks/use-mobile";

interface ChatItem {
  id: string;
  title: string;
  timestamp: string;
  isPinned?: boolean;
  isNew?: boolean;
}

const mockChats: ChatItem[] = [
  {
    id: "1",
    title: "React Components Best Practices",
    timestamp: "2 hours ago",
    isPinned: true,
  },
  {
    id: "2",
    title: "TypeScript Advanced Types",
    timestamp: "Yesterday",
    isNew: true,
  },
  { id: "3", title: "Next.js App Router Guide", timestamp: "2 days ago" },
  { id: "4", title: "Tailwind CSS Tips & Tricks", timestamp: "3 days ago" },
  { id: "5", title: "Database Design Patterns", timestamp: "1 week ago" },
  { id: "6", title: "API Authentication Methods", timestamp: "1 week ago" },
  { id: "7", title: "Performance Optimization", timestamp: "2 weeks ago" },
];

function CollapsedSidebarContent({ onClose }: { onClose?: () => void }) {
  const [selectedChat, setSelectedChat] = useState<string | null>("1");

  const handleChatSelect = (chatId: string) => {
    setSelectedChat(chatId);
    onClose?.();
  };

  return (
    <div className="flex h-screen w-full flex-col border-r border-sidebar-border bg-amber-200">
      <div className="flex items-center justify-center p-4 border-b border-sidebar-border">
        <div className="relative">
          <div className="w-10 h-8 bg-gradient-to-br from-sidebar-primary to-accent rounded-xl flex items-center justify-center shadow-sm">
            <SparklesIcon className="w-5 h-5 text-sidebar-primary-foreground" />
          </div>
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-accent rounded-full border-2 border-sidebar"></div>
        </div>
      </div>

      <div className="p-3">
        <Button
          size="icon"
          className="w-full h-11 bg-sidebar-primary hover:bg-sidebar-primary/90 text-sidebar-primary-foreground rounded-xl shadow-sm transition-all duration-200 hover:shadow-md"
          onClick={() => handleChatSelect("")}
        >
          <PlusIcon className="w-5 h-5" />
        </Button>
      </div>

      <ScrollArea className="flex-1 px-2">
        <div className="space-y-2">
          {mockChats.slice(0, 6).map((chat) => (
            <div
              key={chat.id}
              className={cn(
                "relative flex items-center justify-center rounded-xl p-3 cursor-pointer transition-all duration-200 group",
                selectedChat === chat.id
                  ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-sm"
                  : "text-sidebar-foreground hover:bg-sidebar-accent/50 hover:shadow-sm"
              )}
              onClick={() => handleChatSelect(chat.id)}
            >
              <MessageSquareIcon className="w-5 h-5" />
              {chat.isPinned && (
                <div className="absolute -top-1 -right-1 w-2 h-2 bg-accent rounded-full"></div>
              )}
              {chat.isNew && (
                <div className="absolute -top-1 -right-1 w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              )}
            </div>
          ))}
        </div>
      </ScrollArea>

      <div className="border-t border-sidebar-border p-4 flex justify-center">
        <div className="relative">
          <Avatar className="h-10 w-10 ring-2 ring-sidebar-accent">
            <AvatarImage src="/placeholder.svg?height=40&width=40" />
            <AvatarFallback className="bg-sidebar-primary text-sidebar-primary-foreground font-semibold">
              JD
            </AvatarFallback>
          </Avatar>
          <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-sidebar"></div>
        </div>
      </div>
    </div>
  );
}

function SidebarContent({ onClose }: { onClose?: () => void }) {
  const [selectedChat, setSelectedChat] = useState<string | null>("1");

  const handleChatSelect = (chatId: string) => {
    setSelectedChat(chatId);
    onClose?.();
  };

  return (
    <div className="flex h-full w-full flex-col bg-sidebar border-r border-sidebar-border">
      <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 bg-gradient-to-br from-sidebar-primary to-accent rounded-xl flex items-center justify-center shadow-sm">
              <SparklesIcon className="w-5 h-5 text-sidebar-primary-foreground" />
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-accent rounded-full border-2 border-sidebar"></div>
          </div>
          <div>
            <h1 className="font-bold text-lg text-sidebar-foreground">
              AI Chat
            </h1>
            <p className="text-xs text-sidebar-foreground/60">Powered by AI</p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="h-9 w-9 text-sidebar-foreground hover:bg-sidebar-accent rounded-lg"
        >
          <PlusIcon className="w-4 h-4" />
        </Button>
      </div>

      <div className="p-4">
        <Button
          className="w-full justify-start gap-3 h-11 bg-sidebar-primary hover:bg-sidebar-primary/90 text-sidebar-primary-foreground rounded-xl shadow-sm transition-all duration-200 hover:shadow-md font-medium"
          onClick={() => handleChatSelect("")}
        >
          <PlusIcon className="w-5 h-5" />
          New Chat
        </Button>
      </div>

      <ScrollArea className="flex-1 px-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2 px-3 py-2 text-xs font-semibold text-sidebar-foreground/70 uppercase tracking-wider">
            <ClockIcon className="w-3 h-3" />
            Recent Chats
          </div>
          {mockChats.map((chat) => (
            <div
              key={chat.id}
              className={cn(
                "group relative flex items-center gap-3 rounded-xl px-3 py-3 text-sm cursor-pointer transition-all duration-200",
                selectedChat === chat.id
                  ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-sm"
                  : "text-sidebar-foreground hover:bg-sidebar-accent/50 hover:shadow-sm"
              )}
              onClick={() => handleChatSelect(chat.id)}
            >
              <div className="relative">
                <MessageSquareIcon className="w-4 h-4 shrink-0" />
                {chat.isPinned && (
                  <StarIcon className="w-2 h-2 absolute -top-1 -right-1 text-accent fill-current" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <div className="truncate font-medium">{chat.title}</div>
                  {chat.isNew && (
                    <Badge
                      variant="secondary"
                      className="text-xs px-1.5 py-0.5 bg-blue-100 text-blue-700 border-0"
                    >
                      New
                    </Badge>
                  )}
                </div>
                <div className="text-xs text-sidebar-foreground/60 mt-0.5">
                  {chat.timestamp}
                </div>
              </div>
              <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent rounded-lg"
                >
                  <MoreHorizontalIcon className="w-3 h-3" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      <div className="border-t border-sidebar-border p-4">
        <div className="flex items-center gap-3 p-2 rounded-xl hover:bg-sidebar-accent/50 transition-colors cursor-pointer">
          <div className="relative">
            <Avatar className="h-10 w-10 ring-2 ring-sidebar-accent">
              <AvatarImage src="/placeholder.svg?height=40&width=40" />
              <AvatarFallback className="bg-sidebar-primary text-sidebar-primary-foreground font-semibold">
                JD
              </AvatarFallback>
            </Avatar>
            <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-sidebar"></div>
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-sidebar-foreground truncate">
              John Doe
            </div>
            <div className="text-xs text-sidebar-foreground/60 truncate">
              john@example.com
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent rounded-lg"
          >
            <SettingsIcon className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

export function Sidebar() {
  const isMobile = useIsMobile();
  const [open, setOpen] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  if (isMobile) {
    return (
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="fixed top-4 left-4 z-40 md:hidden rounded-xl shadow-sm"
          >
            <MenuIcon className="w-5 h-5" />
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="w-80 p-0">
          <SidebarContent onClose={() => setOpen(false)} />
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <div
      className={cn(
        "hidden md:flex h-screen transition-all duration-300 ease-in-out",
        isHovered ? "w-80" : "w-16"
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {isHovered ? <SidebarContent /> : <CollapsedSidebarContent />}
    </div>
  );
}
