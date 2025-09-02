'use client'

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  AudioWaveform, 
  Video, 
  GitBranch, 
  FileText, 
  ChevronDown,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  Menu
} from "lucide-react";
import { cn } from "@/lib/utils";
import { FileUploader } from "./FileUpload";

interface StudioSidebarProps {
  isCollapsed?: boolean;
  onToggle?: () => void;
}

export function StudioSidebar({ isCollapsed = false, onToggle }: StudioSidebarProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const studioFeatures = [
    {
      icon: AudioWaveform,
      title: "Audio Overview",
      description: "Generate audio summaries"
    }
  ];

  return (
    <div 
      className={cn(
        "h-screen bg-studio-bg border-l border-studio-border transition-all duration-300 z-50",
        isCollapsed ? "w-16" : "w-80 md:w-[420px]"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-studio-border">
        <h2 className={cn(
          "text-lg font-semibold text-studio-text transition-opacity duration-200",
          isCollapsed && "opacity-0 w-0 overflow-hidden"
        )}>
          Upload
        </h2>
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggle}
          className="text-studio-text hover:bg-studio-card-hover"
        >
          {isCollapsed ? <PanelRightOpen size={20} /> : <PanelRightClose size={20} />}
        </Button>
      </div>

      {/* Collapsed State */}
      {/* {isCollapsed && (
        <div className="flex flex-col items-center p-4 space-y-4">
          <Button
            variant="ghost"
            size="sm"
            className="text-studio-text hover:bg-studio-card-hover"
          >
            <AudioWaveform size={20} />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-studio-text hover:bg-studio-card-hover"
          >
            <Video size={20} />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-studio-text hover:bg-studio-card-hover"
          >
            <GitBranch size={20} />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-studio-text hover:bg-studio-card-hover"
          >
            <FileText size={20} />
          </Button>
        </div>
      )} */}

      {/* Expanded State */}
      {!isCollapsed && (
        <div className="flex flex-col h-full p-3">
          <FileUploader />
          {/* Language Input Section */}
          {/* <div className="p-4">
            <div 
              className={cn(
                "bg-studio-card rounded-lg p-4 border border-studio-border transition-all duration-200",
                isExpanded && "ring-2 ring-studio-accent"
              )}
              onClick={() => setIsExpanded(!isExpanded)}
            >
              <div className="text-sm text-studio-text-muted mb-2">
                Create an Audio Overview in:
              </div>
              
            </div>
          </div> */}
        </div>
      )}
    </div>
  );
}