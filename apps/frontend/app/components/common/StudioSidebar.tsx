'use client'

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { AudioWaveform, Video, GitBranch, FileText, PanelRightClose, PanelRightOpen} from "lucide-react";
import { cn } from "@/lib/utils";
import { FileUploader } from "./FileUpload";
import { DocListTopFive } from "./DocListTopFive";
import { useRouter } from "next/navigation";
// import DocListTopFive from "./DocListTopFive";

interface StudioSidebarProps {
  isCollapsed?: boolean;
  onToggle?: () => void;
}

export function StudioSidebar({ isCollapsed = false, onToggle }: StudioSidebarProps) {
  // const [isExpanded, setIsExpanded] = useState(false);
  const router = useRouter();
  const handleAllDocs = () => {
     router.push("/documents");
  }

  return (
    <div 
      className={cn(
        "right-0 h-screen bg-studio-bg border-l border-studio-border transition-all duration-300 z-50",
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
      {isCollapsed && (
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
      )}

      {/* Expanded State */}
      {!isCollapsed && (
        <>
          <div className="flex flex-col h-auto gap-2 p-1 mt-1 ml-2">
            <FileUploader />
          </div>
          <div className="flex flex-col h-auto gap-2 p-1 mt-1 ml-2">
            <DocListTopFive />
          </div>
          <div className="flex justify-center items-center w-full py-3 px-10">
            <Button className="rounded-sm w-full cursor-pointer" variant="secondary" onClick={handleAllDocs}>All Documents</Button>
          </div>
        </>
      )}
    </div>
  );
}