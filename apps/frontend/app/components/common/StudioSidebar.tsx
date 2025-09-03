// 'use client'

// import { Button } from '@/components/ui/button'
// import {
//   AudioWaveform,
//   Video,
//   GitBranch,
//   FileText,
//   PanelRightClose,
//   PanelRightOpen,
// } from 'lucide-react'
// import { cn } from '@/lib/utils'
// import { FileUploader } from './FileUpload'
// import { DocListTopFive } from './DocListTopFive'
// import { useRouter } from 'next/navigation'
// import { ScrollArea } from '@/components/ui/scroll-area'

// interface StudioSidebarProps {
//   isCollapsed?: boolean
//   onToggle?: () => void
// }

// export function StudioSidebar({
//   isCollapsed = false,
//   onToggle,
// }: StudioSidebarProps) {
//   const router = useRouter()

//   const handleAllDocs = () => {
//     router.push('/documents')
//   }

//   return (
//     <div
//       className={cn(
//         'right-0 h-screen flex flex-col bg-studio-bg border-l border-studio-border transition-all duration-300 z-50',
//         isCollapsed ? 'w-16' : 'w-80 md:w-[420px]',
//       )}
//     >
//       {/* Header */}
//       <div
//         className="flex items-center justify-between p-4 border-b border-studio-border shrink-0"
//       >
//         <h2
//           className={cn(
//             'text-lg font-semibold text-studio-text transition-opacity duration-200',
//             isCollapsed && 'opacity-0 w-0 overflow-hidden',
//           )}
//         >
//           Upload
//         </h2>
//         <Button
//           variant="ghost"
//           size="sm"
//           onClick={onToggle}
//           className="text-studio-text hover:bg-studio-card-hover"
//         >
//           {isCollapsed ? (
//             <PanelRightOpen size={20} />
//           ) : (
//             <PanelRightClose size={20} />
//           )}
//         </Button>
//       </div>

//       {/* Collapsed State */}
//       {isCollapsed && (
//         <ScrollArea className="flex-1 bg-red-200">
//           <div className="flex flex-col items-center p-4 space-y-4">
//             <Button
//               variant="ghost"
//               size="sm"
//               className="text-studio-text hover:bg-studio-card-hover"
//             >
//               <AudioWaveform size={20} />
//             </Button>
//             <Button
//               variant="ghost"
//               size="sm"
//               className="text-studio-text hover:bg-studio-card-hover"
//             >
//               <Video size={20} />
//             </Button>
//             <Button
//               variant="ghost"
//               size="sm"
//               className="text-studio-text hover:bg-studio-card-hover"
//             >
//               <GitBranch size={20} />
//             </Button>
//             <Button
//               variant="ghost"
//               size="sm"
//               className="text-studio-text hover:bg-studio-card-hover"
//             >
//               <FileText size={20} />
//             </Button>
//           </div>
//         </ScrollArea>
//       )}

//       {/* Expanded State */}
//       {!isCollapsed && (
//         <ScrollArea className="flex-1 bg-amber-200">
//           <div className="flex flex-col gap-12 p-1 mt-1 ml-2 bg-red-400">
//             <FileUploader />

//             <DocListTopFive />

//             <div className="flex justify-center items-center w-full py-3 px-10">
//               <Button
//                 className="rounded-sm w-full cursor-pointer"
//                 variant="secondary"
//                 onClick={handleAllDocs}
//               >
//                 All Documents
//               </Button>
//             </div>
//           </div>
//         </ScrollArea>
//       )}
//     </div>
//   )
// }


"use client"

import { Button } from "@/components/ui/button"
import { AudioWaveform, Video, GitBranch, FileText, PanelRightClose, PanelRightOpen } from "lucide-react"
import { cn } from "@/lib/utils"
import { FileUploader } from "./FileUpload"
import { DocListTopFive } from "./DocListTopFive"
import { useRouter } from "next/navigation"
import { ScrollArea } from "@/components/ui/scroll-area"

interface StudioSidebarProps {
  isCollapsed?: boolean
  onToggle?: () => void
}

export function StudioSidebar({ isCollapsed = false, onToggle }: StudioSidebarProps) {
  const router = useRouter()

  const handleAllDocs = () => {
    router.push("/documents")
  }

  return (
    <div
      className={cn(
        "h-screen overflow-hidden flex flex-col bg-studio-bg border-l border-studio-border transition-all duration-300 z-50",
        isCollapsed ? "w-16" : "w-80 md:w-[450px]",
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-studio-border shrink-0">
        <h2
          className={cn(
            "text-lg font-semibold text-studio-text transition-opacity duration-200",
            isCollapsed && "opacity-0 w-0 overflow-hidden",
          )}
        >
          Document Upload
        </h2>
        <Button variant="ghost" size="sm" onClick={onToggle} className="text-studio-text hover:bg-studio-card-hover">
          {isCollapsed ? <PanelRightOpen size={20} /> : <PanelRightClose size={20} />}
        </Button>
      </div>

      {/* Collapsed State */}
      {isCollapsed && (
        <ScrollArea className="flex-1 overflow-y-auto">
          <div className="flex flex-col items-center p-4 space-y-4">
            <Button variant="ghost" size="sm" className="text-studio-text hover:bg-studio-card-hover">
              <AudioWaveform size={20} />
            </Button>
            <Button variant="ghost" size="sm" className="text-studio-text hover:bg-studio-card-hover">
              <Video size={20} />
            </Button>
            <Button variant="ghost" size="sm" className="text-studio-text hover:bg-studio-card-hover">
              <GitBranch size={20} />
            </Button>
            <Button variant="ghost" size="sm" className="text-studio-text hover:bg-studio-card-hover">
              <FileText size={20} />
            </Button>
          </div>
        </ScrollArea>
      )}

      {/* Expanded State */}
      {!isCollapsed && (
        <ScrollArea className="flex-1 overflow-y-auto">
          <div className="flex flex-col gap-5 p-4">
            <FileUploader />

            <DocListTopFive />

            <div className="flex justify-center items-center w-full px-8">
              <Button className="rounded-sm w-full cursor-pointer" variant="secondary" onClick={handleAllDocs}>
                All Documents
              </Button>
            </div>
          </div>
        </ScrollArea>
      )}
    </div>
  )
}
