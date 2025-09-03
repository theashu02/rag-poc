"use client"

import { useEffect, useState } from "react"
import { getUserFiles } from "@/lib/ApiStore/actions/uploadaction"
import { useSelector } from "react-redux"
import type { RootState } from "@/store/store"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { HardDrive } from "lucide-react"

type UserFile = {
  name: string
  createdAt: string
  id?: string
}

function formatDate(dateString: string) {
  const date = new Date(dateString)
  const now = new Date()
  const diffTime = Math.abs(now.getTime() - date.getTime())
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))

  if (diffDays === 1) return "Today"
  if (diffDays === 2) return "Yesterday"
  if (diffDays <= 7) return `${diffDays - 1} days ago`

  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

function truncateFileName(name: string, maxLength = 24) {
  if (name.length <= maxLength) return name;
  const extIndex = name.lastIndexOf(".");
  if (extIndex > 0 && name.length - extIndex <= 8) {
    const base = name.slice(0, maxLength - (name.length - extIndex) - 3);
    return base + "..." + name.slice(extIndex);
  }
  return name.slice(0, maxLength - 3) + "...";
}

export function DocListTopFive() {
  const userId = useSelector((state: RootState) => state.user.userId)
  const [files, setFiles] = useState<UserFile[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    async function fetchFiles() {
      setIsLoading(true)
      try {
        const result = await getUserFiles(userId || "")
        if (result.success) {
          // Sort by createdAt descending and take 5
          const sorted = result.success
            .sort((a: UserFile, b: UserFile) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
            .slice(0, 5)
          setFiles(sorted)
        } else {
          setFiles([])
        }
      } catch {
        setFiles([])
      }
      setIsLoading(false)
    }
    if (userId) fetchFiles()
  }, [userId])

  return (
    <Card className="w-full shadow-sm border-border/50 rounded-lg">
      <CardHeader className="pb-1">
        <CardTitle className="flex items-center gap-2 text-base">
          <HardDrive className="h-5 w-5" />
          Latest Uploads
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="flex items-center gap-3">
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-3 w-16" />
              </div>
            ))}
          </div>
        ) : files.length === 0 ? (
          <div className="text-muted-foreground text-sm text-center py-4">No files found</div>
        ) : (
          <ul className="space-y-3">
            {files.map((file) => (
              <li key={file.id || file.name} className="flex flex-col">
                {/* <span className="font-medium text-sm truncate">{file.name}</span> */}
                <span className="font-medium text-sm truncate" title={file.name}>
                  {truncateFileName(file.name)}
                </span>
                <span className="text-xs text-muted-foreground">{formatDate(file.createdAt)}</span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}