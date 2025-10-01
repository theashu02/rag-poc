import { FileImage, FileVideo, FileAudio, Archive, FileText, File } from "lucide-react"
import { Badge } from "@/components/ui/badge"   
import { cn } from "@/lib/utils"

const FILE_TYPE_COLORS: Record<string, string> = {
  pdf: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-800",
  doc: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
  docx: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
  jpg: "bg-green-50 text-green-700 border-green-200 dark:bg-green-950 dark:text-green-300 dark:border-green-800",
  jpeg: "bg-green-50 text-green-700 border-green-200 dark:bg-green-950 dark:text-green-300 dark:border-green-800",
  png: "bg-green-50 text-green-700 border-green-200 dark:bg-green-950 dark:text-green-300 dark:border-green-800",
  mp4: "bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-950 dark:text-purple-300 dark:border-purple-800",
  zip: "bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950 dark:text-orange-300 dark:border-orange-800",
}

export const getFileIcon = (fileName: string, fileType?: string) => {
  const extension = fileName.split(".").pop()?.toLowerCase()
  const type = fileType?.toLowerCase()

  if (type?.startsWith("image/") || ["jpg", "jpeg", "png", "gif", "webp", "svg"].includes(extension || "")) {
    return <FileImage className="h-5 w-5 text-blue-500" />
  }
  if (type?.startsWith("video/") || ["mp4", "avi", "mov", "wmv", "flv"].includes(extension || "")) {
    return <FileVideo className="h-5 w-5 text-purple-500" />
  }
  if (type?.startsWith("audio/") || ["mp3", "wav", "flac", "aac"].includes(extension || "")) {
    return <FileAudio className="h-5 w-5 text-green-500" />
  }
  if (["zip", "rar", "7z", "tar", "gz"].includes(extension || "")) {
    return <Archive className="h-5 w-5 text-orange-500" />
  }
  if (["txt", "doc", "docx", "pdf", "rtf"].includes(extension || "")) {
    return <FileText className="h-5 w-5 text-red-500" />
  }
  return <File className="h-5 w-5 text-muted-foreground" />
}

export const getFileTypeBadge = (fileName: string, fileType?: string) => {
  const extension = fileName.split(".").pop()?.toLowerCase()
  if (!extension) return null

  return (
    <Badge
      variant="outline"
      className={cn(
        "text-xs font-medium border",
        FILE_TYPE_COLORS[extension] ||
          "bg-gray-50 text-gray-700 border-gray-200 dark:bg-gray-950 dark:text-gray-300 dark:border-gray-800",
      )}
    >
      {extension.toUpperCase()}
    </Badge>
  )
}

export const formatBytes = (bytes: number | string, decimals = 1) => {
  const bytesNum = typeof bytes === "string" ? Number.parseInt(bytes, 10) : bytes
  if (bytesNum === 0) return "0 B"
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ["B", "KB", "MB", "GB", "TB"]
  const i = Math.floor(Math.log(bytesNum) / Math.log(k))
  return Number.parseFloat((bytesNum / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i]
}

export const formatDate = (dateString: string) => {
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
