"use client"

import { useEffect, useState, useMemo } from "react"
import { useSelector } from "react-redux"
import type { RootState } from "@/store/store"
import { deleteUserFile, getUserFiles } from "@/lib/ApiStore/actions/uploadaction"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { toast } from "sonner"
import { LoaderCircle, AlertCircle, Search, RefreshCw, Download, Trash2, File, FileText, FileImage, FileVideo, FileAudio, Archive, MoreHorizontal, ArrowUpDown, Calendar, HardDrive, Grid3X3, List } from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"

type UserFile = {
  name: string
  size: number | string
  createdAt: string
  type?: string
  id?: string
  url?: string
}

type SortField = "name" | "size" | "createdAt"
type SortDirection = "asc" | "desc"
type ViewMode = "table" | "grid"

const getFileIcon = (fileName: string, fileType?: string) => {
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

const getFileTypeBadge = (fileName: string, fileType?: string) => {
  const extension = fileName.split(".").pop()?.toLowerCase()
  if (!extension) return null

  const colors: Record<string, string> = {
    pdf: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-800",
    doc: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
    docx: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
    jpg: "bg-green-50 text-green-700 border-green-200 dark:bg-green-950 dark:text-green-300 dark:border-green-800",
    jpeg: "bg-green-50 text-green-700 border-green-200 dark:bg-green-950 dark:text-green-300 dark:border-green-800",
    png: "bg-green-50 text-green-700 border-green-200 dark:bg-green-950 dark:text-green-300 dark:border-green-800",
    mp4: "bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-950 dark:text-purple-300 dark:border-purple-800",
    zip: "bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950 dark:text-orange-300 dark:border-orange-800",
  }

  return (
    <Badge
      variant="outline"
      className={cn(
        "text-xs font-medium border",
        colors[extension] ||
          "bg-gray-50 text-gray-700 border-gray-200 dark:bg-gray-950 dark:text-gray-300 dark:border-gray-800",
      )}
    >
      {extension.toUpperCase()}
    </Badge>
  )
}

const formatBytes = (bytes: number | string, decimals = 1) => {
  const bytesNum = typeof bytes === "string" ? Number.parseInt(bytes, 10) : bytes
  if (bytesNum === 0) return "0 B"
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ["B", "KB", "MB", "GB", "TB"]
  const i = Math.floor(Math.log(bytesNum) / Math.log(k))
  return Number.parseFloat((bytesNum / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i]
}

const formatDate = (dateString: string) => {
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

const FileCard = ({ file, onDownload, onDelete, isDeleting }: {
  file: UserFile
onDownload: (file: UserFile) => void
  onDelete: (file: UserFile) => void
  isDeleting: boolean
}) => (
  <Card className="group hover:shadow-md transition-all duration-200 border-border/50 hover:border-border">
    <CardContent className="p-4">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3 flex-1 min-w-0">
          {getFileIcon(file.name, file.type)}
          <div className="flex flex-col gap-1 flex-1 min-w-0">
            <h4 className="font-medium text-sm truncate" title={file.name}>
              {file.name}
            </h4>
            {getFileTypeBadge(file.name, file.type)}
          </div>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
              disabled={isDeleting}
            >
              {isDeleting ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <MoreHorizontal className="h-4 w-4" />}
              <span className="sr-only">Open menu</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => onDownload(file)}>
              <Download className="mr-2 h-4 w-4" />
              Download
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onDelete(file)} className="text-destructive focus:text-destructive">
              <Trash2 className="mr-2 h-4 w-4" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span className="font-mono">{formatBytes(file.size)}</span>
        <span>{formatDate(file.createdAt)}</span>
      </div>
    </CardContent>
  </Card>
)

const LoadingSkeleton = () => (
  <Card className="w-full">
    <CardHeader className="pb-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Skeleton className="h-5 w-5" />
          <Skeleton className="h-6 w-24" />
          <Skeleton className="h-5 w-8" />
        </div>
        <Skeleton className="h-9 w-20" />
      </div>
      <div className="flex items-center gap-4 mt-4">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-4 w-32" />
      </div>
    </CardHeader>
    <CardContent>
      <div className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="flex items-center gap-4 p-3 border rounded-lg">
            <Skeleton className="h-5 w-5" />
            <div className="flex-1 space-y-2">
              <Skeleton className="h-4 w-48" />
              <Skeleton className="h-3 w-16" />
            </div>
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-8 w-8" />
          </div>
        ))}
      </div>
    </CardContent>
  </Card>
)

export function UploadedFilesList() {
  const userId = useSelector((state: RootState) => state.user.userId)
  const [files, setFiles] = useState<UserFile[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  const [sortField, setSortField] = useState<SortField>("createdAt")
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc")
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [deletingFile, setDeletingFile] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>("table")

  const filteredAndSortedFiles = useMemo(() => {
    const filtered = files.filter((file) => file.name.toLowerCase().includes(searchQuery.toLowerCase()))

    filtered.sort((a, b) => {
      let aValue: any = a[sortField]
      let bValue: any = b[sortField]

      if (sortField === "size") {
        aValue = typeof aValue === "string" ? Number.parseInt(aValue, 10) : aValue
        bValue = typeof bValue === "string" ? Number.parseInt(bValue, 10) : bValue
      } else if (sortField === "createdAt") {
        aValue = new Date(aValue).getTime()
        bValue = new Date(bValue).getTime()
      }

      if (sortDirection === "asc") {
        return aValue > bValue ? 1 : -1
      } else {
        return aValue < bValue ? 1 : -1
      }
    })

    return filtered
  }, [files, searchQuery, sortField, sortDirection])

  const fetchFiles = async (showRefreshLoader = false) => {
    if (showRefreshLoader) {
      setIsRefreshing(true)
    } else {
      setIsLoading(true)
    }
    setError(null)

    try {
      const result = await getUserFiles(userId || "")

      if (result.success) {
        setFiles(result.success)
      } else {
        setError(result.failure || "Unknown error occurred")
      }
    } catch (err) {
      setError("Failed to load files. Please try again.")
    } finally {
      setIsLoading(false)
      setIsRefreshing(false)
    }
  }

  useEffect(() => {
    if (!userId) {
      setIsLoading(false)
      return
    }
    fetchFiles()
  }, [userId])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("desc")
    }
  }

  const handleDownload = (file: UserFile) => {
    // TODO: Implement download functionality
    console.log("Download file:", file.name)
  }

  const handleDelete = async (file: UserFile) => {
    if (!userId) {
      toast.error("You must be logged in to delete files.")
      return
    }
    setDeletingFile(file.name)
    const result = await deleteUserFile(userId, file.name)
    if (result.success) {
      setFiles((currentFiles) => currentFiles.filter((f) => f.name !== file.name))
      toast.success(`"${file.name}" has been deleted.`)
    } else {
      toast.error(`Failed to delete file: ${result.failure}`)
    }
    setDeletingFile(null)
  }

  if (isLoading) {
    return <LoadingSkeleton />
  }

  if (error) {
    return (
      <Card className="w-full shadow-sm rounded-none">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2">
            <HardDrive className="h-5 w-5" />
            Your Files
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive" className="border-destructive/20">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="flex items-center justify-between">
              <span>{error}</span>
              <Button variant="outline" size="sm" onClick={() => fetchFiles()} className="ml-4">
                <RefreshCw className="h-4 w-4 mr-2" />
                Retry
              </Button>
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full shadow-sm border-border/50 rounded-none">
      <CardHeader className="pb-6">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <HardDrive className="h-5 w-5" />
            Your Files
            {files.length > 0 && (
              <Badge variant="secondary" className="ml-2 bg-muted text-muted-foreground">
                {files.length}
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            <div className="hidden sm:flex items-center border rounded-lg p-1">
              <Button
                variant={viewMode === "table" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setViewMode("table")}
                className="h-7 px-2"
              >
                <List className="h-4 w-4" />
              </Button>
              <Button
                variant={viewMode === "grid" ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setViewMode("grid")}
                className="h-7 px-2"
              >
                <Grid3X3 className="h-4 w-4" />
              </Button>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => fetchFiles(true)}
              disabled={isRefreshing}
              className="flex items-center gap-2"
            >
              <RefreshCw className={cn("h-4 w-4", isRefreshing && "animate-spin")} />
              <span className="hidden sm:inline">Refresh</span>
            </Button>
          </div>
        </div>

        {files.length > 0 && (
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 mt-6">
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search files..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-background"
              />
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Calendar className="h-4 w-4" />
              <span>
                {filteredAndSortedFiles.length} of {files.length} files
              </span>
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="pt-0">
        {files.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="rounded-full bg-muted p-4 mb-4">
              <HardDrive className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-semibold mb-2">No files uploaded yet</h3>
            <p className="text-muted-foreground max-w-sm">
              Upload your first file to get started managing your documents
            </p>
          </div>
        ) : filteredAndSortedFiles.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="rounded-full bg-muted p-4 mb-4">
              <Search className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-semibold mb-2">No files found</h3>
            <p className="text-muted-foreground">Try adjusting your search query</p>
          </div>
        ) : (
          <>
            <div className="block sm:hidden">
              <div className="grid gap-4">
                {filteredAndSortedFiles.map((file, index) => (
                  <FileCard
                    key={file.id || `${file.name}-${index}`}
                    file={file}
                    onDownload={handleDownload}
                    onDelete={handleDelete}
                    isDeleting={deletingFile === file.name}
                  />
                ))}
              </div>
            </div>

            <div className="hidden sm:block">
              {viewMode === "grid" ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {filteredAndSortedFiles.map((file, index) => (
                    <FileCard
                      key={file.id || `${file.name}-${index}`}
                      file={file}
                      onDownload={handleDownload}
                      onDelete={handleDelete}
                      isDeleting={deletingFile === file.name}
                    />
                  ))}
                </div>
              ) : (
                <div className="rounded-lg border border-border/50 overflow-hidden">
                  <Table>
                    <TableHeader className="bg-muted/30">
                      <TableRow className="hover:bg-transparent border-border/50">
                        <TableHead className="w-[45%] font-semibold">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleSort("name")}
                            className="h-auto p-0 font-semibold hover:bg-transparent hover:text-foreground"
                          >
                            File Name
                            <ArrowUpDown className="ml-2 h-4 w-4" />
                          </Button>
                        </TableHead>
                        <TableHead className="w-[20%] font-semibold">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleSort("size")}
                            className="h-auto p-0 font-semibold hover:bg-transparent hover:text-foreground"
                          >
                            Size
                            <ArrowUpDown className="ml-2 h-4 w-4" />
                          </Button>
                        </TableHead>
                        <TableHead className="w-[25%] font-semibold">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleSort("createdAt")}
                            className="h-auto p-0 font-semibold hover:bg-transparent hover:text-foreground"
                          >
                            Upload Date
                            <ArrowUpDown className="ml-2 h-4 w-4" />
                          </Button>
                        </TableHead>
                        <TableHead className="w-[10%]"></TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredAndSortedFiles.map((file, index) => (
                        <TableRow
                          key={file.id || `${file.name}-${index}`}
                          className="hover:bg-muted/50 transition-colors border-border/50"
                        >
                          <TableCell className="py-4">
                            <div className="flex items-center gap-3">
                              {getFileIcon(file.name, file.type)}
                              <div className="flex flex-col gap-1.5">
                                <span className="font-medium text-sm truncate max-w-[300px]" title={file.name}>
                                  {file.name}
                                </span>
                                {getFileTypeBadge(file.name, file.type)}
                              </div>
                            </div>
                          </TableCell>
                          <TableCell className="font-mono text-sm py-4">{formatBytes(file.size)}</TableCell>
                          <TableCell className="text-sm text-muted-foreground py-4">
                            {formatDate(file.createdAt)}
                          </TableCell>
                          <TableCell className="py-4">
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0"
                                  disabled={deletingFile === file.name}
                                >
                                  {deletingFile === file.name ? (
                                    <LoaderCircle className="h-4 w-4 animate-spin" />
                                  ) : (
                                    <MoreHorizontal className="h-4 w-4" />
                                  )}
                                  <span className="sr-only">Open menu</span>
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end" className="w-40">
                                <DropdownMenuItem onClick={() => handleDownload(file)}>
                                  <Download className="mr-2 h-4 w-4" />
                                  Download
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onClick={() => handleDelete(file)}
                                  className="text-destructive focus:text-destructive"
                                >
                                  <Trash2 className="mr-2 h-4 w-4" />
                                  Delete
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}

