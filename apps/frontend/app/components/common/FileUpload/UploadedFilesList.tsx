"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { AlertCircle, RefreshCw } from "lucide-react"
import { LoadingSkeleton } from "./LoadingSkeleton"
import { EmptyState } from "./EmptyState"
import { FileHeader } from "./FileHeader"
import { FileTableView } from "./FileTableView"
import { FileGridView } from "./FileGridView"
import { FileCard } from "./FileCard"
import { useFileManager } from "@/hooks/useFileManager"
import type { ViewMode } from "./file.types"

export function UploadedFilesList() {
  const [viewMode, setViewMode] = useState<ViewMode>("table")
  const {
    files,
    filteredAndSortedFiles,
    isLoading,
    error,
    searchQuery,
    setSearchQuery,
    isRefreshing,
    deletingFile,
    fetchFiles,
    handleSort,
    handleDownload,
    handleDelete,
  } = useFileManager()

  if (isLoading && !isRefreshing) {
    return <LoadingSkeleton />
  }

  if (error) {
    return (
      <Card className="w-full shadow-sm rounded-none h-screen flex flex-col">
        <CardHeader className="pb-4 flex-shrink-0">
          <FileHeader
            fileCount={files.length}
            filteredCount={filteredAndSortedFiles.length}
            viewMode={viewMode}
            setViewMode={setViewMode}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            isRefreshing={isRefreshing}
            onRefresh={() => fetchFiles(true)}
          />
        </CardHeader>
        <CardContent className="flex-1 overflow-auto">
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
    <Card className="w-full shadow-sm border-border/50 rounded-none h-screen flex flex-col">
      <CardHeader className="pb-6 flex-shrink-0">
        <FileHeader
          fileCount={files.length}
          filteredCount={filteredAndSortedFiles.length}
          viewMode={viewMode}
          setViewMode={setViewMode}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          isRefreshing={isRefreshing}
          onRefresh={() => fetchFiles(true)}
        />
      </CardHeader>

      <CardContent className="pt-0 flex-1 overflow-auto">
        {files.length === 0 ? (
          <EmptyState type="no-files" />
        ) : filteredAndSortedFiles.length === 0 ? (
          <EmptyState type="no-results" />
        ) : (
          <>
            {/* Mobile View */}
            <div className="block sm:hidden">
              <div className="grid gap-4 pb-4">
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

            {/* Desktop View */}
            <div className="hidden sm:block">
              {viewMode === "grid" ? (
                <div className="pb-4">
                  <FileGridView
                    files={filteredAndSortedFiles}
                    onDownload={handleDownload}
                    onDelete={handleDelete}
                    deletingFile={deletingFile}
                  />
                </div>
              ) : (
                <div className="pb-4">
                  <FileTableView
                    files={filteredAndSortedFiles}
                    onSort={handleSort}
                    onDownload={handleDownload}
                    onDelete={handleDelete}
                    deletingFile={deletingFile}
                  />
                </div>
              )}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}