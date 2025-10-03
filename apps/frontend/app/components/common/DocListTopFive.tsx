"use client"

import React, { useMemo, memo, useEffect } from "react"
import { useSelector, useDispatch, shallowEqual } from "react-redux"
import type { RootState } from "@/store/store"
import { fetchUserFiles } from "@/store/slices/fileSlice"
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
  if (name.length <= maxLength) return name
  const extIndex = name.lastIndexOf(".")
  if (extIndex > 0 && name.length - extIndex <= 8) {
    const base = name.slice(0, maxLength - (name.length - extIndex) - 3)
    return base + "..." + name.slice(extIndex)
  }
  return name.slice(0, maxLength - 3) + "..."
}

const ListItem = memo(function ListItem({ file }: { file: UserFile }) {
  return (
    <li className="flex flex-col border-2 py-2 rounded-sm pl-3">
      <span className="font-medium text-sm truncate" title={file.name}>
        {truncateFileName(file.name)}
      </span>
      <span className="text-xs text-muted-foreground">
        {formatDate(file.createdAt)}
      </span>
    </li>
  )
})

export const DocListTopFive = memo(function DocListTopFive() {
  const dispatch = useDispatch()
  const userId = useSelector((state: RootState) => state.user.userId)
  const { files, isLoading } = useSelector(
    (state: RootState) => state.files,
    shallowEqual
  )

  // useEffect(() => {
  //   if (userId) {
  //     dispatch(fetchUserFiles(userId) as any) // this render the list on evey mount
  //   }
  // }, [dispatch, userId])

  useEffect(() => {
    // fetch only once per session
    if (userId && files.length === 0 && !isLoading) {
      dispatch(fetchUserFiles(userId) as any)
    }
  }, [dispatch, userId, files.length, isLoading])

  const latestFiles = useMemo(() => {
    return [...files]
      .sort(
        (a, b) =>
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      )
      .slice(0, 5)
  }, [files])

  return (
    <Card className="w-full shadow-sm border-border/50 rounded-lg">
      <CardHeader className="px-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <HardDrive className="h-5 w-5" />
          Latest Uploads
        </CardTitle>
      </CardHeader>
      <CardContent className="px-3">
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
          <div className="text-muted-foreground text-sm text-center py-4">
            No files found
          </div>
        ) : (
          <ul className="space-y-2">
            {latestFiles.map((file) => (
              <ListItem key={file.id || file.name} file={file} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
})