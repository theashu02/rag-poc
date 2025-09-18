import { useState, useEffect, useMemo } from "react"
import { useSelector, useDispatch } from "react-redux"
import { toast } from "sonner"
import type { RootState, AppDispatch } from "@/store/store"
import { fetchUserFiles, removeUserFile } from "@/store/slices/fileSlice"
import { downloadUserFile } from "@/lib/ApiStore/actions/uploadaction"
import type { UserFile, SortField, SortDirection } from "../app/components/common/FileUpload/file.types"

export const useFileManager = () => {
  const userId = useSelector((state: RootState) => state.user.userId)
  const dispatch = useDispatch<AppDispatch>()
  const { files, isLoading, error } = useSelector((state: RootState) => state.files)

  const [searchQuery, setSearchQuery] = useState("")
  const [sortField, setSortField] = useState<SortField>("createdAt")
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc")
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [deletingFile, setDeletingFile] = useState<string | null>(null)
  const [downloadFile, setDownloadFile] = useState<string | null>(null)

  const filteredAndSortedFiles = useMemo(() => {
    const filtered = files.filter((file) => 
      file.name.toLowerCase().includes(searchQuery.toLowerCase())
    )

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
    if (!userId) return
    if (showRefreshLoader) {
      setIsRefreshing(true)
    }
    await dispatch(fetchUserFiles(userId))
    if (showRefreshLoader) {
      setIsRefreshing(false)
    }
  }

  useEffect(() => {
    if (userId) {
      dispatch(fetchUserFiles(userId))
    }
  }, [userId, dispatch])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("desc")
    }
  }

  const handleDownload = async (file: UserFile) => {
    if (!userId) {
      toast.error("You must be logged in to download the files.")
      return
    }
    setDownloadFile(file.name)

    const result: any = await downloadUserFile(userId, file.name)

    try {
      if (result.success?.url) {
        const response = await fetch(result.success.url)
        if (!response.ok) throw new Error("Network error")
        const blob = await response.blob()
        const blobUrl = URL.createObjectURL(blob)

        const link = document.createElement("a")
        link.href = blobUrl
        link.download = file.name
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        URL.revokeObjectURL(blobUrl)

        toast.success(`"${file.name}" is ready to download.`)
      } else if (result.failure) {
        toast.error(`Failed to download file: ${result.failure}`)
      } else {
        toast.error("Failed to download file: Unknown error")
      }
    } catch (err: any) {
      toast.error(`Failed to download file: ${err.message || err}`)
    } finally {
      setDownloadFile(null)
    }
  }

  const handleDelete = async (file: UserFile) => {
    if (!userId) {
      toast.error("You must be logged in to delete files.")
      return
    }
    setDeletingFile(file.name)
    const result: any = await dispatch(removeUserFile({ userId, fileName: file.name }))
    if (result.payload) {
      toast.success(`"${file.name}" has been deleted.`)
    } else {
      toast.error(`Failed to delete file: ${result.error.message}`)
    }
    setDeletingFile(null)
  }

  return {
    files,
    filteredAndSortedFiles,
    isLoading,
    error,
    searchQuery,
    setSearchQuery,
    sortField,
    sortDirection,
    isRefreshing,
    deletingFile,
    downloadFile,
    fetchFiles,
    handleSort,
    handleDownload,
    handleDelete,
  }
}
