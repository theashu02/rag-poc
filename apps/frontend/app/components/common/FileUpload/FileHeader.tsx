import { CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { HardDrive, RefreshCw, Search, Calendar, List, Grid3X3 } from "lucide-react"
import { cn } from "@/lib/utils"
import type { ViewMode } from "./file.types"

interface FileHeaderProps {
  fileCount: number
  filteredCount: number
  viewMode: ViewMode
  setViewMode: (mode: ViewMode) => void
  searchQuery: string
  setSearchQuery: (query: string) => void
  isRefreshing: boolean
  onRefresh: () => void
}

export const FileHeader = ({
  fileCount,
  filteredCount,
  viewMode,
  setViewMode,
  searchQuery,
  setSearchQuery,
  isRefreshing,
  onRefresh,
}: FileHeaderProps) => (
  <>
    <div className="flex items-center justify-between">
      <CardTitle className="flex items-center gap-2">
        <HardDrive className="h-5 w-5" />
        Your Files
        {fileCount > 0 && (
          <Badge variant="secondary" className="ml-2 bg-muted text-muted-foreground">
            {fileCount}
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
          onClick={onRefresh}
          disabled={isRefreshing}
          className="flex items-center gap-2"
        >
          <RefreshCw className={cn("h-4 w-4", isRefreshing && "animate-spin")} />
          <span className="hidden sm:inline">Refresh</span>
        </Button>
      </div>
    </div>

    {fileCount > 0 && (
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
            {filteredCount} of {fileCount} files
          </span>
        </div>
      </div>
    )}
  </>
)
