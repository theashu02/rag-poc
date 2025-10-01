import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { LoaderCircle, Download, Trash2, MoreHorizontal } from "lucide-react"
import { getFileIcon, getFileTypeBadge, formatBytes, formatDate } from "./file.utils"
import type { FileActionProps } from "./file.types"

export const FileCard = ({ file, onDownload, onDelete, isDeleting }: FileActionProps) => (
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
