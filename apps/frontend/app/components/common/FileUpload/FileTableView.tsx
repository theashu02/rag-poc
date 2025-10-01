import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { LoaderCircle, Download, Trash2, MoreHorizontal, ArrowUpDown } from "lucide-react"
import { getFileIcon, getFileTypeBadge, formatBytes, formatDate } from "./file.utils"
import type { UserFile, SortField } from "./file.types"

interface FileTableViewProps {
  files: UserFile[]
  onSort: (field: SortField) => void
  onDownload: (file: UserFile) => void
  onDelete: (file: UserFile) => void
  deletingFile: string | null
}

export const FileTableView = ({ files, onSort, onDownload, onDelete, deletingFile }: FileTableViewProps) => (
  <div className="rounded-lg border border-border/50 overflow-hidden">
    <Table>
      <TableHeader className="bg-muted/30">
        <TableRow className="hover:bg-transparent border-border/50">
          <TableHead className="w-[45%] font-semibold">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onSort("name")}
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
              onClick={() => onSort("size")}
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
              onClick={() => onSort("createdAt")}
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
        {files.map((file, index) => (
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
                  <DropdownMenuItem onClick={() => onDownload(file)}>
                    <Download className="mr-2 h-4 w-4" />
                    Download
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onClick={() => onDelete(file)}
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
)
