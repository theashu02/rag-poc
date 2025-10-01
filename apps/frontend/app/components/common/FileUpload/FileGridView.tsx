import { FileCard } from "./FileCard"
import type { UserFile } from "./file.types"

interface FileGridViewProps {
  files: UserFile[]
  onDownload: (file: UserFile) => void
  onDelete: (file: UserFile) => void
  deletingFile: string | null
}

export const FileGridView = ({ files, onDownload, onDelete, deletingFile }: FileGridViewProps) => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
    {files.map((file, index) => (
      <FileCard
        key={file.id || `${file.name}-${index}`}
        file={file}
        onDownload={onDownload}
        onDelete={onDelete}
        isDeleting={deletingFile === file.name}
      />
    ))}
  </div>
)
