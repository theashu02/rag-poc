import { HardDrive, Search } from "lucide-react"

interface EmptyStateProps {
  type: "no-files" | "no-results"
}

export const EmptyState = ({ type }: EmptyStateProps) => {
  if (type === "no-files") {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <div className="rounded-full bg-muted p-4 mb-4">
          <HardDrive className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-2">No files uploaded yet</h3>
        <p className="text-muted-foreground max-w-sm">
          Upload your first file to get started managing your documents
        </p>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="rounded-full bg-muted p-4 mb-4">
        <Search className="h-8 w-8 text-muted-foreground" />
      </div>
      <h3 className="text-lg font-semibold mb-2">No files found</h3>
      <p className="text-muted-foreground">Try adjusting your search query</p>
    </div>
  )
}
