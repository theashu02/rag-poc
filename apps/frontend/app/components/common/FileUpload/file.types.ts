export type UserFile = {
  name: string;
  size: number | string;
  createdAt: string;
  type?: string;
  id?: string;
  url?: string;
};

export type SortField = "name" | "size" | "createdAt";
export type SortDirection = "asc" | "desc";
export type ViewMode = "table" | "grid";

export interface FileActionProps {
  file: UserFile;
  onDownload: (file: UserFile) => void;
  onDelete: (file: UserFile) => void;
  isDeleting: boolean;
}
