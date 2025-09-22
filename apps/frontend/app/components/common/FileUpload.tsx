"use client";

import { useState, type DragEvent, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import { UploadCloud, CheckCircle2, XCircle, LoaderCircle } from "lucide-react";
import { getSignedUrls } from "@/lib/ApiStore/actions/uploadaction"; // ⬅️ new import
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import type { RootState } from "@/store/store";
import { addFile, UserFile } from "@/store/slices/fileSlice";
import { modernToast } from '@/lib/toast'

type UploadStatus = "idle" | "uploading" | "success" | "error";

interface UploadItem {
  file: File;
  progress: number;
  status: UploadStatus;
  error?: string;
}

const ALLOWED_FILE_TYPES = [
  "application/json",
  "text/csv",
  "text/plain",
  "text/tab-separated-values",
  "application/pdf",
];

export function FileUploader() {
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const userId = useSelector((s: RootState) => s.user.userId);
  const dispatch = useDispatch();

  const reset = () => {
    setUploads([]);
    if (inputRef.current) inputRef.current.value = "";
  };

  const validateFiles = (files: File[]) => {
    const rejected: string[] = [];
    const accepted = files.filter((f) => {
      const ok = ALLOWED_FILE_TYPES.includes(f.type);
      if (!ok) rejected.push(f.name);
      return ok;
    });
    if (rejected.length) {
      toast(`Unsupported file types: ${rejected.join(", ")}`);
    }
    return accepted;
  };

  const startUpload = async (files: File[]) => {
    if (!userId) {
      toast("Please sign in first.");
      return;
    }

    const initialState: UploadItem[] = files.map((file) => ({
      file,
      progress: 0,
      status: "uploading",
    }));
    setUploads(initialState);

    try {
      /* 1️⃣  ask the server for signed URLs for every file */
      const res = await getSignedUrls(
        files.map((f) => ({ name: f.name, type: f.type, size: f.size })),
        userId
      );
      if ("failure" in res) throw new Error(res.failure);

      /* 2️⃣  upload every file in parallel */
      await Promise.all(
        res.success.map(({ url, originalFileName, newFileName }, idx) =>
          uploadSingle(files[idx], url, idx).then(() => {
            // after the PUT succeeds dispatch to Redux
            const newReduxFile: UserFile = {
              name: originalFileName,
              size: files[idx].size,
              type: files[idx].type,
              createdAt: new Date().toISOString(),
            };
            dispatch(addFile(newReduxFile));
            modernToast.info(`${originalFileName} uploaded`);
          })
        )
      );
    } catch (err: any) {
      // set every pending upload to error state
      setUploads((prev) =>
        prev.map((u) =>
          u.status === "uploading"
            ? { ...u, status: "error", error: err.message || "Error" }
            : u
        )
      );
    }
  };

  const uploadSingle = (file: File, url: string, index: number) =>
    new Promise<void>((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.open("PUT", url, true);
      xhr.setRequestHeader("Content-Type", file.type);

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const pct = (e.loaded / e.total) * 100;
          setUploads((prev) =>
            prev.map((u, i) => (i === index ? { ...u, progress: pct } : u))
          );
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          setUploads((prev) =>
            prev.map((u, i) =>
              i === index ? { ...u, status: "success", progress: 100 } : u
            )
          );
          resolve();
        } else {
          const err = `Upload failed: ${xhr.status}`;
          setUploads((prev) =>
            prev.map((u, i) =>
              i === index ? { ...u, status: "error", error: err } : u
            )
          );
          reject(new Error(err));
        }
      };

      xhr.onerror = () => {
        const err = "Network error";
        setUploads((prev) =>
          prev.map((u, i) =>
            i === index ? { ...u, status: "error", error: err } : u
          )
        );
        reject(new Error(err));
      };

      xhr.send(file);
    });

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList) return;
    const filesArr = validateFiles(Array.from(fileList));
    if (filesArr.length) startUpload(filesArr);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    handleFiles(e.dataTransfer.files);
  };

  const renderBody = () => {
    if (uploads.length === 0)
      return (
        <div
          className="relative border-2 border-dashed rounded-lg p-2 text-center transition-colors duration-300 cursor-pointer h-48 flex flex-col items-center justify-center hover:border-primary/50"
          onClick={() => inputRef.current?.click()}
        >
          <UploadCloud className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <p className="text-sm font-semibold text-foreground">
            Drag & drop files
          </p>
          <p className="text-muted-foreground text-xs mt-1">
            or click to browse
          </p>
          <p className="text-[10px] text-muted-foreground mt-4">
            Supports: JSON, TXT, TSV, CSV, PDF (up to 5GB)
          </p>
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            multiple // ⬅️ allow many files
            accept={ALLOWED_FILE_TYPES.join(",")}
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      );

    /* show each file row */
    return uploads.map((u, idx) => (
      <div key={idx} className="border rounded-md p-3 mb-2">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs font-medium truncate max-w-[150px]">
            {u.file.name}
          </p>
          {u.status === "success" && (
            <CheckCircle2 className="text-green-500 w-4 h-4" />
          )}
          {u.status === "error" && <XCircle className="text-red-500 w-4 h-4" />}
          {u.status === "uploading" && (
            <LoaderCircle className="animate-spin text-primary w-4 h-4" />
          )}
        </div>
        {u.status === "uploading" && (
          <Progress value={u.progress} className="h-2" />
        )}
        {u.status === "error" && (
          <Alert variant="destructive" className="mt-2">
            <AlertTitle>Upload failed</AlertTitle>
            <AlertDescription className="text-xs">{u.error}</AlertDescription>
          </Alert>
        )}
      </div>
    ));
  };

  return (
    <Card
      className="w-full p-0"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      <CardContent className="p-2">{renderBody()}</CardContent>
    </Card>
  );
}
