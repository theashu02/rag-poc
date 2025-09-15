import React from "react";
import { UploadedFilesList } from "../components/common/UploadFileList";
import { Sidebar } from "../components/common/Sidebar";

export default function page() {
  return (
    <div className="flex h-screen w-screen bg-background rounded-none">
      <Sidebar />
      <UploadedFilesList />
    </div>
  );
}
