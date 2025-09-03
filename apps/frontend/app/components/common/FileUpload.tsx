'use client';

import { useState, type DragEvent, useRef } from 'react';
import { useDispatch } from 'react-redux';
import { UploadCloud, CheckCircle2, XCircle, LoaderCircle } from 'lucide-react';
import { getSignedUrl } from '@/lib/ApiStore/actions/uploadaction';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { cn } from '@/lib/utils';
import { toast } from "sonner";
import { useSelector } from 'react-redux';
import type { RootState } from '@/store/store';
import { addFile, UserFile } from '@/store/fileSlice';

type Status = 'idle' | 'dragging' | 'uploading' | 'success' | 'error';
type UploadedFileDetails = { originalFileName: string; newFileName: string };

const ALLOWED_FILE_TYPES = [
  'application/json',
  'text/csv',
  'text/plain',
  'text/tab-separated-values',
  'application/pdf',
];

export function FileUploader() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>('idle');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<UploadedFileDetails | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const xhrRef = useRef<XMLHttpRequest | null>(null);
  const userId = useSelector((state: RootState) => state.user.userId);
  const dispatch = useDispatch();

  const handleFileSelect = (selectedFile: File | null) => {
    if (status === 'uploading') return;
    if (!selectedFile) return;

    if (!ALLOWED_FILE_TYPES.includes(selectedFile.type)) {
      setError(`File type "${selectedFile.type}" is not supported.`);
      setStatus('error');
      return;
    }
    
    setFile(selectedFile);
    handleUpload(selectedFile);
  };
  
  const handleUpload = async (fileToUpload: File) => {
    if(!userId) {
      toast('Please sign in first.');
      return;
    }
    setStatus('uploading');
    setProgress(0);
    setError(null);
    setUploadedFile(null);

    try {
      const signedUrlResult = await getSignedUrl({
        name: fileToUpload.name,
        type: fileToUpload.type,
        size: fileToUpload.size,
        userId,
      });

      if('failure' in signedUrlResult){
        throw new Error(signedUrlResult.failure);
      } else {
        const { url, originalFileName, newFileName } = signedUrlResult.success;
        await uploadFile(fileToUpload, url);
        setStatus('success');
        setUploadedFile({ originalFileName, newFileName });

        const newFile: UserFile = {
          name: fileToUpload.name,
          size: fileToUpload.size,
          type: fileToUpload.type,
          createdAt: new Date().toISOString(),
        };
        dispatch(addFile(newFile));

        toast(`${originalFileName} has been successfully uploaded.`);
      }
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
      setStatus('error');
    }
  };

  const uploadFile = (fileToUpload: File, url: string) => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhrRef.current = xhr;

      xhr.open('PUT', url, true);
      xhr.setRequestHeader('Content-Type', fileToUpload.type);

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percentComplete = (event.loaded / event.total) * 100;
          setProgress(percentComplete);
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(xhr.response);
        } else {
          reject(new Error(`Upload failed with status: ${xhr.status}. ${xhr.responseText}`));
        }
      };

      xhr.onerror = () => {
        reject(new Error('Upload failed due to a network error.'));
      };
      
      xhr.onabort = () => {
        reject(new Error('Upload was cancelled.'));
      }

      xhr.send(fileToUpload);
    });
  };

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (status !== 'uploading') setStatus('dragging');
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (status !== 'uploading') setStatus('idle');
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setStatus('idle');
    const droppedFile = e.dataTransfer.files[0];
    handleFileSelect(droppedFile);
  };
  
  const handleReset = () => {
    setFile(null);
    setStatus('idle');
    setProgress(0);
    setError(null);
    setUploadedFile(null);
    if(inputRef.current) inputRef.current.value = "";
  };
  
  const handleCancel = () => {
    if(xhrRef.current){
      xhrRef.current.abort();
    }
    handleReset();
  }

  const renderContent = () => {
    switch (status) {
      case 'uploading':
        return (
          <div className="w-full text-center p-3">
            <LoaderCircle className="mx-auto h-12 w-12 text-primary animate-spin mb-3" />
            <p className="text-lg font-medium text-primary mb-2">Uploading...</p>
            {file && <p className="text-muted-foreground mb-2 text-xs">{file.name}</p>}
            <Progress value={progress} className="w-full h-3" />
            <p className="text-sm font-medium text-primary mt-2">{Math.round(progress)}%</p>
            <Button variant="outline" size="sm" className="mt-4" onClick={handleCancel}>Cancel</Button>
          </div>
        );
      case 'success':
        return (
          <div className="w-full text-center p-3">
            <CheckCircle2 className="mx-auto h-16 w-16 text-green-500 mb-4 animate-in fade-in zoom-in-50 duration-500" />
            <h3 className="text-lg font-bold text-primary mb-2">Upload Successful!</h3>
            {uploadedFile && (
              <div className="text-muted-foreground text-xs space-y-1">
                <p>File name: <span className="font-medium text-foreground">{uploadedFile.originalFileName}</span></p>
                {/* <p>Stored as: <span className="font-medium text-foreground">{uploadedFile.newFileName}</span></p> */}
              </div>
            )}
            <Button className="mt-8" onClick={handleReset}>Upload Another File</Button>
          </div>
        );
      case 'error':
        return (
          <div className="w-full text-center p-8">
            <Alert variant="destructive" className="mb-6 text-left">
              <XCircle className="h-4 w-4" />
              <AlertTitle>Upload Failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
            <Button variant="destructive" onClick={handleReset}>Try Again</Button>
          </div>
        );
      case 'idle':
      case 'dragging':
      default:
        return (
          <div
            className={cn(
              "relative border-2 border-dashed rounded-lg p-2 text-center transition-colors duration-300 cursor-pointer h-48 flex flex-col items-center justify-center",
              status === 'dragging' ? 'border-primary bg-accent/20' : 'border-border hover:border-primary/50'
            )}
            onClick={() => inputRef.current?.click()}
          >
            <UploadCloud className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-sm font-semibold text-foreground">Drag & drop file</p>
            <p className="text-muted-foreground text-xs mt-1">or click to browse</p>
            <p className="text-[10px] text-muted-foreground mt-4">Supports: JSON, TXT, TSV, CSV, PDF (up to 5GB)</p>
            <input
              ref={inputRef}
              type="file"
              className="hidden"
              onChange={(e) => handleFileSelect(e.target.files ? e.target.files[0] : null)}
              accept={ALLOWED_FILE_TYPES.join(',')}
            />
          </div>
        );
    }
  };

  return (
    <Card 
      className="w-full p-0"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <CardContent className="p-2">
        {renderContent()}
      </CardContent>
    </Card>
  );
}
