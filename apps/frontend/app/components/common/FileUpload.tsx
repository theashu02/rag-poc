// 'use client';

// import { useState, type DragEvent, useRef } from 'react';
// import { useDispatch } from 'react-redux';
// import { UploadCloud, CheckCircle2, XCircle, LoaderCircle } from 'lucide-react';
// import { getSignedUrl } from '@/lib/ApiStore/actions/uploadaction';
// import { Card, CardContent } from '@/components/ui/card';
// import { Button } from '@/components/ui/button';
// import { Progress } from '@/components/ui/progress';
// import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
// import { cn } from '@/lib/utils';
// import { toast } from "sonner";
// import { useSelector } from 'react-redux';
// import type { RootState } from '@/store/store';
// import { addFile, UserFile } from '@/store/slices/fileSlice';

// type Status = 'idle' | 'dragging' | 'uploading' | 'success' | 'error';
// type UploadedFileDetails = { originalFileName: string; newFileName: string };

// const ALLOWED_FILE_TYPES = [
//   'application/json',
//   'text/csv',
//   'text/plain',
//   'text/tab-separated-values',
//   'application/pdf',
// ];

// export function FileUploader() {
//   const [file, setFile] = useState<File | null>(null);
//   const [status, setStatus] = useState<Status>('idle');
//   const [progress, setProgress] = useState(0);
//   const [error, setError] = useState<string | null>(null);
//   const [uploadedFile, setUploadedFile] = useState<UploadedFileDetails | null>(null);
//   const inputRef = useRef<HTMLInputElement>(null);
//   const xhrRef = useRef<XMLHttpRequest | null>(null);
//   const userId = useSelector((state: RootState) => state.user.userId);
//   const dispatch = useDispatch();

//   const handleFileSelect = (selectedFile: File | null) => {
//     if (status === 'uploading') return;
//     if (!selectedFile) return;

//     if (!ALLOWED_FILE_TYPES.includes(selectedFile.type)) {
//       setError(`File type "${selectedFile.type}" is not supported.`);
//       setStatus('error');
//       return;
//     }
    
//     setFile(selectedFile);
//     handleUpload(selectedFile);
//   };
  
//   const handleUpload = async (fileToUpload: File) => {
//     if(!userId) {
//       toast('Please sign in first.');
//       return;
//     }
//     setStatus('uploading');
//     setProgress(0);
//     setError(null);
//     setUploadedFile(null);

//     try {
//       const signedUrlResult = await getSignedUrl({
//         name: fileToUpload.name,
//         type: fileToUpload.type,
//         size: fileToUpload.size,
//         userId,
//       });

//       if('failure' in signedUrlResult){
//         throw new Error(signedUrlResult.failure);
//       } else {
//         const { url, originalFileName, newFileName } = signedUrlResult.success;
//         await uploadFile(fileToUpload, url);
//         setStatus('success');
//         setUploadedFile({ originalFileName, newFileName });

//         const newFile: UserFile = {
//           name: fileToUpload.name,
//           size: fileToUpload.size,
//           type: fileToUpload.type,
//           createdAt: new Date().toISOString(),
//         };
//         dispatch(addFile(newFile));

//         toast(`${originalFileName} has been successfully uploaded.`);
//       }
//     } catch (err: any) {
//       setError(err.message || 'An unknown error occurred.');
//       setStatus('error');
//     }
//   };

//   const uploadFile = (fileToUpload: File, url: string) => {
//     return new Promise((resolve, reject) => {
//       const xhr = new XMLHttpRequest();
//       xhrRef.current = xhr;

//       xhr.open('PUT', url, true);
//       xhr.setRequestHeader('Content-Type', fileToUpload.type);

//       xhr.upload.onprogress = (event) => {
//         if (event.lengthComputable) {
//           const percentComplete = (event.loaded / event.total) * 100;
//           setProgress(percentComplete);
//         }
//       };

//       xhr.onload = () => {
//         if (xhr.status >= 200 && xhr.status < 300) {
//           resolve(xhr.response);
//         } else {
//           reject(new Error(`Upload failed with status: ${xhr.status}. ${xhr.responseText}`));
//         }
//       };

//       xhr.onerror = () => {
//         reject(new Error('Upload failed due to a network error.'));
//       };
      
//       xhr.onabort = () => {
//         reject(new Error('Upload was cancelled.'));
//       }

//       xhr.send(fileToUpload);
//     });
//   };

//   const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
//     e.preventDefault();
//     e.stopPropagation();
//     if (status !== 'uploading') setStatus('dragging');
//   };

//   const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
//     e.preventDefault();
//     e.stopPropagation();
//     if (status !== 'uploading') setStatus('idle');
//   };

//   const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
//     e.preventDefault();
//     e.stopPropagation();
//   };

//   const handleDrop = (e: DragEvent<HTMLDivElement>) => {
//     e.preventDefault();
//     e.stopPropagation();
//     setStatus('idle');
//     const droppedFile = e.dataTransfer.files[0];
//     handleFileSelect(droppedFile);
//   };
  
//   const handleReset = () => {
//     setFile(null);
//     setStatus('idle');
//     setProgress(0);
//     setError(null);
//     setUploadedFile(null);
//     if(inputRef.current) inputRef.current.value = "";
//   };
  
//   const handleCancel = () => {
//     if(xhrRef.current){
//       xhrRef.current.abort();
//     }
//     handleReset();
//   }

//   const renderContent = () => {
//     switch (status) {
//       case 'uploading':
//         return (
//           <div className="w-full text-center p-3">
//             <LoaderCircle className="mx-auto h-12 w-12 text-primary animate-spin mb-3" />
//             <p className="text-lg font-medium text-primary mb-2">Uploading...</p>
//             {file && <p className="text-muted-foreground mb-2 text-xs">{file.name}</p>}
//             <Progress value={progress} className="w-full h-3" />
//             <p className="text-sm font-medium text-primary mt-2">{Math.round(progress)}%</p>
//             <Button variant="outline" size="sm" className="mt-4" onClick={handleCancel}>Cancel</Button>
//           </div>
//         );
//       case 'success':
//         return (
//           <div className="w-full text-center p-3">
//             <CheckCircle2 className="mx-auto h-16 w-16 text-green-500 mb-4 animate-in fade-in zoom-in-50 duration-500" />
//             <h3 className="text-lg font-bold text-primary mb-2">Upload Successful!</h3>
//             {uploadedFile && (
//               <div className="text-muted-foreground text-xs space-y-1">
//                 <p>File name: <span className="font-medium text-foreground">{uploadedFile.originalFileName}</span></p>
//                 {/* <p>Stored as: <span className="font-medium text-foreground">{uploadedFile.newFileName}</span></p> */}
//               </div>
//             )}
//             <Button className="mt-8" onClick={handleReset}>Upload Another File</Button>
//           </div>
//         );
//       case 'error':
//         return (
//           <div className="w-full text-center p-8">
//             <Alert variant="destructive" className="mb-6 text-left">
//               <XCircle className="h-4 w-4" />
//               <AlertTitle>Upload Failed</AlertTitle>
//               <AlertDescription>{error}</AlertDescription>
//             </Alert>
//             <Button variant="destructive" onClick={handleReset}>Try Again</Button>
//           </div>
//         );
//       case 'idle':
//       case 'dragging':
//       default:
//         return (
//           <div
//             className={cn(
//               "relative border-2 border-dashed rounded-lg p-2 text-center transition-colors duration-300 cursor-pointer h-48 flex flex-col items-center justify-center",
//               status === 'dragging' ? 'border-primary bg-accent/20' : 'border-border hover:border-primary/50'
//             )}
//             onClick={() => inputRef.current?.click()}
//           >
//             <UploadCloud className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
//             <p className="text-sm font-semibold text-foreground">Drag & drop file</p>
//             <p className="text-muted-foreground text-xs mt-1">or click to browse</p>
//             <p className="text-[10px] text-muted-foreground mt-4">Supports: JSON, TXT, TSV, CSV, PDF (up to 5GB)</p>
//             <input
//               ref={inputRef}
//               type="file"
//               className="hidden"
//               onChange={(e) => handleFileSelect(e.target.files ? e.target.files[0] : null)}
//               accept={ALLOWED_FILE_TYPES.join(',')}
//             />
//           </div>
//         );
//     }
//   };

//   return (
//     <Card 
//       className="w-full p-0"
//       onDragEnter={handleDragEnter}
//       onDragLeave={handleDragLeave}
//       onDragOver={handleDragOver}
//       onDrop={handleDrop}
//     >
//       <CardContent className="p-2">
//         {renderContent()}
//       </CardContent>
//     </Card>
//   );
// }


'use client'

import { useState, type DragEvent, useRef } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { UploadCloud, CheckCircle2, XCircle, LoaderCircle } from 'lucide-react'
import { getSignedUrls } from '@/lib/ApiStore/actions/uploadaction'   // ⬅️ new import
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import type { RootState } from '@/store/store'
import { addFile, UserFile } from '@/store/slices/fileSlice'

/* ------------------------------------------------------------------ */
/*                          Helper types                              */
/* ------------------------------------------------------------------ */
type UploadStatus = 'idle' | 'uploading' | 'success' | 'error'

interface UploadItem {
  file: File
  progress: number
  status: UploadStatus
  error?: string
}

const ALLOWED_FILE_TYPES = [
  'application/json',
  'text/csv',
  'text/plain',
  'text/tab-separated-values',
  'application/pdf',
]

export function FileUploader() {
  const [uploads, setUploads] = useState<UploadItem[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const userId = useSelector((s: RootState) => s.user.userId)
  const dispatch = useDispatch()

  /* -------------------------------------------------------------- */
  /*                     UTILITY HELPERS                            */
  /* -------------------------------------------------------------- */
  const reset = () => {
    setUploads([])
    if (inputRef.current) inputRef.current.value = ''
  }

  const validateFiles = (files: File[]) => {
    const rejected: string[] = []
    const accepted = files.filter((f) => {
      const ok = ALLOWED_FILE_TYPES.includes(f.type)
      if (!ok) rejected.push(f.name)
      return ok
    })
    if (rejected.length) {
      toast(`Unsupported file types: ${rejected.join(', ')}`)
    }
    return accepted
  }

  /* -------------------------------------------------------------- */
  /*                       UPLOAD FLOW                              */
  /* -------------------------------------------------------------- */
  const startUpload = async (files: File[]) => {
    if (!userId) {
      toast('Please sign in first.')
      return
    }

    const initialState: UploadItem[] = files.map((file) => ({
      file,
      progress: 0,
      status: 'uploading',
    }))
    setUploads(initialState)

    try {
      /* 1️⃣  ask the server for signed URLs for every file */
      const res = await getSignedUrls(
        files.map((f) => ({ name: f.name, type: f.type, size: f.size })),
        userId,
      )
      if ('failure' in res) throw new Error(res.failure)

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
            }
            dispatch(addFile(newReduxFile))
            toast(`${originalFileName} uploaded`)
          }),
        ),
      )
    } catch (err: any) {
      // set every pending upload to error state
      setUploads((prev) =>
        prev.map((u) =>
          u.status === 'uploading'
            ? { ...u, status: 'error', error: err.message || 'Error' }
            : u,
        ),
      )
    }
  }

  const uploadSingle = (file: File, url: string, index: number) =>
    new Promise<void>((resolve, reject) => {
      const xhr = new XMLHttpRequest()

      xhr.open('PUT', url, true)
      xhr.setRequestHeader('Content-Type', file.type)

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const pct = (e.loaded / e.total) * 100
          setUploads((prev) =>
            prev.map((u, i) => (i === index ? { ...u, progress: pct } : u)),
          )
        }
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          setUploads((prev) =>
            prev.map((u, i) =>
              i === index ? { ...u, status: 'success', progress: 100 } : u,
            ),
          )
          resolve()
        } else {
          const err = `Upload failed: ${xhr.status}`
          setUploads((prev) =>
            prev.map((u, i) =>
              i === index ? { ...u, status: 'error', error: err } : u,
            ),
          )
          reject(new Error(err))
        }
      }

      xhr.onerror = () => {
        const err = 'Network error'
        setUploads((prev) =>
          prev.map((u, i) =>
            i === index ? { ...u, status: 'error', error: err } : u,
          ),
        )
        reject(new Error(err))
      }

      xhr.send(file)
    })

  /* -------------------------------------------------------------- */
  /*                     DOM EVENT HANDLERS                         */
  /* -------------------------------------------------------------- */
  const handleFiles = (fileList: FileList | null) => {
    if (!fileList) return
    const filesArr = validateFiles(Array.from(fileList))
    if (filesArr.length) startUpload(filesArr)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    handleFiles(e.dataTransfer.files)
  }

  /* -------------------------------------------------------------- */
  /*                       RENDER HELPERS                           */
  /* -------------------------------------------------------------- */
  const renderBody = () => {
    if (uploads.length === 0)
      return (
        <div
          className="relative border-2 border-dashed rounded-lg p-2 text-center transition-colors duration-300 cursor-pointer h-48 flex flex-col items-center justify-center hover:border-primary/50"
          onClick={() => inputRef.current?.click()}
        >
          <UploadCloud className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <p className="text-sm font-semibold text-foreground">Drag & drop files</p>
          <p className="text-muted-foreground text-xs mt-1">or click to browse</p>
          <p className="text-[10px] text-muted-foreground mt-4">
            Supports: JSON, TXT, TSV, CSV, PDF (up to 5GB)
          </p>
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            multiple                                   // ⬅️ allow many files
            accept={ALLOWED_FILE_TYPES.join(',')}
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      )

    /* show each file row */
    return uploads.map((u, idx) => (
      <div key={idx} className="border rounded-md p-3 mb-2">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs font-medium truncate max-w-[150px]">{u.file.name}</p>
          {u.status === 'success' && <CheckCircle2 className="text-green-500 w-4 h-4" />}
          {u.status === 'error' && <XCircle className="text-red-500 w-4 h-4" />}
          {u.status === 'uploading' && (
            <LoaderCircle className="animate-spin text-primary w-4 h-4" />
          )}
        </div>
        {u.status === 'uploading' && (
          <Progress value={u.progress} className="h-2" />
        )}
        {u.status === 'error' && (
          <Alert variant="destructive" className="mt-2">
            <AlertTitle>Upload failed</AlertTitle>
            <AlertDescription className="text-xs">{u.error}</AlertDescription>
          </Alert>
        )}
      </div>
    ))
  }

  return (
    <Card className="w-full p-0" onDragOver={(e) => e.preventDefault()} onDrop={handleDrop}>
      <CardContent className="p-2">{renderBody()}</CardContent>
    </Card>
  )
}
