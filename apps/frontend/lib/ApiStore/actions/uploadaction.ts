'use server';

import { Storage, type File } from '@google-cloud/storage';
import { z } from 'zod';

const gcsProjectId = process.env.GCS_PROJECT_ID;
const gcsClientEmail = process.env.GCS_CLIENT_EMAIL;
const gcsPrivateKey = process.env.GCS_PRIVATE_KEY;
const gcsBucketName = process.env.GCS_BUCKET_NAME;

console.log('Environment check:', {
  projectId: !!gcsProjectId,
  clientEmail: !!gcsClientEmail,
  privateKey: !!gcsPrivateKey,
  bucketName: gcsBucketName
});

if (!gcsProjectId || !gcsClientEmail || !gcsPrivateKey || !gcsBucketName) {
  throw new Error('Google Cloud Storage environment variables are not set.');
}

let storage: Storage;
let bucket: any;

try {
  storage = new Storage({
    projectId: gcsProjectId,
    credentials: {
      client_email: gcsClientEmail,
      private_key: gcsPrivateKey.replace(/\\n/g, '\n').trim(),
    },
  });

  bucket = storage.bucket(gcsBucketName);
} catch (error) {
  console.error('Storage initialization error:', error);
  throw new Error('Failed to initialize Google Cloud Storage');
}

const MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024; // 5GB
const ALLOWED_FILE_TYPES = [
  'application/json',
  'text/csv',
  'text/plain',
  'text/tab-separated-values',
  'application/pdf',
  "application/vnd.openxmlformats-officedocument.presentationml.presentation", // .pptx
  "application/msword", // .doc
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // .docx
];

const actionSchema = z.object({
  name: z.string().min(1),
  type: z.string().min(1),
  size: z.number().positive(),
});

export async function getSignedUrl(rawInput: { name: string; type: string; size: number; userId: string; }) {
  const { userId, ...file} = rawInput;
  try {
    console.log('Input received:', rawInput, userId);

    const input = actionSchema.parse(file);

    if (input.size > MAX_FILE_SIZE) {
      return { failure: 'File is too large. Maximum size is 5GB.' };
    }

    if (!ALLOWED_FILE_TYPES.includes(input.type)) {
      return { failure: `File type "${input.type}" is not allowed.` };
    }
    
    // const newFileName = await getNextFileName(input.name, userId);
    const newFileName = input.name;
    const filePath = `Data/${userId}/${newFileName}`;

    console.log('Generating signed URL for:', filePath);

    const options = {
      version: 'v4' as const,
      action: 'write' as const,
      expires: Date.now() + 15 * 60 * 1000, 
      contentType: input.type,
    };

    // Test bucket access
    try {
      await bucket.exists();
      console.log('Bucket exists and is accessible');
    } catch (bucketError) {
      console.error('Bucket access error:', bucketError);
      return { failure: 'Unable to access storage bucket. Please check bucket permissions.' };
    }

    const [url] = await bucket.file(filePath).getSignedUrl(options);

    console.log('Signed URL generated successfully');

    return { success: { url, originalFileName: input.name, newFileName } };
  } catch (error) {
    console.error('Error getting signed URL:', error);
    
    if (error instanceof z.ZodError) {
      console.error('Validation error:', error.errors);
      return { failure: `Invalid input: ${error.errors.map(e => e.message).join(', ')}` };
    }
    
    // More specific error handling
    if (error instanceof Error) {
      if (error.message.includes('private_key')) {
        return { failure: 'Invalid service account private key. Please check your GCS_PRIVATE_KEY environment variable.' };
      }
      if (error.message.includes('client_email')) {
        return { failure: 'Invalid service account email. Please check your GCS_CLIENT_EMAIL environment variable.' };
      }
      if (error.message.includes('project')) {
        return { failure: 'Invalid project ID. Please check your GCS_PROJECT_ID environment variable.' };
      }
      if (error.message.includes('bucket')) {
        return { failure: 'Bucket not found or inaccessible. Please check your GCS_BUCKET_NAME and permissions.' };
      }
      
      return { failure: `Error: ${error.message}` };
    }
    
    return { failure: 'Could not get signed URL. Please check server configuration and try again.' };
  }
}

export async function getSignedUrls(
  files: { name: string; type: string; size: number }[],
  userId: string,
) {
  try {
    const results = await Promise.all(
      files.map((f) => getSignedUrl({ ...f, userId })),
    )

    const failures = results.filter((r) => 'failure' in r) as { failure: string }[]
    if (failures.length) {
      // just return the first error for simplicity
      return { failure: failures[0].failure }
    }

    return {
      success: results.map((r) => (r as any).success),
    }
  } catch (err: any) {
    return { failure: err.message || 'Could not get signed URLs' }
  }
}

// get the list based on the user id from the GCS
export async function getUserFiles(userId: string) {
  if (!userId) {
    return { failure: 'User not authenticated.' };
  }

  try {
    const prefix = `Data/${userId}/`;
    const [files] = await bucket.getFiles({ prefix });

    if (files.length === 1 && files[0].name === prefix) {
      return { success: [] };
    }
    const fileDetails = files
      .filter((file: File) => file.name !== prefix)
      .map((file: File) => ({
        name: file.name.replace(prefix, ''), 
        size: file.metadata.size,
        createdAt: file.metadata.timeCreated,
      }));
    return { success: fileDetails };
  } catch (error) {
    console.error('Error getting user files:', error);
    return { failure: 'Could not retrieve file list. Please try again later.' };
  }
}

export async function deleteUserFile(userId: string, fileName: string) {
  if (!userId) {
    return { failure: 'User not authenticated.' };
  }
  if (!fileName) {
    return { failure: 'File name not provided.' };
  }
  try {
    const filePath = `Data/${userId}/${fileName}`;

    console.log(`Attempting to delete file: ${filePath}`);
    await bucket.file(filePath).delete();
    console.log(`Successfully deleted file: ${filePath}`);
    return { success: true };
    
  } catch (error: any) {
    console.error(`Error deleting file for user ${userId}:`, error);
    if (error.code === 404) {
      return { failure: 'File not found. It may have already been deleted.' };
    }
    return { failure: 'Could not delete the file. Please try again later.' };
  }
}

export async function downloadUserFile(userId: string, fileName: string){
  if (!userId) {
    return { failure: 'User not autheticated.' };
  }
  if (!fileName) {
    return { failure: 'File name not provided.' };
  }
  try {
    const filePath = `Data/${userId}/${fileName}`;
    const options = {
      version: 'v4' as const,
      action: 'read' as const,
      expires: Date.now() + 15 * 60 * 1000, // 15 minutes
      responseDisposition: `attachment; filename="${fileName}"`,
    }

    const [url] = await bucket.file(filePath).getSignedUrl(options);
    return { success: {url} };
  } catch (error: any) {
    console.log(`Error to dowm=nload the file ${userId}: `, error);
    if(error.code === 404) {
      return { failure: 'File not found, It may have not in the bucket.' };
    }
    return { failure: 'Could not able to download the file. Please try again later.' };
  }
}