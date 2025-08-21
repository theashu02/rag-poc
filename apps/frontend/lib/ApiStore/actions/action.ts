'use server';

import { Storage } from '@google-cloud/storage';
import { z } from 'zod';
import path from 'path';

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

// const userId = localStorage.getItem('userId');

const MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024; // 5GB
const ALLOWED_FILE_TYPES = [
  'application/json',
  'text/csv',
  'text/plain',
  'text/tab-separated-values',
  'application/pdf',
];

const userId = "userxyz"

const getNextFileName = async (originalFileName: string) => {
  try {

    const prefix = `Data/${userId}/`;
    const [files] = await bucket.getFiles({ 
      prefix,
      maxResults: 1000 // Add limit to avoid timeout on large buckets
    });
    
    const fileNumbers = files
      .map((file: any) => {
        const match = file.name.match(new RegExp(`^${prefix}file(\\d+)\\..+$`));
        return match ? parseInt(match[1], 10) : 0;
      })
      .filter((num: number) => num > 0);

    const nextFileNumber = fileNumbers.length > 0 ? Math.max(...fileNumbers) + 1 : 1;
    const extension = path.extname(originalFileName);
    return `file${nextFileNumber}${extension}`;
  } catch (error) {
    console.error('Error getting next file name:', error);
    // Fallback to timestamp-based naming
    const timestamp = Date.now();
    const extension = path.extname(originalFileName);
    return `file_${timestamp}${extension}`;
  }
};

const actionSchema = z.object({
  name: z.string().min(1),
  type: z.string().min(1),
  size: z.number().positive(),
});

export async function getSignedUrl(rawInput: { name: string; type: string; size: number }) {
  try {
    console.log('Input received:', rawInput);

    const input = actionSchema.parse(rawInput);

    if (input.size > MAX_FILE_SIZE) {
      return { failure: 'File is too large. Maximum size is 5GB.' };
    }

    if (!ALLOWED_FILE_TYPES.includes(input.type)) {
      return { failure: `File type "${input.type}" is not allowed.` };
    }
    
    const newFileName = await getNextFileName(input.name);
    const filePath = `Data/${userId}/${newFileName}`;

    console.log('Generating signed URL for:', filePath);

    const options = {
      version: 'v4' as const,
      action: 'write' as const,
      expires: Date.now() + 15 * 60 * 1000, // 15 minutes
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