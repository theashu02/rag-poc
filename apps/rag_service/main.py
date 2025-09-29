import os
import tempfile
import asyncio
from google.cloud import storage
import functions_framework
from concurrent.futures import ThreadPoolExecutor

from services.pipeline import process_file, upload_to_pinecone

storage_client = storage.Client()

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))

@functions_framework.cloud_event
def handle_gcs_event(cloud_event):

    print(f"Received Cloud Storage event: {cloud_event.data}")

    try:
        event_data = cloud_event.data
        bucket_name = event_data["bucket"]
        file_path = event_data["name"]

        # Expect path: "Data/user_id/filename"
        path_parts = file_path.split('/')
        if len(path_parts) < 3 or path_parts[0] != 'Data':
            print(f"ERROR: File path '{file_path}' is not in the expected 'Data/UserID/file.txt' format.")
            return

        user_id = path_parts[1]
        original_filename = os.path.basename(file_path)
        if not original_filename:
            print(f"Ignoring folder creation event for path: {file_path}")
            return

        print(f"Extracted userID: '{user_id}', filename: '{original_filename}'")

        # Process file asynchronously to avoid blocking
        asyncio.run(process_file_async(
            bucket_name, file_path, user_id, original_filename
        ))

    except Exception as e:
        print(f"🚨 A critical error occurred: {e}")
        # Consider adding retry logic or dead-letter queue for failed processing
        raise e

async def process_file_async(bucket_name, file_path, user_id, original_filename):
    """Process file asynchronously to avoid blocking the main function"""
    loop = asyncio.get_running_loop()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        await loop.run_in_executor(
            executor, 
            process_single_file,
            bucket_name, file_path, user_id, original_filename
        )

def process_single_file(bucket_name, file_path, user_id, original_filename):
    """Process a single file - extracted for clarity"""
    try:
        # Download the file to a temp location
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        with tempfile.NamedTemporaryFile() as temp_local_file:
            print(f"Downloading gs://{bucket_name}/{file_path} to {temp_local_file.name}...")
            blob.download_to_filename(temp_local_file.name)

            print(f"Processing file for user '{user_id}'...")
            chunks = process_file(file_path=temp_local_file.name, original_filename=original_filename)

            if not chunks:
                print("No chunks were generated from the file.")
                return

            print(f"Uploading {len(chunks)} chunks to Pinecone using namespace: '{user_id}'")
            total_uploaded = upload_to_pinecone(chunks, namespace=user_id)
            
        print(f"✅ Success! Processed and uploaded {total_uploaded} vectors for {file_path}.")
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")