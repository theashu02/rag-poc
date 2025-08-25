import os
import tempfile
from google.cloud import storage
import functions_framework

from rag_pipeline import process_file, upload_to_pinecone

# Initialize the Google Cloud Storage client globally
storage_client = storage.Client()

# CloudEvent handler for GCS triggers
@functions_framework.cloud_event
def handle_gcs_event(cloud_event):
    """
    Triggered by a change to a Cloud Storage bucket.
    Downloads the file, extracts userID, and processes it.
    """
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

        # Download the file to a temp location
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        with tempfile.NamedTemporaryFile() as temp_local_file:
            print(f"Downloading gs://{bucket_name}/{file_path} to {temp_local_file.name}...")
            blob.download_to_filename(temp_local_file.name)

            # Process the file
            print(f"Processing file for user '{user_id}'...")
            chunks = process_file(file_path=temp_local_file.name, original_filename=original_filename)

            if not chunks:
                print("No chunks were generated from the file.")
                return

            print(f"Uploading {len(chunks)} chunks to Pinecone using namespace: '{user_id}'")
            total_uploaded = upload_to_pinecone(chunks, namespace=user_id)

        print(f"✅ Success! Processed and uploaded {total_uploaded} vectors for {file_path}.")

    except Exception as e:
        print(f"🚨 A critical error occurred: {e}")
        raise e


# import os
# import tempfile
# from google.cloud import storage
# import functions_framework

# # Import the key functions from your RAG script
# from rag_pipeline import process_file, upload_to_pinecone

# # Initialize the Google Cloud Storage client (it's best to do this globally)
# storage_client = storage.Client()

# # This decorator registers the function to handle CloudEvents from Cloud Storage.
# @functions_framework.cloud_event
# def handle_gcs_event(cloud_event):
#     """
#     Receives a CloudEvent from a Cloud Storage trigger, downloads the file,
#     extracts the userID, and triggers the RAG processing pipeline.
#     """
#     print(f"Received Cloud Storage event: {cloud_event.data}")

#     try:
#         # The event data is in the 'data' attribute of the CloudEvent object
#         event_data = cloud_event.data
#         bucket_name = event_data["bucket"]
#         file_path = event_data["name"]

#         # 1. Extract the userID from the file path
#         # The path is expected to be: "Data/some_user_id/some_file.txt"
#         path_parts = file_path.split('/')

#         if len(path_parts) < 3 or path_parts[0] != 'Data':
#             msg = f"File path '{file_path}' is not in the expected 'Data/UserID/file.txt' format."
#             print(f"ERROR: {msg}")
#             return  # Exit the function

#         user_id = path_parts[1]
#         original_filename = os.path.basename(file_path)
        
#         # This check ensures the event is for a file, not a folder creation
#         if not original_filename:
#             print(f"Ignoring folder creation event for path: {file_path}")
#             return # Exit the function

#         print(f"Successfully extracted userID: '{user_id}' and filename: '{original_filename}'")

#         # 2. Download the uploaded file to a temporary location
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.blob(file_path)
        
#         with tempfile.NamedTemporaryFile() as temp_local_file:
#             print(f"Downloading gs://{bucket_name}/{file_path} to {temp_local_file.name}...")
#             blob.download_to_filename(temp_local_file.name)

#             # 3. Execute your RAG pipeline
#             print(f"Processing file for user '{user_id}'...")
#             chunks = process_file(file_path=temp_local_file.name, original_filename=original_filename)

#             if not chunks:
#                 print("No chunks were generated from the file.")
#                 return # Exit the function

#             print(f"Uploading {len(chunks)} chunks to Pinecone using namespace: '{user_id}'")
#             total_uploaded = upload_to_pinecone(chunks, namespace=user_id)

#         print(f"✅ Success! Processed and uploaded {total_uploaded} vectors for {file_path}.")
#         # In a Cloud Function, you don't need to return a JSON response for background events.
#         # Logging is sufficient.

#     except Exception as e:
#         print(f"🚨 A critical error occurred: {e}")
#         # Raising an exception can help in retrying the function if configured to do so.
#         raise e