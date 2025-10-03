import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import functions_framework
from google.cloud import storage

from services.pipeline import process_file
from services.uploader import upload_to_pinecone

storage_client = storage.Client()
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)


@functions_framework.cloud_event
def handle_gcs_event(cloud_event):
    print(f"Received Cloud Storage event: {cloud_event.data}")

    try:
        event_data = cloud_event.data
        bucket_name = event_data["bucket"]
        file_path = event_data["name"]

        path_parts = file_path.split("/")
        if len(path_parts) < 3 or path_parts[0] != "Data":
            print(
                f"[Ingest] Skipping '{file_path}' because it is not in 'Data/<user_id>/<file>' format."
            )
            return

        user_id = path_parts[1]
        original_filename = os.path.basename(file_path)
        if not original_filename:
            print(f"[Ingest] Ignoring folder placeholder event for path: {file_path}")
            return

        print(f"[Ingest] user_id={user_id} filename={original_filename}")
        asyncio.run(
            process_file_async(
                bucket_name,
                file_path,
                user_id,
                original_filename,
            )
        )

    except Exception as exc:
        print(f"[Ingest] Critical error: {exc}")
        raise


def _download_blob_to_temp(bucket_name: str, file_path: str) -> str:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    blob.download_to_filename(temp_file.name)
    print(f"[Ingest] Downloaded gs://{bucket_name}/{file_path} to {temp_file.name}")
    return temp_file.name


async def process_file_async(bucket_name: str, file_path: str, user_id: str, original_filename: str):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        _EXECUTOR,
        process_single_file,
        bucket_name,
        file_path,
        user_id,
        original_filename,
    )


def process_single_file(bucket_name: str, file_path: str, user_id: str, original_filename: str):
    temp_local_file = None
    try:
        temp_local_file = _download_blob_to_temp(bucket_name, file_path)
        chunks = process_file(file_path=temp_local_file, original_filename=original_filename)

        if not chunks:
            print(f"[Ingest] No chunks generated for {original_filename}")
            return

        print(f"[Ingest] Uploading {len(chunks)} chunks to Pinecone namespace '{user_id}'")
        total_uploaded = upload_to_pinecone(chunks, namespace=user_id)
        print(
            f"[Ingest] Uploaded {total_uploaded} vectors for gs://{bucket_name}/{file_path}"
        )

    except Exception as exc:
        print(f"[Ingest] Error processing {file_path}: {exc}")
        raise
    finally:
        if temp_local_file and os.path.exists(temp_local_file):
            try:
                os.unlink(temp_local_file)
            except OSError as cleanup_exc:
                print(f"[Ingest] Cleanup warning: {cleanup_exc}")
