import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from google.cloud import storage

load_dotenv()

# Confirm before nuking
print("This will DELETE:")
print("  - All PDFs in the GCS bucket")
print("  - All chunks in MongoDB `documents` collection")
print("  - All chat history in `chat_history` collection")
confirm = input("Type 'YES' to confirm: ")
if confirm != "YES":
    print("Aborted.")
    sys.exit(0)

# MongoDB
mongo = MongoClient(os.environ["MONGODB_URI"])
db = mongo[os.environ.get("MONGODB_DB_NAME", "smartstudy")]

docs_count = db["documents"].count_documents({})
hist_count = db["chat_history"].count_documents({})
db["documents"].delete_many({})
db["chat_history"].delete_many({})
print(f"✅ MongoDB: deleted {docs_count} chunks and {hist_count} chat history entries.")

# GCS
bucket_name = os.environ.get("GCS_BUCKET_NAME") or f"smartstudy-pdfs-{os.environ.get('GCP_PROJECT_ID', '')}"
gcs = storage.Client()
bucket = gcs.bucket(bucket_name)
blobs = list(bucket.list_blobs())
for blob in blobs:
    blob.delete()
print(f"✅ GCS: deleted {len(blobs)} files from gs://{bucket_name}/")

print("\nDone. The database is now empty.")