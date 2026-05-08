import os
import sys
from dotenv import load_dotenv

load_dotenv()

import main # imports our Cloud Function code


class MockCloudEvent:
    """Minimal stand-in for a CloudEvent — only `.data` is needed by ingest_pdf."""
    def __init__(self, data):
        self.data = data


bucket = os.environ.get("GCS_BUCKET_NAME", "smartstudy-pdfs-smartstudy-infoh505")
file_name = sys.argv[1] if len(sys.argv) > 1 else "test-home-insurance.pdf"

print(f"\nTriggering ingest_pdf for gs://{bucket}/{file_name}\n")

event = MockCloudEvent(data={
    "bucket": bucket,
    "name": file_name,
    "contentType": "application/pdf",
})

main.ingest_pdf(event)
print("\nDone.\n")