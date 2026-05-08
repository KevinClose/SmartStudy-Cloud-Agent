from dotenv import load_dotenv
load_dotenv()  # in production this is a no-op (no .env file present)


import os
import logging
import tempfile
import functions_framework
from typing import List
from google.cloud import storage
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Step 1: Extract text from a PDF stored in GCS
def extract_pdf_text(bucket_name, file_name):
    """Downloads a PDF from a GCS bucket and extracts its text page by page."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        local_path = tmp.name

    try:
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded gs://{bucket_name}/{file_name} -> {local_path}")

        loader = PyPDFLoader(local_path)
        pages = loader.load() # one Document per page

        for page in pages:
            page.metadata["source"] = file_name

        logger.info(f"Extracted {len(pages)} pages from {file_name}")
        return pages

    finally:
        # Clean up the temp file even if extraction fails.
        if os.path.exists(local_path):
            os.remove(local_path)


# Step 2: Split each page into smaller, overlapping text chunks
def chunk_documents(pages, chunk_size=1000, chunk_overlap=200):
    """Splits a list of page-level Documents into smaller, overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)
    logger.info(f"Split {len(pages)} pages into {len(chunks)} chunks "
                f"(size={chunk_size}, overlap={chunk_overlap})")
    
    return chunks


# Step 3: Embed chunks and upsert them into MongoDB Atlas Vector Search
def embed_and_upsert(chunks):
    """Embeds each text chunk via Vertex AI's `text-embedding-005` (768 dims) and inserts the chunks into the MongoDB Atlas vector collection."""

    if not chunks:
        logger.warning("No chunks to embed, skipping upsert.")
        return 0
    
    # Read configuration from env vars
    mongo_uri = os.environ["MONGODB_URI"]
    db_name = os.environ.get("MONGODB_DB_NAME", "smartstudy")
    collection_name = os.environ.get("MONGODB_CONTEXT_COLLECTION", "documents")
    index_name = os.environ.get("MONGODB_VECTOR_INDEX", "vector_index")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-005")

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    collection = client[db_name][collection_name]

    # Idempotency: remove any previous chunks from the same source
    source = chunks[0].metadata.get("source")
    if source:
        result = collection.delete_many({"source": source})
        if result.deleted_count:
            logger.info(
                f"Removed {result.deleted_count} stale chunks for "
                f"source='{source}' before re-ingestion."
            )

    # Vertex AI embeddings (768 dims)
    embeddings = VertexAIEmbeddings(model=embedding_model)

    # LangChain wrapper that handles embed + insert in one shot
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=index_name,
        text_key="text", # field name that stores the raw chunk text
        embedding_key="embedding", # field name that stores the vector
    )

    inserted_ids = vector_store.add_documents(chunks)
    logger.info(
        f"Inserted {len(inserted_ids)} chunks into "
        f"{db_name}.{collection_name} (model={embedding_model})"
    )

    return len(inserted_ids)


# Cloud Function entry point: orchestrates the 3 steps above
@functions_framework.cloud_event
def ingest_pdf(cloud_event):
    """
    Cloud Function triggered by a `google.cloud.storage.object.v1.finalized`
    event on the SmartStudy PDFs bucket.

    Pipeline:
        1. Extract text from the PDF (per page).
        2. Split each page into chunks.
        3. Embed chunks via Vertex AI and upsert them in MongoDB Atlas.
    """

    data = cloud_event.data
    bucket_name  = data["bucket"]
    file_name = data["name"]
    content_type = data.get("contentType", "")

    logger.info(
        f"Received GCS event: gs://{bucket_name}/{file_name} "
        f"(contentType={content_type})"
    )

    # Filter: only ingest .pdf files
    if not file_name.lower().endswith(".pdf"):
        logger.info(f"Skipping non-PDF file: {file_name}")
        return

    try:
        # Step 1: Extract
        pages = extract_pdf_text(bucket_name, file_name)
        if not pages:
            logger.warning(f"No pages extracted from {file_name}, aborting.")
            return

        # Step 2: Chunk
        chunks = chunk_documents(pages)
        if not chunks:
            logger.warning(f"No chunks produced from {file_name}, aborting.")
            return

        # Step 3: Embed + upsert
        n = embed_and_upsert(chunks)
        logger.info(f"Ingestion complete: {file_name} -> {n} chunks indexed.")

    except Exception as exc:
        # Log full stack trac, re-raise so Cloud Functions marks the run as failed
        logger.exception(f"Ingestion failed for {file_name}: {exc}")
        raise