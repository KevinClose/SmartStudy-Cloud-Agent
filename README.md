# SmartStudy Cloud Agent

An AI tutor that answers students' questions based on their own course PDFs. Upload your slides, ask a question, and get a grounded answer with page citations and a follow-up question to deepen your understanding.

## Live demo

https://smartstudy-app-930005856895.europe-west1.run.app

## Stack

Streamlit on Cloud Run for the UI, a Cloud Function (Python) for PDF ingestion, MongoDB Atlas Vector Search for retrieval, Vertex AI Gemini 2.5 Flash for generation, and `text-embedding-005` for embeddings. Orchestrated with LangChain LCEL.

## Run locally

Requires Python 3.11+, a Google Cloud project with the gcloud CLI, and a MongoDB Atlas cluster (free M0 is enough) with a vector index named `vector_index` on the `documents` collection (768 dimensions, cosine similarity, path `embedding`).

```bash
git clone https://github.com/<your-username>/SmartStudy-Cloud-Agent.git
cd SmartStudy-Cloud-Agent/smartstudy

python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
pip install -r cloud_function/requirements.txt

cp .env.example .env
cp app/.env.example app/.env
cp cloud_function/.env.example cloud_function/.env
# fill the .env files with your own values (MongoDB URI, GCP project ID, etc.)

# place your service account JSON key at the repo root
mv ~/Downloads/your-key.json service-account-key.json

streamlit run app/app.py
```

The app then opens at `localhost:8501`.

## Repository layout

```
smartstudy/
├── app/                Streamlit UI + RAG chain
├── cloud_function/     PDF ingestion function
├── scripts/            Utility scripts (e.g. wipe.py)
└── docs/               Architecture diagram
```
