import os
import logging
import uuid
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from google.cloud import storage
from pymongo import MongoClient
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(
    page_title="SmartStudy — Academic Tutor",
    page_icon="🎓",
    layout="wide",
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration (from env vars)
MONGODB_URI = os.environ["MONGODB_URI"]
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "smartstudy")
MONGODB_CONTEXT_COLLECTION = os.environ.get("MONGODB_CONTEXT_COLLECTION", "documents")
MONGODB_HISTORY_COLLECTION = os.environ.get("MONGODB_HISTORY_COLLECTION", "chat_history")
MONGODB_VECTOR_INDEX = os.environ.get("MONGODB_VECTOR_INDEX", "vector_index")
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-005")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")


# PDF management helpers
def _upload_pdf_to_gcs(uploaded_file):
    """Push the PDF to GCS: Cloud Function handles ingestion automatically."""

    bucket = storage.Client().bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(uploaded_file.name)
    blob.upload_from_string(
        uploaded_file.getvalue(),
        content_type="application/pdf",
    )


def _delete_pdf_everywhere(filename):
    """Remove the PDF from GCS AND all its chunks from MongoDB.
    Returns the number of chunks deleted."""

    bucket = storage.Client().bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(filename)
    if blob.exists():
        blob.delete()

    coll = MongoClient(MONGODB_URI)[MONGODB_DB_NAME][MONGODB_CONTEXT_COLLECTION]
    result = coll.delete_many({"source": filename})

    return result.deleted_count


@st.cache_data
def list_ingested_pdfs():
    """Returns the sorted list of unique PDF filenames currently in MongoDB."""

    coll = MongoClient(MONGODB_URI)[MONGODB_DB_NAME][MONGODB_CONTEXT_COLLECTION]

    return sorted(coll.distinct("source"))


# Build the retriever (MongoDB Atlas Vector Search)
@st.cache_resource
def build_retriever(k):
    """Returns a LangChain retriever wired to MongoDB Atlas Vector Search."""

    client = MongoClient(MONGODB_URI)
    collection = client[MONGODB_DB_NAME][MONGODB_CONTEXT_COLLECTION]

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=VertexAIEmbeddings(model=EMBEDDING_MODEL),
        index_name=MONGODB_VECTOR_INDEX,
        text_key="text",
        embedding_key="embedding",
    )
    logger.info(f"Retriever ready (collection={MONGODB_CONTEXT_COLLECTION}, k={k}).")

    return vector_store.as_retriever(search_kwargs={"k": k})


retriever = build_retriever(k=5)


# Tutor persona prompt
SYSTEM_PROMPT = """\
You are SmartStudy, a formal academic tutor specialized in helping university students prepare for exams.

# Your Pedagogical Style
- Adopt the tone of a patient, rigorous mentor — clear, structured, never condescending.
- Always answer in the same language as the student's question (English, French, Dutch, etc.).
- Structure complex answers: short overview → detailed explanation → concrete example when relevant.
- Use **bold** sparingly to highlight key terms; use bullet lists only when listing more than three items.
- Avoid filler phrases like "Great question!" or "Of course!". Get to the point.

# Grounding Rules (CRITICAL)
- Base your answer EXCLUSIVELY on the <context> below, which contains excerpts from the student's course materials.
- If the context does not contain enough information to answer, SAY SO explicitly:
  "I don't find this in your course materials. Could you upload the relevant chapter, or rephrase your question?"
- DO NOT use general knowledge from your training to fill gaps, unless you clearly label it as
  "(general knowledge — verify with your textbook)".

# Source Citations (CRITICAL)
- Every factual claim MUST end with an inline citation in the form: [filename.pdf, p.X]
- If a claim spans multiple sources, list all of them: [fileA.pdf, p.3] [fileB.pdf, p.7]
- The filename comes from the chunk's `source` field and the page from `page_label`.

# Pedagogical Engagement
- After your main answer, ask ONE short follow-up question to deepen the student's understanding.
  Good examples:
    - "Would you like me to walk you through a worked example?"
    - "Can you explain in your own words why X follows from Y?"
    - "Which part of this concept do you find least intuitive?"
- The follow-up should target a specific learning objective, not be generic ("Any other questions?" is bad).

# Off-topic Questions
- If the student asks something unrelated to academic study (weather, jokes, personal advice), politely redirect:
  "I'm focused on helping you with your course materials. What topic would you like to study?"
"""

USER_PROMPT_TEMPLATE = """\
<context>
{context}
</context>

<student_question>
{question}
</student_question>
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("user", USER_PROMPT_TEMPLATE),
])


# Build the LLM (Gemini 2.5 Flash)
@st.cache_resource
def build_llm():
    return ChatVertexAI(
        model_name=LLM_MODEL,
        temperature=0.3,
        max_output_tokens=2048,
    )


llm = build_llm()


# Format retrieved chunks for the prompt context
def format_docs(docs):
    if not docs:
        return "(no relevant chunks found in your course materials)"

    formatted = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        page_label = d.metadata.get("page_label")
        if page_label is None:
            page_idx = d.metadata.get("page")
            page_label = str(page_idx + 1) if isinstance(page_idx, int) else "?"
        formatted.append(f"[{source}, p.{page_label}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)


# RAG chain (LCEL)
rag_chain = (
    RunnablePassthrough.assign(
        docs=lambda x: retriever.invoke(x["question"])
    )
    | RunnablePassthrough.assign(
        context=lambda x: format_docs(x["docs"])
    )
    | RunnablePassthrough.assign(
        answer=prompt | llm | StrOutputParser()
    )
)


# Conversation memory (MongoDB-backed)
def get_chat_history(session_id: str):
    return MongoDBChatMessageHistory(
        connection_string=MONGODB_URI,
        session_id=session_id,
        database_name=MONGODB_DB_NAME,
        collection_name=MONGODB_HISTORY_COLLECTION,
    )


# Per-Streamlit-session unique session_id
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id
chat_history = get_chat_history(session_id)


# Sidebar
with st.sidebar:
    st.title("🎓 SmartStudy")
    st.caption("Your formal academic tutor")

    st.divider()

    if st.button("🗑️ Clear conversation"):
        chat_history.clear()
        st.session_state.session_id = str(uuid.uuid4())
        if "pdf_uploader" in st.session_state:
            del st.session_state["pdf_uploader"]
        st.rerun()

    # Upload section
    st.divider()
    st.subheader("📤 Upload a PDF")

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader(
        label="Drop a course PDF here",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_key}",
        help=(
            f"Auto-uploads to `gs://{GCS_BUCKET_NAME}/`. "
            "Cloud Function will ingest it (~30 sec)."
        ),
    )

    # Auto-upload on selection, then clear the widget so the
    # filename + cross don't linger in the sidebar.
    if (uploaded_file is not None
            and uploaded_file.name not in st.session_state.uploaded_files):
        with st.spinner(f"Uploading `{uploaded_file.name}`…"):
            try:
                _upload_pdf_to_gcs(uploaded_file)
                st.session_state.uploaded_files.add(uploaded_file.name)
                # Force the file_uploader to reset by changing its key
                st.session_state.uploader_key += 1
                # Invalidate the ingested-list cache
                list_ingested_pdfs.clear()
                st.toast(
                    f"✅ `{uploaded_file.name}` uploaded! ",
                    icon="📨",
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    # Ingested PDFs
    st.divider()
    st.subheader("📚 Uploaded PDFs")

    ingested_pdfs = list_ingested_pdfs()
    ingested_set  = set(ingested_pdfs)

    # PDFs uploaded this session but not yet visible in MongoDB -> "pending"
    pending_pdfs = sorted(st.session_state.uploaded_files - ingested_set)

    # Auto-refresh every 3 seconds while there are PDFs being ingested.
    # Stops automatically once all pending PDFs have been ingested
    # (or after `limit` refreshes as a safety net).
    if pending_pdfs:
        list_ingested_pdfs.clear() # force fresh MongoDB query at each rerun
        st_autorefresh(interval=3000, key="pending_refresh", limit=20)

    if not ingested_pdfs and not pending_pdfs:
        st.caption("_No PDFs uploaded yet. Upload one above._")
    else:
        # Show pending PDFs (still being ingested by the Cloud Function)
        for pdf_name in pending_pdfs:
            col_name, col_btn = st.columns([4, 1])
            col_name.caption(f"⏳ `{pdf_name}` _(ingesting…)_")
            col_btn.button(
                "🗑️",
                key=f"del_pending_{pdf_name}",
                disabled=True,
                help="Wait until ingestion finishes to delete",
            )

        # Show fully ingested PDFs
        for pdf_name in ingested_pdfs:
            col_name, col_btn = st.columns([4, 1])
            col_name.caption(f"📄 `{pdf_name}`")
            if col_btn.button(
                "🗑️",
                key=f"del_{pdf_name}",
                help=f"Remove {pdf_name} from your tutor",
            ):
                try:
                    n = _delete_pdf_everywhere(pdf_name)
                    list_ingested_pdfs.clear()
                    st.session_state.uploaded_files.discard(pdf_name)
                    st.toast(f"🗑️ Removed `{pdf_name}` ({n} chunks)")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Deletion failed: {exc}")


# Main chat area
st.title("Ask your tutor")

for message in chat_history.messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)


if user_question := st.chat_input("Ask a question about your course materials..."):
    with st.chat_message("user"):
        st.markdown(user_question)
    chat_history.add_user_message(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = rag_chain.invoke({
                    "question": user_question,
                    "history": chat_history.messages[:-1],
                })
                answer = result["answer"]
                docs = result["docs"]

                st.markdown(answer)

                if docs:
                    with st.expander(f"📚 Sources ({len(docs)} chunks)"):
                        for i, d in enumerate(docs, 1):
                            source = d.metadata.get("source", "unknown")
                            page_label = d.metadata.get("page_label", "?")
                            st.markdown(f"**{i}. `{source}`, p.{page_label}**")
                            preview = d.page_content[:300]
                            if len(d.page_content) > 300:
                                preview += "…"
                            st.caption(preview)

                chat_history.add_ai_message(answer)

            except Exception as exc:
                st.error(f"Sorry, an error occurred: {exc}")
                logger.exception("RAG chain failed")