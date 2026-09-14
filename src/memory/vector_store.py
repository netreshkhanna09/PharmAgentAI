# ==============================================================
# Vector Store Manager
# ==============================================================
# PURPOSE: Builds and manages the FAISS vector store containing
#          chunked, embedded FDA guideline documents.
#
# TWO-PHASE RAG ARCHITECTURE:
#
# Phase 1 — INDEXING (runs once at startup or when docs change):
#   FDA text files --> Chunker --> Embedder --> FAISS index saved to disk
#
# Phase 2 — RETRIEVAL (runs every compliance check):
#   Claim text --> Embed --> FAISS similarity search --> Top-K chunks
#
# WHY sentence-transformers (all-MiniLM-L6-v2)?
#   - Runs completely LOCAL -- no API key, no cost, no data leaving machine
#   - 384-dimensional embeddings -- fast search, small storage
#   - Good quality for domain-specific regulatory text
#   - Downloads once (~90MB), then runs from cache
#
# WHY FAISS over Pinecone/ChromaDB?
#   - Zero infrastructure -- a single .pkl file
#   - Sub-millisecond search on small datasets (our FDA docs ~ 50 chunks)
#   - Works in air-gapped pharmaceutical environments
#   - Free forever
# ==============================================================

import os
import pickle
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ── Paths ───────────────────────────────────────────────────────
GUIDELINES_DIR = Path("data/fda_guidelines")
FAISS_INDEX_DIR = Path("data/faiss_index")


# ── Embedding Model ─────────────────────────────────────────────
# all-MiniLM-L6-v2: 384 dimensions, fast, free, local
# Downloads to ~/.cache/huggingface/ on first run (~90MB)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Enables cosine similarity
    )


# ── Phase 1: Load and Chunk FDA Documents ──────────────────────
def load_fda_documents() -> List[Document]:
    """
    Loads all .txt files from data/fda_guidelines/ and splits
    them into chunks using RecursiveCharacterTextSplitter.

    CHUNKING STRATEGY: RecursiveCharacterTextSplitter
    - Tries to split on: paragraphs -> sentences -> words -> chars
    - Keeps semantically meaningful units together
    - chunk_size=600: sweet spot for regulatory text
    - chunk_overlap=60: prevents losing context at chunk boundaries

    Returns:
        List of LangChain Document objects (text + metadata)
    """
    if not GUIDELINES_DIR.exists():
        raise FileNotFoundError(f"Guidelines directory not found: {GUIDELINES_DIR}")

    # Text splitter with overlap to preserve context at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,        # ~450 tokens -- safe for any LLM context
        chunk_overlap=60,      # 10% overlap -- prevents boundary context loss
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    txt_files = list(GUIDELINES_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {GUIDELINES_DIR}")

    for filepath in sorted(txt_files):
        text = filepath.read_text(encoding="utf-8")

        # Split into chunks
        chunks = splitter.split_text(text)

        # Wrap each chunk in a LangChain Document with metadata
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": filepath.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            ))

    print(f"[VectorStore] Loaded {len(txt_files)} FDA guideline files")
    print(f"[VectorStore] Created {len(documents)} chunks")
    return documents


# ── Phase 1: Build and Save FAISS Index ────────────────────────
def build_vector_store() -> FAISS:
    """
    Builds the FAISS vector store from FDA guideline documents.
    Saves the index to disk so it only needs to be built once.

    HOW IT WORKS:
    1. Load FDA documents from text files
    2. Split into 600-char chunks with 60-char overlap
    3. Embed each chunk with all-MiniLM-L6-v2 (384 dimensions)
    4. Store embeddings + text in FAISS index
    5. Save index to data/faiss_index/ for reuse

    Returns:
        FAISS vector store object
    """
    print("[VectorStore] Building FAISS index from FDA guidelines...")
    print("[VectorStore] (Downloading embedding model on first run -- ~90MB)")

    documents = load_fda_documents()
    embeddings = get_embeddings()

    # Build FAISS index from documents
    # FAISS.from_documents() embeds all chunks and builds the index
    vector_store = FAISS.from_documents(documents, embeddings)

    # Save to disk (so we don't rebuild every run)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_INDEX_DIR))

    print(f"[VectorStore] Index saved to {FAISS_INDEX_DIR}/")
    print(f"[VectorStore] Indexed {len(documents)} chunks from FDA guidelines")

    return vector_store


# ── Phase 2: Load Existing Index ───────────────────────────────
def load_vector_store() -> FAISS:
    """
    Loads the FAISS index from disk.
    Falls back to building it if not found.
    """
    if FAISS_INDEX_DIR.exists() and any(FAISS_INDEX_DIR.iterdir()):
        print("[VectorStore] Loading existing FAISS index from disk...")
        embeddings = get_embeddings()
        vector_store = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,  # Safe -- our own files
        )
        print("[VectorStore] Index loaded successfully")
        return vector_store
    else:
        print("[VectorStore] No existing index found -- building now...")
        return build_vector_store()


# ── Phase 2: Retrieve Relevant FDA Rules ───────────────────────
def retrieve_relevant_rules(
    claim_text: str,
    vector_store: FAISS,
    top_k: int = 3,
) -> List[Tuple[str, str, float]]:
    """
    Retrieves the most relevant FDA guideline chunks for a given claim.

    Uses cosine similarity (because normalize_embeddings=True above).

    WHY top_k=3?
    - More than 3 chunks risks "lost in the middle" problem
    - 3 chunks ~ 1800 chars -- stays within token budget
    - For FDA compliance, precision matters more than recall

    Args:
        claim_text: The pharmaceutical claim to check
        vector_store: Loaded FAISS index
        top_k: Number of chunks to retrieve

    Returns:
        List of (chunk_text, source_file, similarity_score) tuples
    """
    # similarity_search_with_score returns (Document, score) pairs
    # Score is cosine distance (lower = more similar when not normalized)
    # Since we use normalize_embeddings=True, higher score = more similar
    results = vector_store.similarity_search_with_score(claim_text, k=top_k)

    retrieved = []
    for doc, score in results:
        retrieved.append((
            doc.page_content,
            doc.metadata.get("source", "unknown"),
            float(score),
        ))

    return retrieved


# ── Singleton Vector Store ──────────────────────────────────────
# Loaded once at module import time.
# All agents share this single instance -- no redundant loading.
print("[VectorStore] Initializing...")
_vector_store = load_vector_store()


def get_vector_store() -> FAISS:
    """Returns the shared vector store singleton."""
    return _vector_store
