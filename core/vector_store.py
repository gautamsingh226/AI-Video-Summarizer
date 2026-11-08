
import os

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# 1. Load Hugging Face Embedding Model
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )


# 2. Build Vector Store
def build_vector_store(transcript: str) -> Chroma:

    print("Building Vector Store...")

    # Split transcript into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(transcript)

    # Convert chunks into Documents
    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    # Generate embeddings
    embeddings = get_embeddings()

    # Store documents in ChromaDB
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    print(f"Vector Store created with {len(docs)} chunks!")

    return vector_store


# 3. Load Existing Vector Store
def load_vector_store() -> Chroma:

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store


# 4. Create Retriever
def get_retriever(vector_store: Chroma, k: int = 4):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
