import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Configuration
RAW_DATA_DIR = "data/raw/"
VECTOR_DB_PATH = "data/vector_db"
COLLECTION_NAME = "legalsathi_baseline"

def run_ingestion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    # 1. Initialize modern HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}
    )

    # 2. Load PDFs
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    print("📂 Loading PDFs from data/raw/...")
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Error: {RAW_DATA_DIR} not found!")
        return

    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)
                try:
                    loader = PyPDFLoader(path)
                    data = loader.load()
                    for doc in data:
                        doc.metadata["source"] = file
                    
                    chunks = text_splitter.split_documents(data)
                    all_docs.extend(chunks)
                    print(f"✅ Chunked: {file}")
                except Exception as e:
                    print(f"❌ Error in {file}: {e}")

    if not all_docs:
        print("⚠️ No documents found. Check your data/raw folder.")
        return

    # 3. Store in Qdrant (Local Persistent Mode)
    print(f"📦 Building Vector DB at {VECTOR_DB_PATH}...")
    
    vectorstore = QdrantVectorStore.from_documents(
        all_docs,
        embeddings,
        path=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME,
        force_recreate=True # Ensures a clean baseline
    )
    
    print("🎯 Success! Database built.")

if __name__ == "__main__":
    run_ingestion()