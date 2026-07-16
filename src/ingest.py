import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Global Config
RAW_DATA_DIR = "data/raw/"
VECTOR_DB_PATH = "data/vector_db"
COLLECTION_NAME = "legalsathi_baseline"

def run_ingestion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Ingestion Engine: Using {device}")

    # 1. Faster Embeddings Initialization
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}
    )

    # 2. Optimized Loading
    all_docs = []
    # Smaller chunks = Faster retrieval + more precision
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

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
                    print(f"✅ Processed: {file}")
                except Exception as e:
                    print(f"❌ Skip {file}: {e}")

    # 3. Vector Storage
    print("📦 Committing to Database...")
    QdrantVectorStore.from_documents(
        all_docs,
        embeddings,
        path=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME,
        force_recreate=True
    )
    print("🎯 Success: Knowledge base is live.")

if __name__ == "__main__":
    run_ingestion()