import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

class LegalRetriever:
    def __init__(self, collection="legalsathi_baseline"):
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        self.db_path = os.path.join(project_root, "data", "vector_db")
        
        # GPU Check for MNC-grade speed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔌 Connecting Database at: {self.db_path} (Device: {self.device.upper()})")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': self.device}
        )

        try:
            self.client = QdrantClient(path=self.db_path, prefer_grpc=False)
            if not self.client.collection_exists(collection):
                raise ValueError(f"Collection '{collection}' not found. Run ingest.py first.")

            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=collection,
                embedding=self.embeddings
            )
        except Exception as e:
            print(f"🚨 DB Error: {e}")
            raise e

    def search(self, query, k=5):
        return self.vectorstore.similarity_search(query, k=k)