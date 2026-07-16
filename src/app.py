import os
import torch
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. Initialization
load_dotenv()

# Constants
VECTOR_DB_PATH = "data/vector_db"
COLLECTION_NAME = "legalsathi_baseline"

def run_baseline_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  System Status: Running on {device}")

    # 2. Embedding Layer
    embeddings_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}
    )

    # 3. Connection Layer
    client = QdrantClient(path=VECTOR_DB_PATH)
    if not client.collection_exists(COLLECTION_NAME):
        print(f"❌ Error: Collection '{COLLECTION_NAME}' not found.")
        return

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings_model,
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. LLM Layer (Updated Model String)
    print("🤖 Initializing LLM: Llama-3.3-70b-versatile")
    
    # We pass the key explicitly to avoid environment-read red lines
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0, 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 5. Prompt Layer
    template = """You are LegalSathi AI, an expert Indian Legal Assistant. 
    Use ONLY the context below to answer the user's question. 
    If you don't know the answer, state that it is not in the knowledge base.
    
    CRITICAL: You must cite the 'Source' file for your answer.

    Context: {context}
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 6. Chain Construction
    def format_docs(docs):
        return "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Execution Logic
    test_queries = [
        # Pecuniary/Fees
        "What is the fee for filing a consumer case for a product worth 4 Lakhs?",
        "What is the pecuniary jurisdiction of the District Commission under the 2019 Act?",
        
        # RTI Timelines & Fees
        "Within how many days should a PIO respond to an RTI application?",
        "What is the response time for an RTI if it concerns the life or liberty of a person?",
        "Does a person below the poverty line (BPL) have to pay any fee for an RTI?",
        
        # E-commerce & Grievance
        "What are the requirements for an e-commerce entity regarding a grievance officer?",
        "Is an e-commerce entity required to provide a refund if a product is defective?",
        
        # Rights & Definitions
        "How is a 'consumer' defined under the Consumer Protection Act 2019?",
        "What are the six consumer rights defined in the 2019 Act?",
        
        # Appeals & Procedures
        "Within how many days must a First Appeal be filed against an RTI decision?",
        "What is the time limit to file an appeal in the State Commission against a District Commission order?",
        "Can a consumer dispute be referred to mediation? If so, at what stage?",
        
        # CCPA & Penalties
        "What is the role of the Central Consumer Protection Authority (CCPA)?",
        "What is the penalty for manufacturers for a misleading advertisement?",
        "What is the definition of 'product liability' under the new Act?"
    ]

    print("\n🚀 --- STARTING BASELINE EVALUATION ---\n")
    for query in test_queries:
        print(f"❓ Query: {query}")
        try:
            # We use .invoke() for single-turn logic
            response = rag_chain.invoke(query)
            print(f"💡 AI Response: {response}")
        except Exception as e:
            print(f"❌ Execution Error: {str(e)}")
        print("-" * 60)

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found. Check your .env file.")
    else:
        run_baseline_eval()