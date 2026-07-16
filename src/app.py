import os
import torch
import requests
import time
from operator import itemgetter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# MNC Constants
VECTOR_DB_PATH = os.path.join(os.getcwd(), "data", "vector_db")
COLLECTION_NAME = "legalsathi_baseline"
OLLAMA_URL = "http://127.0.0.1:11434"

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def get_rag_chain():
    # Force CPU and clean cache
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'}
    )
    
    # Use prefer_grpc=False for higher stability in low-ram environments
    client = QdrantClient(path=VECTOR_DB_PATH, prefer_grpc=False)
    vectorstore = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embedding=embeddings
    )
    
    # MNC FIX: Reduce K to 3. This saves 50% of RAM during inference
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(
        model="qwen2.5:3b-instruct", 
        temperature=0.1, 
        streaming=True, 
        base_url=OLLAMA_URL,
        num_ctx=2048 # Limit context window to prevent Segfault
    )

    system_instr = (
        "You are LegalSathi AI. 1. Explain law in user's language. 2. Ask for details. 3. Draft letter. "
        "Cite Source and Section. ALWAYS ask if the user wants to draft a letter."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instr),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])

    def format_docs(docs):
        return "\n\n".join([f"Source: {d.metadata.get('source')}\nContent: {d.page_content}" for d in docs])

    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(chain, get_session_history, input_messages_key="question", history_messages_key="history")

def main():
    print("⚖️ LegalSathi AI: Initializing Optimized CPU Engine...")
    try:
        requests.get(OLLAMA_URL, timeout=5)
        chain = get_rag_chain()
    except:
        print("❌ Error: Restart the server using Step 1.")
        return

    sid = "prod_session_01"
    print("✅ System Ready (Memory Optimized).")

    while True:
        u_input = input("\n👤 User: ")
        if u_input.lower() in ['exit', 'quit']: break
        
        print("\n🔍 Analysing context...")
        print("🤖 AI: ", end="", flush=True)
        
        full_reply = ""
        try:
            for chunk in chain.stream({"question": u_input}, config={"configurable": {"session_id": sid}}):
                print(chunk, end="", flush=True)
                full_reply += chunk
        except Exception as e:
            print(f"\n❌ [Inference Error]: {e}")
            continue

        if "draft" in full_reply.lower() or "letter" in full_reply.lower():
            decision = input("\n👉 Draft a letter? (yes/no): ")
            if decision.lower() == 'yes':
                lang = input("🌐 Language (English/Telugu/Hindi): ")
                details = input("📝 Provide details (Name, Order ID): ")
                
                print(f"\n📄 Generating {lang} draft...\n")
                draft_prompt = f"Draft a formal legal complaint in {lang} script. User details: {details}. Use professional format."
                
                for chunk in chain.stream({"question": draft_prompt}, config={"configurable": {"session_id": sid}}):
                    print(chunk, end="", flush=True)
                print("\n" + "="*50)

if __name__ == "__main__":
    main()