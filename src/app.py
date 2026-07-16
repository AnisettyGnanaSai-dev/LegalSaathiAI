import os
import requests
from operator import itemgetter
from retriever import LegalRetriever
from generator import LegalGenerator
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

OLLAMA_URL = "http://127.0.0.1:11434"

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def main():
    print("="*50)
    print("⚖️  LegalSathi AI: Production Interactive Engine")
    print("="*50)
    
    try:
        # Liveness Probe
        requests.get(OLLAMA_URL, timeout=3)
        retriever = LegalRetriever()
        generator = LegalGenerator()
    except Exception as e:
        print(f"❌ Error: Engine Offline. Ensure Ollama is running. ({e})")
        return

    def format_docs(docs):
        return "\n\n".join([f"Source: {d.metadata.get('source', 'N/A')}\nContent: {d.page_content}" for d in docs])

    # 1. Primary Explanation Chain
    explain_chain = (
        {
            "context": itemgetter("question") | retriever.vectorstore.as_retriever(search_kwargs={"k": 3}) | format_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history")
        }
        | generator.get_explanation_prompt()
        | generator.llm
        | generator.parser
    )

    app_with_history = RunnableWithMessageHistory(
        explain_chain, 
        get_session_history, 
        input_messages_key="question", 
        history_messages_key="history"
    )

    # 2. Secondary Drafting Chain
    draft_chain = generator.get_drafting_prompt() | generator.llm | generator.parser

    sid = "user_prod_session"
    print("✅ System Ready. (Type 'exit' to quit)")

    while True:
        u_input = input("\n👤 User: ")
        if u_input.lower() in ['exit', 'quit']: break
        
        print("\n🔍 Analyzing statutes...")
        print("🤖 AI: ", end="", flush=True)
        
        full_reply = ""
        try:
            for chunk in app_with_history.stream({"question": u_input}, config={"configurable": {"session_id": sid}}):
                print(chunk, end="", flush=True)
                full_reply += chunk
        except Exception as e:
            print(f"\n❌ Inference Error: {e}")
            continue
        print("\n")

        # INTERACTIVE DRAFTING TRIGGER
        if "draft" in full_reply.lower() or "letter" in full_reply.lower() or "complaint" in full_reply.lower():
            decision = input("👉 System: Draft this letter for you? (yes/no): ")
            if decision.lower() == 'yes':
                target_lang = input("🌐 Select Language (English/Telugu/Hindi): ")
                user_details = input("📝 Provide Details (Name, Order ID, Company Name): ")
                
                print(f"\n📄 Drafting in {target_lang}...")
                
                # Retrieve template
                t_docs = retriever.search("complaint proforma template", k=1)
                t_context = format_docs(t_docs) if t_docs else "Standard Legal Format"
                
                # Execute drafting chain
                for chunk in draft_chain.stream({
                    "target_lang": target_lang,
                    "template_context": t_context,
                    "user_details": user_details
                }):
                    print(chunk, end="", flush=True)
                print("\n" + "="*50)
            else:
                print("👍 System: Understood. Let me know if you need anything else.")

if __name__ == "__main__":
    main()