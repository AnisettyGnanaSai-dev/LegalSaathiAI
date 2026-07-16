from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class LegalGenerator:
    def __init__(self):
        # MNC Standard: Explicitly defining local loopback to avoid DNS latency
        self.llm = ChatOllama(
            model="qwen2.5:3b-instruct", 
            temperature=0.1, 
            streaming=True,
            base_url="http://127.0.0.1:11434"
        )
        self.parser = StrOutputParser()

    def get_explanation_prompt(self, context, question):
        system_msg = (
            "You are LegalSathi AI, a professional Legal Copilot. "
            "Explain laws clearly in the user's language (Hinglish/Telnglish/Native). "
            "Always cite the source PDF at the end of each point. "
            "If you find a conflict in context (e.g. jurisdiction), point it out. "
            "End by asking: 'Would you like me to draft a legal complaint letter for you?'"
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", f"Context: {context}\n\nUser Question: {question}")
        ])

    def get_drafting_prompt(self, details, template_context, language):
        return ChatPromptTemplate.from_messages([
            ("system", f"You are a Senior Indian Advocate. Draft a professional legal complaint in {language}. "
                       "Strictly follow the template structure provided. Insert these user details: {details}"),
            ("human", f"Template Context: {template_context}")
        ])