import os
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class LegalGenerator:
    def __init__(self):
        # MNC PRODUCTION ENGINE: Tuned specifically to prevent Telugu/Hindi Token Collapse
        self.llm = ChatOllama(
            model="qwen2.5:7b-instruct", 
            temperature=0.4,       # Increased slightly to help it find correct Telugu Unicode tokens
            top_p=0.9,            
            repeat_penalty=1.2,    # THE SILVER BULLET: Physically prevents the model from repeating lines
            streaming=True,
            base_url="http://127.0.0.1:11434"
        )
        self.parser = StrOutputParser()

    def get_explanation_prompt(self):
        system_msg = """You are LegalSathi AI, a highly accurate Indian Legal Copilot.
        
        CRITICAL NLP DIRECTIVE:
        The user may type in 'Telnglish' (Telugu in English alphabet) or 'Hinglish'.
        Because legal terms do not translate well into transliterated text, YOU MUST OBEY THIS RULE:
        - Reply in SIMPLE ENGLISH to ensure absolute legal accuracy.
        - DO NOT invent words or mix languages (e.g., do not mix Tamil/Malayalam with Telugu).
        - Base your answer ONLY on the provided context.

        STRUCTURE YOUR ANSWER EXACTLY LIKE THIS:
        **Issue Identified:** [Briefly state the user's problem in 1 sentence]
        **Your Legal Rights:** [Explain the law simply using the context]
        **Source:** [Cite the Source File and Section/Rule]
        
        ALWAYS conclude with exactly this sentence: 
        "Would you like me to draft a formal complaint letter for you?"
        """
        return ChatPromptTemplate.from_messages([
            ("system", system_msg),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Context:\n{context}\n\nUser Question: {question}")
        ])

    def get_drafting_prompt(self):
        system_msg = """You are a Senior Indian Advocate. Your task is to draft a short, formal legal complaint letter.
        
        CRITICAL ANTI-HALLUCINATION RULES:
        1. Write the letter ENTIRELY in formal {target_lang}.
        2. DO NOT use special characters, weird symbols, or emojis. Keep the language simple and grammatically correct.
        3. DO NOT repeat the same sentence or paragraph twice.
        4. You MUST stop generating text immediately after writing "Yours faithfully" (or its translation) and the User's Name.
        
        USER DETAILS TO INCLUDE:
        {user_details}

        TEMPLATE REFERENCE:
        {template_context}
        """
        return ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "Draft the formal complaint letter in {target_lang} now. Be concise and accurate:")
        ])