# LegalSathi AI: System Design Specification

## 1. Executive Summary

Despite sweeping technological advancements in India, a critical access-to-justice gap remains. With over 45 million pending cases across the judicial system and widespread legal illiteracy among ordinary citizens, navigating statutory rights—specifically concerning Consumer Protection and Right to Information (RTI)—is prohibitively complex. 

LegalSathi AI is conceptualized not as a general-purpose conversational "chatbot," but as an intelligent **Workflow Copilot**. By bridging the gap between archaic legal codifications and layman understanding, LegalSathi empowers users to proactively assert their rights. It shifts the paradigm from simple question-answering to automated, legally-grounded task execution, specifically drafting precision-targeted legal notices, consumer complaints, and RTI appeals backed by official government statutes.

## 2. Problem Statement

### 2.1 The Complexity Gap
The primary barrier to legal recourse is the cognitive overhead required to parse legislative texts. Ordinary citizens face a "Complexity Gap" wherein the archaic syntax, nested clauses, and cross-referenced schedules of official documents (e.g., the Consumer Protection Act, 2019) are effectively unintelligible without professional legal counsel.

### 2.2 The Citation Gap
Existing Large Language Models (LLMs) such as ChatGPT or Claude fail in the legal domain due to the "Citation Gap." General-purpose LLMs rely on parametric memory, often hallucinating legal facts, applying foreign jurisprudence, or providing generalized advice without verifiable sources. For a system to be trusted in a legal context, it must guarantee strict traceability to official, sovereign-backed documentation. Any advice generated must explicitly cite the governing Act, Section, and Subsection.

## 3. Data Engineering & Knowledge Corpus

To enforce domain specificity and eliminate hallucination, LegalSathi AI restricts its knowledge space strictly to a curated, tiered corpus of official Government of India documentation.

### 3.1 Knowledge Base Hierarchy & Curated Corpus
The corpus encompasses 20+ strictly verified documents, organized hierarchically:

**Tier 1: Statutory Acts (The Core Law)**
* `act_cp_2019_en.pdf`, `act_cp_2019_hi.pdf`
* `act_rti_2005_en.pdf`, `act_rti_2005_hi.pdf`, `act_rti_2005_te.pdf`

**Tier 2: Procedural Rules (The Implementation Mechanics)**
* `rule_cp_central_en_hi.pdf`
* `rule_cp_commission_general_en.pdf`
* `rule_cp_ecommerce_amend_2021_en.pdf`
* `rule_cp_ecommerce_sellers_2020_en.pdf`
* `rule_cp_ecommerce_en_hi.pdf`
* `rule_cp_mediation_en_hi.pdf`
* `rule_rti_2012_en.pdf`, `rule_rti_2012_hi.pdf`
* `rule_cp_jurisdiction_2021_en.pdf`

**Tier 3: Practical Guides (Citizen Knowledge Base)**
* `hb_cp_govt_handbook_en.pdf`
* `faq_cp_metrology_en.pdf`

**Tier 4: Document Templates (Generation Scaffolding)**
* `tmp_cp_complaint_en.pdf`
* `tmp_rti_application_en.pdf`
* `tmp_rti_appeal_1_en.pdf`
* `tmp_rti_appeal_2_en.pdf`

### 3.2 Naming Convention
A strict metadata-embedded file naming convention is enforced for programmatic routing and metadata extraction during ingestion:
`{category}_{subject}_{year/modifier}_{language}.pdf` (e.g., `rule_cp_jurisdiction_2021_en.pdf`).

### 3.3 Semantic Section-Based Chunking
Standard fixed-length character splitting (e.g., 1000 tokens) fragments legal context, often severing a legal condition from its prerequisite. LegalSathi AI employs **Semantic Section-based splitting**. The ingestion engine parses documents hierarchically (Chapter -> Section -> Subsection), ensuring that atomic chunks preserve the complete legal thought and localized context required for accurate reasoning.

## 4. System Architecture

The architecture represents a state-of-the-art Retrieval-Augmented Generation (RAG) pipeline designed for high-precision legal query resolution.

### 4.1 Ingestion Layer
* **Layout-Aware Parsing:** Raw PDFs are processed using a layout-aware parsing engine that intelligently handles multi-column layouts, nested legal schedules, and tabular data, converting them into structured markdown before chunking.

### 4.2 Retrieval Layer
* **Embedding Model:** `BAAI/bge-m3` is utilized for robust, multilingual dense embeddings.
* **Vector Store:** `Qdrant` serves as the high-performance vector database.
* **Hybrid Search Strategy:** To capture both semantic intent ("How do I get my money back?") and exact legal terminology ("Section 2(11) defective goods"), the system utilizes Hybrid Search, merging Dense Semantic Vector retrieval with BM25 Keyword Search.
* **Cross-Encoder Reranking:** The initial retrieved candidate chunks (Top-K) are passed through a Cross-Encoder Reranker to recalculate relevance scores against the user query, maximizing Context Precision.
* **Metadata Filtering:** Queries are programmatically routed using metadata filters (e.g., restricting the search space to `document_type="Act"` and `domain="RTI"`).
* **Multilingual Strategy:** The system executes cross-lingual retrieval. Queries in regional languages (e.g., Telugu, Hindi) are embedded in a unified vector space. English Statutes act as the "Primary Source of Truth." The LLM retrieves the English statute, reasons over it, and translates the final verifiable advice back into the user's native language.

### 4.3 Reasoning Layer
* **LLM Engine:** `Qwen-2.5-7B (Instruct)` is deployed as the core reasoning engine.
* **Chain-of-Thought (CoT) Prompting:** The inference pipeline mandates a CoT approach. The model is instructed to first extract facts from the user query, map those facts against the retrieved legal sections, and explicitly output the logical deduction before formulating the final answer or drafting a document.
* **Deterministic Citations:** The prompt enforces strict citation injection, appending `[Source: {act_name}, Section {section_no}]` to every generated claim.

## 5. Evaluation Framework

To guarantee production readiness and regulatory safety, the system is subjected to rigorous, automated evaluation.

### 5.1 The "Golden 15" Ground Truth Set
A curated benchmark dataset of 15 highly complex, edge-case citizen queries paired with human-expert validated legal answers and document drafts. All architecture iterations are evaluated against this baseline.

### 5.2 RAGAS Metrics
The system is quantitatively measured using the RAGAS framework:
* **Faithfulness:** Ensuring the generated answer is strictly derivable from the retrieved context (Target: >0.95).
* **Answer Relevancy:** Measuring how well the answer directly addresses the user's prompt (Target: >0.90).
* **Context Precision:** Evaluating whether the relevant legal chunks were ranked highest in the retrieval pipeline (Target: >0.85).

### 5.3 Hallucination Control
The model operates under **"Strict Context Adherence."** The system prompt contains explicit override instructions: if the retrieved context does not contain sufficient information to address the query, the LLM is hardcoded to respond with a localized variation of *"I don't know based on the provided documents,"* preventing fabricated jurisprudence.

## 6. Deployment & Scale

The system architecture is decoupled to allow independent scaling of the reasoning and application layers.
* **Backend:** A high-concurrency `FastAPI` application serves as the orchestration layer, managing the ingestion pipeline, Qdrant interactions, and LLM inference.
* **Frontend:** A responsive `React`-based frontend provides the citizen interface, handling document uploads, rendering conversational UI, and displaying parsed OCR text and generated legal drafts.
