# 📘 Development Plan: Basic Question Answering App

**Knowledge Base:** `Articles.pdf`\
**Frontend:** Streamlit\
**Backend:** Python

------------------------------------------------------------------------

## 1️⃣ Project Overview

Build a simple Question Answering (QA) application that allows users to:

-   Use pre-loaded `Articles.pdf`
-   Ask questions related to the document
-   Receive contextual answers based only on the PDF content

The app will use:

-   PDF text extraction
-   Text chunking
-   Embeddings
-   Vector similarity search
-   LLM-based answer generation

------------------------------------------------------------------------

## 2️⃣ Architecture Overview

User (Streamlit UI)\
↓\
User Question\
↓\
Backend (Python)\
↓\
Vector Search (Relevant Chunks)\
↓\
LLM (Answer Generation)\
↓\
Streamlit UI (Answer Display)

------------------------------------------------------------------------

## 3️⃣ Technology Stack

  Layer          Tool
  -------------- ----------------------------------------
  Frontend       Streamlit
  Backend        Python
  PDF Parsing    PyPDF
  Embeddings     SentenceTransformers 
  Vector Store   FAISS
  LLM            Hugging facemodel

------------------------------------------------------------------------

# 🚀 Phase 1: Environment Setup

## Install Dependencies

pip install streamlit\
pip install pypdf\
pip install faiss-cpu\
pip install sentence-transformers\


------------------------------------------------------------------------

# 📄 Phase 2: PDF Processing

### Steps

1.  Load `Articles.pdf`
2.  Extract text
3.  Clean text
4.  Split into chunks (500--1000 chars, 100 overlap)

------------------------------------------------------------------------

# 🧠 Phase 3: Create Embeddings

1.  Convert text chunks into embeddings
2.  Store embeddings in FAISS
3.  Save vector index locally

Outputs: - index.faiss\
- metadata.pkl

------------------------------------------------------------------------

# 🔍 Phase 4: Question Processing Flow

1.  Convert question → embedding\
2.  Perform similarity search\
3.  Retrieve top-k relevant chunks\
4.  Send context + question to LLM\
5.  Generate answer

------------------------------------------------------------------------

# 📝 Prompt Template

You are a helpful assistant.\
Answer ONLY using the context provided.

Context:\
{retrieved_chunks}

Question:\
{user_question}

Answer:

------------------------------------------------------------------------

# 🎨 Phase 5: Streamlit Frontend

-   Title
-   Question input box
-   Submit button
-   Answer display

Run app:

streamlit run app.py

------------------------------------------------------------------------

# 🔐 Phase 6: Guardrails

-   If similarity score is low → "Answer not found in document."
-   Prevent hallucination using strict context-based prompting

------------------------------------------------------------------------

# 🎯 Success Criteria

✔ Answers grounded in PDF\
✔ No hallucinations\
✔ Fast response 
✔ Clean UI

------------------------------------------------------------------------

