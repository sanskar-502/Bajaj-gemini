# LLM-powered Intelligent Query–Retrieval System

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)

An intelligent document analyst assistant designed to process and understand complex legal, insurance, HR, and compliance documents. The system answers user questions based on deep semantic understanding, clause relevance, and AI-generated logical reasoning.

---

## 🎯 System Overview

This system provides a complete Retrieval-Augmented Generation (RAG) pipeline:
- **Multi-LLM Support**: Integrates with both **Google Gemini** and **OpenAI** models.
- **Advanced RAG**: Goes beyond simple Q&A to provide answers with supporting evidence and a traceable reasoning process.
- **High-Performance Vector Search**: Uses **FAISS** (local) or **Pinecone** (cloud) for fast and accurate semantic retrieval.
- **Developer-Friendly API**: Built with **FastAPI**, offering a clean, modern REST API with automatic interactive documentation.

---

## ✨ Features

### Document Processing
-   Supports **PDF, DOCX, TXT, and PPTX** formats, with OCR for image-based PDFs.
-   **Automatic Document Classification** into types like Insurance, Legal, HR, or Compliance.
-   Intelligent text chunking with configurable overlap to preserve context.
-   Heuristic-based extraction of key metadata like company names.

### Query Processing
-   Natural Language Understanding (NLU) to interpret user questions.
-   **AI-Generated Logic Trees** that show the reasoning process behind an answer.
-   Confidence scoring to indicate the reliability of the answer based on the provided documents.

### Vector Storage
-   Flexible backend support for local development (**FAISS**) and scalable cloud deployment (**Pinecone**).
-   Uses powerful open-source **Sentence Transformers** for high-quality text embeddings.
-   Full document lifecycle management via API (upload, list, delete).

---

## 🛠️ Technology Stack

-   **Backend Framework**: FastAPI
-   **LLM Providers**: Google Gemini, OpenAI
-   **Vector Databases**: Pinecone, FAISS
-   **Embeddings**: Sentence Transformers
-   **Document Parsing**: PyMuPDF, python-docx, python-pptx, Tesseract (for OCR)
-   **Data Validation**: Pydantic

---

## 🚀 Quick Start

### Prerequisites
-   Python 3.9+
-   An API key for your chosen LLM (Gemini or OpenAI).
-   (For OCR) **Tesseract-OCR** installed on your system.
-   (Optional) **Pinecone** API key.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd intelligent-query-retrieval-system
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Copy the example file to create your own local configuration.
    ```bash
    cp env_example.txt .env
    ```
    Now, **edit the `.env` file** with your API keys.

4.  **Run the application:**
    ```bash
    python main.py
    ```
    The API will start at `http://127.0.0.1:8000`. Access the interactive documentation at `http://127.0.0.1:8000/docs`.

---

## 📋 Configuration (`.env` file)

Your `.env` file is the central place for configuration.

```bash
# --- Core AI Configuration ---
# Choose your Large Language Model provider: "gemini" or "openai"
LLM_PROVIDER=gemini

# --- Provider API Keys (only one is needed based on the provider above) ---
GEMINI_API_KEY=your_google_ai_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# --- Vector Database Configuration ---
# Choose your vector database: "faiss" (local) or "pinecone" (cloud)
VECTOR_DB_TYPE=faiss

# --- Pinecone Settings (only needed if VECTOR_DB_TYPE="pinecone") ---
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=legal-llm-query

# --- Embedding & Retrieval Settings ---
# A smaller, faster model is recommended for the hackathon to avoid timeouts.
EMBEDDING_MODEL=all-MiniLM-L6-v2
# A larger, more powerful model for general use:
# EMBEDDING_MODEL=intfloat/multilingual-e5-large

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# --- API Server Settings ---
API_HOST=0.0.0.0
API_PORT=8000
🔌 API Endpoints
Visit http://127.0.0.1:8000/docs for a full interactive API specification.

POST /upload: Upload a document for general querying.

POST /query: Ask a question about generally uploaded documents.

GET /documents: List metadata of all documents in the main store.

POST /hackrx/run: Special endpoint for the hackathon submission.

🏆 Hackathon Submission Instructions
To get a high score, you must use the Fast On-the-Fly Processing strategy to avoid timeouts.

Step 1: Optimize Your Configuration
This is the most critical step. The key to making your API fast enough is to use a smaller embedding model. Open your .env file and ensure your EMBEDDING_MODEL is set correctly.

# In your .env file
EMBEDDING_MODEL=all-MiniLM-L6-v2
Step 2: Start Your API Server
With the faster model configured, start your main application.



python main.py
Step 3: Expose Your Server with ngrok
Open a new terminal and run ngrok to create a public URL that tunnels to your local server.


ngrok http 8000
ngrok will give you a public https forwarding URL, for example: https://<random-string>.ngrok-free.app

Step 4: Construct and Submit Your Webhook URL
Combine the ngrok URL with the required endpoint path (/hackrx/run).

Example Final URL:

https://<random-string>.ngrok-free.app/hackrx/run
Copy this full URL and paste it into the "Webhook URL" field on the submission page.

🚨 Troubleshooting
Configuration Error on Startup: The most common issue. The app will stop with a clear error message (e.g., ValueError: GEMINI_API_KEY is missing...). Double-check your .env file and ensure the required keys for your selected LLM_PROVIDER and VECTOR_DB_TYPE are present.

ModuleNotFoundError: Ensure you have installed all dependencies using pip install -r requirements.txt.

PDF Processing Fails / OCR Errors: Make sure you have installed the Tesseract-OCR application on your computer, not just the Python library.

Check app.log: For any other issues, the app.log file will contain detailed error messages.