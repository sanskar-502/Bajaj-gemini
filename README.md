# 🧠 LLM-powered Intelligent Query–Retrieval System

An intelligent document analyst assistant designed to process and understand complex legal, insurance, HR, and compliance documents. The system answers user questions based on deep semantic understanding, clause relevance, and AI-generated logical reasoning.

---

## 🎯 System Overview

This system provides a complete Retrieval-Augmented Generation (RAG) pipeline:

- **Multi-LLM Support**: Integrates with both Google Gemini and OpenAI models.  
- **Advanced RAG**: Goes beyond simple Q&A to provide answers with supporting evidence and a traceable reasoning process.  
- **High-Performance Vector Search**: Uses FAISS (local) or Pinecone (cloud) for fast and accurate semantic retrieval.  
- **Developer-Friendly API**: Built with FastAPI, offering a clean, modern REST API with automatic interactive documentation.

---

## ✨ Features

- **Robust Document Processing**: Supports PDF, DOCX, TXT, and PPTX formats.
- **Advanced Text Extraction**: Includes OCR to extract text even from scanned, image-based PDFs.
- **Intelligent Chunking**: Splits documents into meaningful, overlapping chunks to preserve context.
- **Multi-Provider Flexibility**: Easily switch between different LLMs and vector databases via configuration.
- **AI-Generated Reasoning**: Provides a `logic_tree` in responses to show how it arrived at an answer.
- **Full API Control**: Manage the entire document lifecycle (upload, query, list, delete) through REST API endpoints.

---

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI  
- **LLM Providers**: Google Gemini, OpenAI  
- **Vector Databases**: Pinecone, FAISS  
- **Embeddings**: Sentence Transformers  
- **Document Parsing**: PyMuPDF, python-docx, python-pptx  
- **OCR Engine**: Tesseract & Poppler  
- **Data Validation**: Pydantic  

---

## 🚀 Getting Started

### Step 1: System-Level Dependencies

#### 🧠 Tesseract OCR

- **Windows**: Download from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Make sure to **add to system PATH** during installation.  
- **macOS**:
  ```bash
  brew install tesseract
  ```
- **Ubuntu/Debian**:
  ```bash
  sudo apt-get update && sudo apt-get install tesseract-ocr
  ```

#### 📄 Poppler

- **Windows**: Download and extract from Poppler for Windows. Add the `bin/` folder to your system's `PATH`.
- **macOS**:
  ```bash
  brew install poppler
  ```
- **Ubuntu/Debian**:
  ```bash
  sudo apt-get install poppler-utils
  ```

---

### Step 2: Clone the Repository

```bash
# Clone the repository
git clone <repository-url>
cd intelligent-query-retrieval-system
```

### **Step 3: Set Up Python Environment**

# Create and activate a virtual environment
```bash
python -m venv venv
```


# On Windows
```bash
.\venv\Scripts\Activate.ps1
```

### **Step 4: Install Dependencies**

# Install required Python packages
```bash
pip install -r requirements.txt
```

### **Step 5: Configure Environment Variables**

# Create a .env file by copying the example file.

```bash
cp env_example.txt .env
```
# Now, edit the .env file with your API keys and preferred configurations.

## ⚙️ Configuration (.env)

Variable	Description	Example Values
LLM_PROVIDER	The main switch for your Language Model	gemini or openai
GEMINI_API_KEY	Your secret API key for Google Gemini	AIzaSy...
OPENAI_API_KEY	Your secret API key for OpenAI	sk-...
VECTOR_DB_TYPE	The vector database to use	faiss or pinecone
EMBEDDING_MODEL	The model used for text embeddings	all-MiniLM-L6-v2, intfloat/multilingual-e5-large
SIMILARITY_THRESHOLD	Minimum similarity score for relevance


▶️ Running the Application
Once configured, start the application with a single command.

python main.py
You can access the application at:

API: http://127.0.0.1:8000

Docs: http://127.0.0.1:8000/docs

⚡ Performance Tuning
Adjust the embedding model based on your needs for speed vs. accuracy.

For Speed (Hackathons/CPU):
Set this in your .env file for faster processing on less powerful hardware.


EMBEDDING_MODEL=all-MiniLM-L6-v2
For Accuracy (Production/GPU):
Set this in your .env file for higher quality results, especially with a GPU.



EMBEDDING_MODEL=intfloat/multilingual-e5-large
🏆 Hackathon Submission Instructions
Step 1: Optimize Configuration
For the hackathon, ensure your .env file is set for speed:


EMBEDDING_MODEL=all-MiniLM-L6-v2
Step 2: Start the Server


python main.py
Step 3: Expose Your Local Server with Ngrok


ngrok http 8000
Ngrok will generate a public URL for your local server, like:
https://<random-string>.ngrok-free.app

Step 4: Submit Your Webhook URL
The final URL you need to submit will be your Ngrok URL followed by the API endpoint:
https://<random-string>.ngrok-free.app/hackrx/run

🚨 Troubleshooting
Startup Errors: Double-check that your .env file is correctly named and that your API keys are valid.

Missing Modules: Make sure your virtual environment is activated and run pip install -r requirements.txt again.

OCR/Poppler Errors: Confirm that Tesseract and Poppler are installed correctly and that their locations are added to your system's PATH.

Other Issues: Check the app.log file for detailed error messages and debug logs.

📄 License & Docs
License: This project is licensed under the MIT License.

API Docs: Full API documentation is available at the /docs endpoint when the application is running.

🚀 Happy Hacking!
