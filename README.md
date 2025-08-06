# LLM-powered Intelligent Query–Retrieval System

An intelligent legal-insurance-HR-compliance document analyst assistant that processes lengthy documents, understands their legal/compliance structure, and answers user questions based on deep semantic understanding, clause relevance, and logical reasoning.

## 🎯 System Overview

This system provides:
- **Semantic Search**: Uses FAISS or Pinecone vector databases for dense semantic retrieval
- **Clause Retrieval**: Identifies specific contractual or policy clauses with full context
- **Logic & Rule Evaluation**: Evaluates complex conditions and multi-clause logic trees
- **Structured Output**: Returns JSON responses with natural language answers, relevant clauses, and logic traces

## 📦 Features

### Document Processing
- Supports PDF, DOCX, TXT, and PPTX formats
- Automatic document type detection (Insurance, Legal, HR, Compliance)
- Intelligent text chunking with overlap
- Metadata extraction (company names, page counts, etc.)

### Query Processing
- Natural language question understanding
- Intent and entity extraction
- Semantic similarity search
- Confidence scoring
- Logic tree generation

### Vector Storage
- FAISS (local) or Pinecone (cloud) support
- OpenAI embeddings or Sentence Transformers
- Configurable similarity thresholds
- Document filtering and management

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- (Optional) Pinecone API key for cloud vector storage

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd intelligent-query-retrieval-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env_example.txt .env
   # Edit .env with your API keys and configuration
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 📋 Configuration

### Environment Variables

Copy `env_example.txt` to `.env` and configure:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for Pinecone)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# System Configuration
VECTOR_DB_TYPE=faiss  # or pinecone
EMBEDDING_MODEL=text-embedding-ada-002
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

## 🔌 API Endpoints

### Health Check
```bash
GET /health
```

### Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

file: <document_file>
```

### Query Documents
```bash
POST /query
Content-Type: application/json

{
  "question": "Can the employee be terminated without notice in case of policy violation?",
  "document_ids": ["optional_specific_docs"],
  "include_logic": true,
  "max_results": 5
}
```

### List Documents
```bash
GET /documents
```

### Search Documents
```bash
GET /search?query=termination&top_k=5&document_ids=doc1,doc2
```

### System Statistics
```bash
GET /stats
```

## 📖 Usage Examples

### Example 1: Insurance Policy Query

**Question**: "Is pre-existing condition covered under this policy?"

**Response**:
```json
{
  "answer": "Based on the policy document, pre-existing conditions are covered after a 12-month waiting period, provided the condition was disclosed during application.",
  "clauses_used": [
    {
      "title": "Pre-existing Conditions Coverage",
      "text": "Pre-existing conditions will be covered after a 12-month waiting period...",
      "document": "health_insurance_policy.pdf",
      "page": 8,
      "relevance_score": 0.95
    }
  ],
  "logic_tree": {
    "type": "AND",
    "conditions": [
      "Condition was disclosed during application",
      "12-month waiting period has elapsed"
    ]
  },
  "confidence": 0.92
}
```

### Example 2: HR Policy Query

**Question**: "What happens if a customer cancels within 30 days?"

**Response**:
```json
{
  "answer": "According to the service agreement, customers may cancel within 30 days for a full refund, minus any processing fees.",
  "clauses_used": [
    {
      "title": "Cancellation Policy",
      "text": "Customers have the right to cancel this agreement within 30 days...",
      "document": "service_agreement.pdf",
      "page": 12,
      "relevance_score": 0.89
    }
  ],
  "logic_tree": {
    "type": "CONDITIONAL",
    "conditions": [
      "Cancellation occurs within 30 days of agreement"
    ]
  },
  "confidence": 0.89
}
```

## 🏗️ Architecture

### Core Components

1. **Document Processor** (`document_processor.py`)
   - Handles file parsing and text extraction
   - Creates intelligent chunks with metadata
   - Detects document types and extracts entities

2. **Vector Store** (`vector_store.py`)
   - Manages document embeddings
   - Provides similarity search functionality
   - Supports FAISS and Pinecone backends

3. **Query Engine** (`query_engine.py`)
   - Processes natural language queries
   - Generates structured responses
   - Builds logic trees for reasoning

4. **API Layer** (`api.py`)
   - FastAPI REST endpoints
   - File upload handling
   - Background task processing

### Data Flow

```
Document Upload → Text Extraction → Chunking → Embedding → Vector Store
                                                           ↓
Query → Intent Extraction → Semantic Search → LLM Processing → Structured Response
```

## 🔧 Development

### Project Structure
```
├── main.py                 # Application entry point
├── api.py                  # FastAPI application
├── config.py              # Configuration management
├── models.py              # Pydantic data models
├── document_processor.py  # Document processing logic
├── vector_store.py        # Vector database operations
├── query_engine.py        # Query processing and LLM integration
├── requirements.txt       # Python dependencies
├── env_example.txt        # Environment variables template
├── README.md             # This file
├── uploads/              # Temporary file storage
└── vector_store/         # FAISS index storage
```

### Adding New Document Types

1. Extend `DocumentType` enum in `models.py`
2. Add detection logic in `document_processor.py`
3. Update keyword patterns for type detection

### Customizing Embeddings

1. Change `EMBEDDING_MODEL` in configuration
2. Support both OpenAI and Sentence Transformers
3. Adjust chunk size and overlap as needed

## 🧪 Testing

### Manual Testing

1. **Upload a document**:
   ```bash
   curl -X POST "http://localhost:8000/upload" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@your_document.pdf"
   ```

2. **Query the document**:
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '{"question": "What are the termination conditions?"}'
   ```

### Automated Testing

Create test files in a `tests/` directory:
```python
# tests/test_query_engine.py
import pytest
from query_engine import QueryEngine
from vector_store import VectorStore

def test_query_processing():
    vector_store = VectorStore()
    query_engine = QueryEngine(vector_store)
    # Add test cases
```

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure `OPENAI_API_KEY` is set in `.env`
   - Verify the key is valid and has sufficient credits

2. **Vector Store Initialization Error**
   - For FAISS: Check disk space and permissions
   - For Pinecone: Verify API key and environment

3. **Document Processing Errors**
   - Check file format support
   - Verify file size limits
   - Ensure proper file encoding

4. **Memory Issues**
   - Reduce `CHUNK_SIZE` for large documents
   - Use Pinecone instead of FAISS for large datasets
   - Monitor system resources

### Logs

Check `app.log` for detailed error information:
```bash
tail -f app.log
```

## 📈 Performance Optimization

### For Large Document Collections

1. **Use Pinecone** for cloud-based vector storage
2. **Increase chunk overlap** for better context preservation
3. **Adjust similarity threshold** based on accuracy needs
4. **Implement caching** for frequent queries

### For Real-time Applications

1. **Background processing** for document uploads
2. **Async query processing** for better responsiveness
3. **Connection pooling** for database operations
4. **Load balancing** for high-traffic scenarios

## 🔒 Security Considerations

1. **API Key Management**: Store keys securely in environment variables
2. **File Validation**: Validate uploaded files for malicious content
3. **Access Control**: Implement authentication for production use
4. **Data Privacy**: Ensure compliance with data protection regulations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Create an issue with detailed information
4. Include system configuration and error messages

---

**Built with ❤️ for intelligent document analysis** 