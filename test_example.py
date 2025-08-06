#!/usr/bin/env python3
"""
Test example for the LLM-powered Intelligent Query–Retrieval System
This file demonstrates how to use the system programmatically
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from query_engine import QueryEngine
from models import QueryRequest

def create_sample_document():
    """Create a sample document for testing"""
    sample_text = """
    EMPLOYMENT AGREEMENT
    
    This Employment Agreement (the "Agreement") is entered into between ABC Corporation (the "Company") and John Doe (the "Employee") effective as of January 1, 2024.
    
    SECTION 1: TERM OF EMPLOYMENT
    The Employee's employment shall commence on January 1, 2024, and shall continue until terminated in accordance with the terms of this Agreement.
    
    SECTION 2: TERMINATION
    2.1 The Company may terminate this Agreement at any time for cause, including but not limited to:
        - Violation of company policies
        - Misconduct or insubordination
        - Poor performance
        - Breach of confidentiality
    
    2.2 The Company may terminate this Agreement without cause by providing 30 days written notice to the Employee.
    
    2.3 The Employee may terminate this Agreement by providing 14 days written notice to the Company.
    
    SECTION 3: CONFIDENTIALITY
    The Employee agrees to maintain the confidentiality of all proprietary and confidential information of the Company during and after employment.
    
    SECTION 4: NON-COMPETE
    For a period of 12 months following termination, the Employee shall not engage in any business that competes with the Company within a 50-mile radius.
    
    SECTION 5: SEVERANCE
    If the Company terminates the Employee without cause, the Employee shall be entitled to severance pay equal to one month's salary for each year of service.
    """
    
    # Create sample document file
    sample_file = "sample_employment_agreement.txt"
    with open(sample_file, "w") as f:
        f.write(sample_text)
    
    return sample_file

def test_document_processing():
    """Test document processing functionality"""
    print("🧪 Testing Document Processing...")
    
    # Create sample document
    sample_file = create_sample_document()
    
    try:
        # Initialize document processor
        processor = DocumentProcessor()
        
        # Process document
        result = processor.process_document(sample_file)
        
        print(f"✅ Document processed successfully!")
        print(f"   Document ID: {result['document_id']}")
        print(f"   Total chunks: {result['total_chunks']}")
        print(f"   Document type: {result['metadata'].document_type}")
        print(f"   Company name: {result['metadata'].company_name}")
        
        return result
        
    except Exception as e:
        print(f"❌ Document processing failed: {str(e)}")
        return None
    finally:
        # Clean up sample file
        if os.path.exists(sample_file):
            os.remove(sample_file)

def test_vector_store():
    """Test vector store functionality"""
    print("\n🧪 Testing Vector Store...")
    
    try:
        # Initialize vector store
        vector_store = VectorStore()
        
        # Create sample chunks
        sample_chunks = [
            {
                "id": "chunk_1",
                "text": "The Company may terminate this Agreement at any time for cause.",
                "metadata": {
                    "document_id": "employment_agreement.pdf",
                    "chunk_id": 1,
                    "document_type": "legal_contract",
                    "company_name": "ABC Corporation",
                    "page_count": 5,
                    "upload_timestamp": "2024-01-01T00:00:00"
                }
            },
            {
                "id": "chunk_2",
                "text": "The Employee agrees to maintain confidentiality of all proprietary information.",
                "metadata": {
                    "document_id": "employment_agreement.pdf",
                    "chunk_id": 2,
                    "document_type": "legal_contract",
                    "company_name": "ABC Corporation",
                    "page_count": 5,
                    "upload_timestamp": "2024-01-01T00:00:00"
                }
            }
        ]
        
        # Add documents to vector store
        vector_store.add_documents(sample_chunks)
        
        # Test search
        search_results = vector_store.search("termination conditions", top_k=2)
        
        print(f"✅ Vector store test successful!")
        print(f"   Search results: {len(search_results)}")
        for i, result in enumerate(search_results):
            print(f"   Result {i+1}: Score {result.score:.3f}")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Vector store test failed: {str(e)}")
        return None

def test_query_engine(vector_store):
    """Test query engine functionality"""
    print("\n🧪 Testing Query Engine...")
    
    try:
        # Initialize query engine
        query_engine = QueryEngine(vector_store)
        
        # Test queries
        test_queries = [
            "Can the employee be terminated without notice?",
            "What are the confidentiality requirements?",
            "What happens if the company terminates without cause?"
        ]
        
        for query in test_queries:
            print(f"\n   Query: {query}")
            
            # Create query request
            request = QueryRequest(
                question=query,
                include_logic=True,
                max_results=3
            )
            
            # Process query
            response = query_engine.process_query(request)
            
            print(f"   Answer: {response.answer[:100]}...")
            print(f"   Confidence: {response.confidence:.2f}")
            print(f"   Clauses used: {len(response.clauses_used)}")
            print(f"   Logic type: {response.logic_tree.type}")
        
        print(f"✅ Query engine test successful!")
        return query_engine
        
    except Exception as e:
        print(f"❌ Query engine test failed: {str(e)}")
        return None

def test_end_to_end():
    """Test the complete system end-to-end"""
    print("\n🧪 Testing End-to-End System...")
    
    try:
        # Step 1: Process document
        doc_result = test_document_processing()
        if not doc_result:
            return False
        
        # Step 2: Initialize vector store and add documents
        vector_store = VectorStore()
        vector_store.add_documents(doc_result["chunks"])
        
        # Step 3: Test query engine
        query_engine = QueryEngine(vector_store)
        
        # Step 4: Test a complex query
        complex_query = "What are the conditions for termination and what severance is provided?"
        
        request = QueryRequest(
            question=complex_query,
            include_logic=True,
            max_results=5
        )
        
        response = query_engine.process_query(request)
        
        print(f"\n📋 Complex Query Results:")
        print(f"   Query: {complex_query}")
        print(f"   Answer: {response.answer}")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Intent: {response.query_intent}")
        print(f"   Clauses found: {len(response.clauses_used)}")
        
        for i, clause in enumerate(response.clauses_used):
            print(f"   Clause {i+1}: {clause.title} (Score: {clause.relevance_score:.2f})")
        
        print(f"   Logic tree: {response.logic_tree.type} with {len(response.logic_tree.conditions)} conditions")
        
        print(f"\n✅ End-to-end test successful!")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 LLM-powered Intelligent Query–Retrieval System - Test Suite")
    print("=" * 70)
    
    # Check environment
    if not Config.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set. Please set it in your environment or .env file.")
        return
    
    # Run tests
    success_count = 0
    total_tests = 4
    
    # Test 1: Document Processing
    if test_document_processing():
        success_count += 1
    
    # Test 2: Vector Store
    vector_store = test_vector_store()
    if vector_store:
        success_count += 1
    
    # Test 3: Query Engine
    if vector_store and test_query_engine(vector_store):
        success_count += 1
    
    # Test 4: End-to-End
    if test_end_to_end():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the configuration and try again.")
    
    print("\n💡 Next steps:")
    print("   1. Set up your .env file with proper API keys")
    print("   2. Run 'python main.py' to start the API server")
    print("   3. Use the API endpoints to upload documents and query them")
    print("   4. Check the README.md for detailed usage instructions")

if __name__ == "__main__":
    main() 