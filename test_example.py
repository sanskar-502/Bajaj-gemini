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

    2.2 The Company may terminate this Agreement without cause by providing 30 days written notice to the Employee.

    SECTION 3: SEVERANCE
    If the Company terminates the Employee without cause, the Employee shall be entitled to severance pay equal to one month's salary for each year of service.
    """
    sample_file = "sample_employment_agreement.txt"
    with open(sample_file, "w") as f:
        f.write(sample_text)
    return sample_file


def test_document_processing():
    """Test document processing functionality"""
    print("🧪 Testing Document Processing...")
    sample_file = create_sample_document()
    try:
        processor = DocumentProcessor()
        # FIX: Provide a document_id as the second argument
        result = processor.process_document(sample_file, document_id="test_doc_proc.txt")
        print("✅ Document processed successfully!")
        print(f"   Document ID: {result['document_id']}")
        print(f"   Total chunks: {result['total_chunks']}")
        return result
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        return None
    finally:
        if os.path.exists(sample_file):
            os.remove(sample_file)


def test_vector_store():
    """Test vector store functionality"""
    print("\n🧪 Testing Vector Store...")
    try:
        vector_store = VectorStore()
        sample_chunks = [{
            "id": "vs_chunk_1",
            "text": "The Company may terminate this Agreement at any time for cause.",
            "metadata": {"document_id": "vs_test.pdf", "chunk_id": 1}
        }]
        vector_store.add_documents(sample_chunks)
        search_results = vector_store.search("termination conditions", top_k=1)
        print("✅ Vector store test successful!")
        print(f"   Search results: {len(search_results)}")
        return vector_store
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return None


def test_query_engine(vector_store: VectorStore):
    """Test query engine functionality"""
    print("\n🧪 Testing Query Engine...")
    if not vector_store:
        print("⏩ Skipping Query Engine test because Vector Store is unavailable.")
        return None
    try:
        query_engine = QueryEngine(vector_store)
        request = QueryRequest(question="What are the conditions for termination for cause?")
        response = query_engine.process_query(request)
        print("✅ Query engine test successful!")
        print(f"   Answer: {response.answer[:70]}...")
        return query_engine
    except Exception as e:
        print(f"❌ Query engine test failed: {e}")
        return None


def test_end_to_end():
    """Test the complete system end-to-end"""
    print("\n🧪 Testing End-to-End System...")
    try:
        # Step 1: Process document
        doc_result = test_document_processing()
        if not doc_result:
            raise ValueError("Document processing step failed.")

        # Step 2: Initialize vector store and add documents
        vector_store = VectorStore()
        vector_store.add_documents(doc_result["chunks"])

        # Step 3: Test query engine
        query_engine = QueryEngine(vector_store)
        request = QueryRequest(question="What severance is provided for termination without cause?")
        response = query_engine.process_query(request)

        print("✅ End-to-end test successful!")
        print(f"   Query: {request.question}")
        print(f"   Answer: {response.answer}")
        return True
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 LLM-powered Intelligent Query–Retrieval System - Test Suite")
    print("=" * 70)

    # Sequentially run tests
    passed_tests = 0
    total_tests = 4

    # Test 1: Document Processing
    doc_result = test_document_processing()
    if doc_result:
        passed_tests += 1

    # Test 2: Vector Store
    vector_store = test_vector_store()
    if vector_store:
        passed_tests += 1

    # Test 3: Query Engine (depends on vector store)
    if test_query_engine(vector_store):
        passed_tests += 1

    # Test 4: End-to-End
    if test_end_to_end():
        passed_tests += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! The system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the logs above for details.")

if __name__ == "__main__":
    main()