#!/usr/bin/env python3
"""
An improved test suite for the LLM-powered Intelligent Query–Retrieval System.
This script programmatically tests the full pipeline and uses the project's
.env configuration for initialization.
"""

import os
import sys
import shutil
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from query_engine import QueryEngine
from models import QueryRequest

# --- Test Constants ---
SAMPLE_DOC_FILENAME = "test_agreement.txt"
SAMPLE_DOC_ID = "sample-agreement-123"
TEST_QUERY = "What severance is provided for termination without cause?"
EXPECTED_ANSWER_SNIPPET = "one month's salary"
SAMPLE_TEXT = """
EMPLOYMENT AGREEMENT

This Agreement is between Global Tech Inc. and Jane Smith, effective July 1, 2024.

SECTION 1: TERMINATION
The Company may terminate this Agreement without cause by providing 30 days notice.

SECTION 2: SEVERANCE
If the Company terminates the Employee without cause, the Employee shall be entitled to
severance pay equal to one month's salary for each year of service.
"""

class SystemTests:
    """A class to encapsulate all system integration tests."""
    
    config = None
    doc_processor = None
    vector_store = None
    query_engine = None

    @classmethod
    def setup_class(cls):
        """
        Set up the test environment once for all tests.
        This initializes all components based on the .env file.
        """
        print("\n--- Setting up test environment ---")
        try:
            cls.config = Config()
            cls.doc_processor = DocumentProcessor()
            cls.vector_store = VectorStore()
            cls.query_engine = QueryEngine(cls.vector_store)

            with open(SAMPLE_DOC_FILENAME, "w") as f:
                f.write(SAMPLE_TEXT)

            print(f"Processing sample document: {SAMPLE_DOC_FILENAME}")
            doc_result = cls.doc_processor.process_document(SAMPLE_DOC_FILENAME, SAMPLE_DOC_ID)
            cls.vector_store.add_documents(doc_result["chunks"])
            print("--- Setup complete ---")
        except Exception as e:
            print(f"--- ❌ SETUP FAILED: {e} ---")
            cls.teardown_class()
            sys.exit(1)

    @classmethod
    def teardown_class(cls):
        """Clean up the test environment after all tests are run."""
        print("\n--- Tearing down test environment ---")
        if os.path.exists(SAMPLE_DOC_FILENAME):
            os.remove(SAMPLE_DOC_FILENAME)
        
        if cls.config and cls.config.VECTOR_DB_TYPE == "faiss":
            if os.path.exists(cls.config.VECTOR_STORE_DIR):
                shutil.rmtree(cls.config.VECTOR_STORE_DIR)
        
        print("--- Teardown complete ---")

    def test_document_is_processed(self):
        """Test 1: Verifies that the document was processed and indexed."""
        print("\n🧪 1. Testing Document Indexing...")
        stats = self.vector_store.get_stats()
        
        assert stats["total_vectors"] > 0, "No vectors were added to the store."
        print(f"✅ PASSED: Vector store contains {stats['total_vectors']} vectors.")

    def test_retrieval_and_query(self):
        """Test 2: Verifies the full RAG pipeline from query to answer."""
        print("\n🧪 2. Testing End-to-End Query and Retrieval...")
        
        request = QueryRequest(question=TEST_QUERY, document_ids=[SAMPLE_DOC_ID])
        response = self.query_engine.process_query(request)
        
        assert response is not None, "Query engine returned no response."
        assert response.answer is not None, "Response object has no answer."
        assert EXPECTED_ANSWER_SNIPPET in response.answer.lower(), \
            f"Answer did not contain expected text. Got: '{response.answer}'"
            
        print("✅ PASSED: End-to-end query returned the correct answer.")
        print(f"   Query: {TEST_QUERY}")
        print(f"   Answer: {response.answer}")


def main():
    """Main function to run the test suite."""
    print("🚀 LLM-powered Intelligent Query–Retrieval System - Test Suite")
    print("=" * 60)
    
    SystemTests.setup_class()
    
    test_suite = SystemTests()
    passed_tests = 0
    total_tests = 0
    
    try:
        total_tests += 1
        test_suite.test_document_is_processed()
        passed_tests += 1

        total_tests += 1
        test_suite.test_retrieval_and_query()
        passed_tests += 1

    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"❌ AN UNEXPECTED ERROR OCCURRED: {e}")
    
    SystemTests.teardown_class()
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed.")

if __name__ == "__main__":
    main()