# test_pinecone.py

"""
A standalone script to test Pinecone connectivity and basic operations.
It performs the following steps:
1. Loads the Pinecone API key from the environment.
2. Initializes the Pinecone client and a sentence-transformer model.
3. Checks if a specific Pinecone index exists, creating it if necessary.
4. Embeds sample text data and upserts it into the index.
5. Performs a similarity search query and prints the results.
"""

import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

def main():
    """Main function to run the Pinecone test."""
    
    # --- 1. Configuration ---
    print("--- Step 1: Configuring Environment ---")
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY is not set in the .env file.")
        print("This script cannot run without a valid Pinecone API key.")
        return  # Exit the function

    print("✅ Pinecone API key loaded.")
    print("-" * 35)

    # --- 2. Initialize Clients ---
    print("--- Step 2: Initializing Clients ---")
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("✅ Successfully connected to Pinecone.")

        # Initialize the embedding model from Hugging Face
        model = SentenceTransformer("intfloat/multilingual-e5-large")
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model 'multilingual-e5-large' loaded (Dimension: {embedding_dim}).")
    except Exception as e:
        print(f"❌ Failed to initialize clients: {e}")
        return
    print("-" * 35)

    # --- 3. Verify Pinecone Index ---
    print("--- Step 3: Verifying Pinecone Index ---")
    INDEX_NAME = "legal-llm-query"

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' not found. Creating a new serverless index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=embedding_dim,
            metric="cosine",  # Cosine similarity is great for text embeddings
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        # Wait for the index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            print("Waiting for index to become ready...")
            time.sleep(5)
        print(f"✅ Index '{INDEX_NAME}' is ready.")
    else:
        print(f"✅ Found existing index: '{INDEX_NAME}'.")
    
    # Connect to the index
    index = pc.Index(INDEX_NAME)
    print("-" * 35)

    # --- 4. Upsert Data ---
    print("--- Step 4: Upserting Sample Data ---")
    sample_texts = [
        "The termination clause outlines the conditions under which the agreement can be ended.",
        "Force majeure allows a party to suspend or terminate the contract upon the occurrence of certain unforeseeable events.",
        "Indemnity is a contractual obligation of one party to compensate for the losses incurred by another party."
    ]
    
    # Create unique IDs and embeddings for each text
    vectors_to_upsert = []
    for i, text in enumerate(sample_texts):
        embedding = model.encode(text).tolist()
        vector = (f"sample-id-{i}", embedding, {"text": text})
        vectors_to_upsert.append(vector)

    # Upsert the vectors into the index
    index.upsert(vectors=vectors_to_upsert)
    print(f"✅ Upserted {len(vectors_to_upsert)} vectors into the index.")
    
    # It can take a few moments for upserted vectors to be indexed
    print("Waiting a few seconds for vectors to be indexed...")
    time.sleep(10)
    print("-" * 35)

    # --- 5. Run a Query ---
    print("--- Step 5: Running a Similarity Search Query ---")
    query_text = "What happens if a party cannot fulfill the contract due to a natural disaster?"
    query_embedding = model.encode(query_text).tolist()

    print(f"Query: \"{query_text}\"\n")
    
    results = index.query(
        vector=query_embedding, 
        top_k=2, 
        include_metadata=True
    )

    print("Top search results:")
    for match in results.matches:
        print(f"  Score: {match.score:.4f}")
        print(f"  Text:  {match.metadata['text']}\n")
    print("-" * 35)
    
    # --- 6. Cleanup (Optional) ---
    # print(f"--- Step 6: Deleting Index '{INDEX_NAME}' ---")
    # pc.delete_index(INDEX_NAME)
    # print("✅ Index deleted.")
    # print("-" * 35)

if __name__ == "__main__":
    main()