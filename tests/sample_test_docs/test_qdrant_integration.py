import pytest
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from qdrantsearch import search_db

@pytest.fixture
def setup_qdrant():
    """Fixture to set up the Qdrant client using the existing 'internship2024' collection."""
    # Initialize Qdrant Client
    client = QdrantClient(host="localhost", port=6333)
    
    # Use the existing collection name
    collection_name = "internship2024"
    
    # Ensure the collection exists
    assert client.collection_exists(collection_name), f"Collection '{collection_name}' does not exist."

    yield client, collection_name

def test_search_db_with_existing_collection(setup_qdrant):
    client, collection_name = setup_qdrant
    
    # Initialize embedding model
    embed_model = TextEmbedding()
    
    # Define documents and embed them
    docs = ["The internship class requires 80 on-field hours", "On-field hours may be remote or in-person"]
    embeds = embed_model.embed(docs)
    batch = models.Batch(
        ids=[1, 2],
        vectors=list(embeds),
        payloads=[{"text": docs[0]}, {"text": docs[1]}]
    )
    
    # Upsert the batch of embeddings into Qdrant
    client.upsert(collection_name=collection_name, points=batch)

    # Query Qdrant with a search term
    query_text = "How many hours do I need"
    query_vector = next(embed_model.embed([query_text]))
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector
    )

    # Print search results for debugging
    print("Search Results:", search_result)
    
    # Check that at least one result is returned and contains expected content
    assert search_result, "No results returned from Qdrant search."
    assert any("80 on-field hours" in result.payload["text"] for result in search_result), \
        "Expected text '80 on-field hours' not found in search results."
    
    client.close()
