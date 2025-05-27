from  qdrant_client import QdrantClient, models
from fastembed import TextEmbedding


def search_db(client, q_text, embed_model, collection_name="internship2024"):
    """" query vector DB, returns http.models.models object"""
    search_result = client.search(
    collection_name=collection_name,
    limit = 15,
    query_vector = next(embed_model.embed([q_text]))
    )
    return search_result

if __name__ == "__main__":
    client = QdrantClient( host='localhost' )
    embed_model = TextEmbedding()
    result = search_db(client, "What is the schedule for week 4", embed_model)
    print(result[0].payload.values())