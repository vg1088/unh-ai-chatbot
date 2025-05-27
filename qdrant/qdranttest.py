import os
from  qdrant_client import QdrantClient, models
from fastembed import TextEmbedding


docs = ["The internship class requires 80 on field hours", "On field hours may be remote or in person"]
exec_path = os.path.dirname(__file__)
store_path = os.path.join(exec_path, "qdrant_data" )
# path = storepath
#if os.path.exists(store_path) == False:
 #   os.mkdir(store_path)

client = QdrantClient( host='localhost' )
embed_model = TextEmbedding()

embeds = embed_model.embed(docs)
embeds = models.Batch( ids=[1,2], vectors = list(embeds), payloads = [{"1" : docs[0]}, {"2" : docs[1]}] )

#client.create_collection(collection_name = "test_collection2",
#vectors_config = models.VectorParams(size=384, distance=models.Distance.DOT) )

client.upsert(collection_name = "test_collection2",
              points = embeds
             )

#print(list(embeds))

search_result = client.search(
    collection_name="test_collection2",
    query_vector = next(embed_model.embed(["How many hours do I need"]))
)
print(search_result)

client.close()
