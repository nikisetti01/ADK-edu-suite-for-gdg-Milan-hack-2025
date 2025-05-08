from qdrant_client import QdrantClient
from sentence_transformers import sentence_transformers

# Connessione
client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="my_collection",
    vectors_config={"size": 384, "distance": "Cosine"}
)

model = SentenceTrasformer('all-MiniLm-L6-v2')

def saveTextToQdrant(text, metadata=none):
    embeddings = model.encode(text)

    client.upsert(
        collection_name="collectionprova",
        points = [
            "id" : None,
            "vector": embedding.tolist(),
            "payload": metadata or {"text":text}
        ]
    )
