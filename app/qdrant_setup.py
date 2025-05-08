from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Setup Qdrant DB
client = QdrantClient(host="localhost", port=6333)

# Create collection
client.recreate_collection(
    collection_name="hackathon-rag",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
print("Collection created")