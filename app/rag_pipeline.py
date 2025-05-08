from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

from qdrant_client import QdrantClient

# Initialize vector DB
client = QdrantClient(host="localhost", port=6333)
embedding = HuggingFaceEmbeddings()

qdrant = Qdrant(
    client=client,
    collection_name="hackathon-rag",
    embeddings=embedding
)

# Setup LLM
llm = OpenAI(temperature=0)

# RetrievalQA pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=qdrant.as_retriever(),
    return_source_documents=True
)

# Query example
query = "Spiegami il protocollo A2A"
result = qa.run(query)
print(result)
