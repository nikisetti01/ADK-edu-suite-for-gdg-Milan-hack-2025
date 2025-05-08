# ✅ app/rag_pipeline.py
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


# ✅ app/chatbot.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

app = FastAPI()

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

class Message(BaseModel):
    user_input: str

@app.post("/chat")
def chat(message: Message):
    response = chain.run(message.user_input)
    return {"response": response}


# ✅ app/qdrant_setup.py
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


# ✅ app/tools.py
from langchain.agents import tool

@tool
def get_current_weather(location: str) -> str:
    """Retrieve current weather in a given location."""
    # Mocked function
    return f"Weather in {location} is sunny and 25°C"

@tool
def search_hackathon_docs(query: str) -> str:
    """Search hackathon docs for specific protocol references."""
    # Placeholder for integration with vector store
    return f"Found info for query: {query}"


# ✅ app/agent_with_memory.py
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from app.tools import get_current_weather, search_hackathon_docs

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory()

agent = initialize_agent(
    tools=[get_current_weather, search_hackathon_docs],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

if __name__ == "__main__":
    result = agent.run("Che tempo fa a Roma? E spiegami il protocollo MCP")
    print(result)


# ✅ app/graph_rag.py
from langchain.graphs import GraphRAGRetriever
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Placeholder setup (graph retriever integration)
graph_retriever = GraphRAGRetriever.from_graph_data_source("hackathon-graph")

llm = OpenAI(temperature=0)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=graph_retriever
)

query = "Come funziona GraphRAG?"
result = rag_chain.run(query)
print(result)


# ✅ app/utils.py
import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key():
    return os.getenv("OPENAI_API_KEY")


# ✅ app/example_mcp_agent.py
"""
Esempio che mostra come integrare un agent LLM che ragiona sul protocollo MCP (Model Context Protocol)
"""
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# Tool simulato per leggere specifica MCP
def get_mcp_spec(_: str) -> str:
    return "Il protocollo MCP è progettato per descrivere il contesto dei modelli LLM attraverso messaggi strutturati e metadata."

mcp_tool = Tool(
    name="GetMCPSpec",
    func=get_mcp_spec,
    description="Spiega come funziona il protocollo MCP"
)

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory()

agent = initialize_agent(
    tools=[mcp_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

if __name__ == "__main__":
    print(agent.run("Mi spieghi come funziona MCP?"))
