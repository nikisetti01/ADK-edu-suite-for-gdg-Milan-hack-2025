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
