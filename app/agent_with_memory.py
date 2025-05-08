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