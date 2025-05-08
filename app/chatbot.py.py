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