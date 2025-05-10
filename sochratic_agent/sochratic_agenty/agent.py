from google.adk.agents import Agent, SequentialAgent, BaseAgent
from google.adk.tools.function_tool import FunctionTool
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import os

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Setup Qdrant
client = QdrantClient(":memory:")

# Create collection
print("Creating collection...")
client.recreate_collection(
    collection_name="my_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Load documents
folder = os.path.join(os.path.dirname(__file__), "data")
points = []
for idx, filename in enumerate(os.listdir(folder)):
    with open(os.path.join(folder, filename), "r") as f:
        text = f.read()
        embedding = model.encode(text).tolist()
        points.append(PointStruct(id=idx, vector=embedding, payload={"text": text, "filename": filename}))

# Insert into Qdrant
print("Inserting documents...")
client.upsert(collection_name="my_docs", points=points)

# Define function with type hints
def qdrant_retrieve(query: str) -> str:
    query_vector = model.encode(query).tolist()
    results = client.search(
        collection_name="my_docs",
        query_vector=query_vector,
        limit=5
    )
    return "\n\n".join([hit.payload["text"] for hit in results])

question_agent = Agent(
    model='gemini-2.0-flash-001',
    name="questioner",
    description="This agent asks a question to the user about the French Revolution.",
    instruction="You are interested in just one thing. Ask the user one clear and specific question about the French Revolution. Wait the answer of the user and give back controll to the pipeline_agent.",
    output_key="question",
)



imprecision_detector = Agent(
    model='gemini-2.0-flash-001',
    name="imprecision_detector",
    description="This agent identifies inaccuracies or imprecise statements in the user's answer.",
    instruction="""You are an imprecision detector. 
Wait for the answer of the user. You receive the user's answer and the original question: {question}.
Use the retrieval tool to verify facts. 
Return to the user his sentence highlighting with bold characters the wrong parts of his answer. 
Do **not** correct or explain the errors. Set the list of errors in your state and turn yourself off.""",
    output_key="errors",
    tools=[qdrant_retrieve]
)



sochratic_agent = Agent(
    model='gemini-2.0-flash-001',
    name="sochratic_agent",
    description="This agent formulates a follow-up question based on a mistake the user made.",
    instruction="""You are a Socratic agent. 
User the highleighted errors received from the imprecision_detector agent to understand the user's mistakes in {errors}.
Use the retrieval tool if needed to better understand the topic.
Then, generate **one thought-provoking question** that focuses specifically on one of the incorrect phrases, encouraging the user to reflect or elaborate on that point.
Your answer must include:
- A single new Socratic question to ask the user about one of those errors.
than put the answer of the user to the imprecision_detector agent and restart the pipeline.
""",
    output_key="question",
    tools=[qdrant_retrieve]
)

pipeline_agent = SequentialAgent(
    name="pipeline_agent",
    description="Executes a sequence of detection and sochratic question.",
    sub_agents=[imprecision_detector, sochratic_agent],
)

root_agent = Agent(
    name="agent",
    model='gemini-2.0-flash',
    description="Coordinates all sub-agents.",
    instruction="""You are the coordinator agent. 
When the user greets you (e.g., says 'hi'), start the process by giving him the question using the question agent. Than do NOT  EVER use the question again but use pipeline_agent.
When the user responds handle the response by forwarding it to the  pipeline_agent. Each time the user answer at the socratic question restart the pipeline""",

    sub_agents=[question_agent,pipeline_agent],
)