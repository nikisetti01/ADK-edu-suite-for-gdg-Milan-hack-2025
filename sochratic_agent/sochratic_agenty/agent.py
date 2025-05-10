from google.adk.agents import Agent
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
folder = os.path.join(os.path.dirname(_file_), "data")
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

question_agent=Agent(
    model='gemini-2.0-flash-001',
    name="questioner",
    instruction="you are a questioner, ask a question to the user and wait for the answer about revolution French.",
    output_key="question"
)


imprecision_detector=Agent(
    model='gemini-2.0-flash-001',
    name="imprecision_detector",
    description="you are an imprecision_detector, just find and return the imprecision in the user answer.",
    instruction= """Using the answer of the user and the {question} receive by the
    agent and the retrieve tool find the phrases used by the user answer that are incorrect for the answer or not totaly correct and answer with them, DO NOT correct just report the error phrases to the sochratic agent without answer .""",
    output_key="errors",
    tools=[qdrant_retrieve]
)


sochratic_agent=Agent(
    model='gemini-2.0-flash-001',
    name="sochratique",
    description="you are a sochratique agent, you need to make a question about the error the user made in the answer .",
    instruction="you have to ask the user a question based on the error made, just one simple question that go deep in that topic"
    "use the tool retrieve to understand and create a better question about that topic. For answer report the {errors} received from the imprecision_detector agent and the new question you have to ask the user.",
    output_key="question",
    tools=[qdrant_retrieve]
)


pipeline_agent= Agent(
    model='gemini-2.0-flash-001',
    name="pipeline",
    description="you are a pipeline agent, you have to link the imprecision_detector and the sochratic_agent.",
    instruction="you are the pipeline agent, you have to use the imprecision_detector and pass its answer to the sochratic_agent.",
    output_key="question",
    tools=[qdrant_retrieve]
)


root_agent=Agent(
    model='gemini-2.0-flash-001',
    name="coordinator",
    description="you are the coordinator you have to coordinate the subagent.",
    instruction="""you are the coordinator when the user say hi you have to call the sub-agent question_agent produce the question than when the user answer you pass the answer of the user
    to the subagent  pipeline_agent.""",
    sub_agents=[question_agent, imprecision_detector, sochratic_agent],
)
