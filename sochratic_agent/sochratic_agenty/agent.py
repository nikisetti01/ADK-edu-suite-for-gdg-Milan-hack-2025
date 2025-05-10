from google.adk.agents import Agent, SequentialAgent, BaseAgent
from google.adk.tools.function_tool import FunctionTool
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import os
from google.cloud import texttospeech
import pygame
import asyncio
import sounddevice as sd
from google.genai import Client
import google.genai as genai
from scipy.io.wavfile import write
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.runners import Runner
from typing import Optional
from google.genai import types 
from google.adk.sessions import InMemorySessionService

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Setup Qdrant
clientQ = QdrantClient(":memory:")
client = texttospeech.TextToSpeechClient()
clientG = genai.Client(api_key="AIzaSyA2liZpTTjGNF29AQ4Z1wv7kBO48oYj-m4")

# Create collection
print("Creating collection...")
clientQ.recreate_collection(
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
clientQ.upsert(collection_name="my_docs", points=points)

# Define function with type hints
def qdrant_retrieve(query: str) -> str:
    query_vector = model.encode(query).tolist()
    results = clientQ.search(
        collection_name="my_docs",
        query_vector=query_vector,
        limit=5
    )
    return "\n\n".join([hit.payload["text"] for hit in results])

async def play_mp3_async(path: str):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    # Aspetta che finisca, ma non blocca il thread principale
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)

async def tts_google(data : str):
    synthesis_input = texttospeech.SynthesisInput(text=data)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')
    
    await play_mp3_async("output.mp3")

# def record_audio(filename="audio.wav", duration=10, fs=44100):
#     print("Registrazione in corso...")
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()
#     write(filename, fs, audio)
#     print(f"Audio salvato in: {filename}")
#     #stt_tool(filename)
#     return None

def record_audio(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM request or skips the call."""
    agent_name = callback_context.agent_name
    print(f"[Callback] Before model call for agent: {agent_name}")

    # Inspect the last user message in the request contents
    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
         if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text
    print(f"[Callback] Inspecting last user message: '{last_user_message}'")


# def stt_tool(filepath, output_txt="trascrizione.txt"):
#     print("Caricamento file e trascrizione...")
#     try:
#         audio_file = genai.upload_file(path=filepath)
#         myfile = clientG.files.upload(file=filepath)

#         response = clientG.models.generate_content(
#             model="gemini-1.5-pro-001",
#             contents=["Describe this audio clip", myfile]
#         )

#         trascrizione = response.text

#         # Salva nel file .txt
#         with open(output_txt, "w", encoding="utf-8") as f:
#             f.write(trascrizione)

#         # Rimuovi il file da GenAI
#         genai.delete_file(audio_file.name)

#         print("Testo trascritto:")
#         print(trascrizione)
#         print(f"Trascrizione salvata in: {output_txt}")

#         return trascrizione
#     except Exception as e:
#         print("Errore durante la trascrizione:", e)


question_agent = Agent(
    model='gemini-2.0-flash-001',
    name="questioner",
    description="This agent asks a question to the user about the French Revolution.",
    instruction="You are interested in just one thing. Ask the user one clear and specific question about the French Revolution. Wait the answer of the user and give back controll to the pipeline_agent.",
    output_key="question",
    before_agent_callback=record_audio
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
- A single new Socratic question to ask the user about one of those errors. your answer must be created using the tts_google tool. use it always.
than put the answer of the user to the imprecision_detector agent and restart the pipeline. use the tts_google tool to create the speech version after every response.
""",
    output_key="question",
    tools=[qdrant_retrieve, tts_google]
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