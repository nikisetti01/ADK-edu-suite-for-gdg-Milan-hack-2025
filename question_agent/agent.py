from google.adk.agents import Agent
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from google import genai
import os
import sounddevice as sd
from google.genai import Client
import google.genai as genai
from scipy.io.wavfile import write
from google.cloud import texttospeech

client = genai.Client(api_key="AIzaSyA2liZpTTjGNF29AQ4Z1wv7kBO48oYj-m4")

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


def tts_google(data : str):
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


def record_audio(filename="audio.wav", duration=5, fs=44100):
    print("Registrazione in corso...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print(f"Audio salvato in: {filename}")
    return filename

def stt_tool(filepath):
    print("Caricamento file e trascrizione...")
    try:
        audio_file = genai.upload_file(path=filepath)
        #prompt = "Trascrivi parola per parola il contenuto dell'audio, se è silenzio o rumore scrivi [SILENZIO]"
        myfile = client.files.upload(file="audio.wav")

        response = client.models.generate_content(
        model="gemini-1.5-pro-001", contents=["Describe this audio clip", myfile]
        )
        genai.delete_file(audio_file.name)
        print("Testo trascritto:")
        print(response.text)
        return response.text
    except Exception as e:
        print("Errore durante la trascrizione:", e)

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
    output_key="question",
    #before_agent_callback=stt_tool
)

sst_agent=Agent(
    model='gemini-2.0-flash-001',
    name="pipeline",
    instruction="you are a pipeline agent, you have to link the imprecision_detector and the sochratic_agent.",
    output_key="speech",
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
    "use the tool retrieve to understand and create a better question about that topic. For answer report the {errors} received from the imprecision_detector agent and the new question you have to ask the user. use the tts_google tool to convert text to speech",
    output_key="question",
    tools=[qdrant_retrieve,tts_google]
)

# TTS_agent=Agent(
#     model='gemini-2.0-flash-001',
#     name="TTS",
#     description="you are a text to speech agent, you create a speech about the error the user made in the answer .",
#     instruction="you have to ask the user a question based on the error made, just one simple question that go deep in that topic"
#     "use the tool TTS to speech about that topic",
#     output_key="question",
#     tools=[TTS]
# )

# #PIPELINE PER FAR PARLARE IL SOCRATICO
# pipeline_sochraticTospeech = Agent(
#     model='gemini-2.0-flash-001',
#     name="pipeline",
#     description="you are a pipeline agent, you have to link the imprecision_detector and the sochratic_agent.",
#     instruction="you are the pipeline agent, you have to use the tts_agent and pass its answer to the sochratic_agent.",
#     output_key="question",
#     tools=[qdrant_retrieve]
# )

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
    instruction="""you are the coordinator when the user say hi you have to call the sub-agent stt agent, then question_agent produce the question than when the user answer you pass the answer of the user
    to the subagent  pipeline_agent.""",
    sub_agents=[sst_agent,question_agent, imprecision_detector, sochratic_agent],
)