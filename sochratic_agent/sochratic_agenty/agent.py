from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import ToolContext
from google.adk.tools import agent_tool 
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


# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


def penalty_function(tool_context :ToolContext):
    # Get the session from the context
    penalty = tool_context.state.get("penalty", 0)

    signal = int(tool_context.state.get("stop_signal", -1))

    # Reset the penalty if the coming from previous session
    if signal == -1 or (signal == 1 and penalty != 0):
        penalty = 0
    
    penalty = penalty + 1
    tool_context.state["penalty"] = penalty
    
def last_score(tool_context :ToolContext) -> float:
    score = float(tool_context.state.get("score", -1))

    signal = int(tool_context.state.get("stop_signal", -1))
    
    # Get the score from the session
    if score != -1  and signal == 1:
        # If score is not set, set it to 0.5
        tool_context.state["score"] = -999

    if signal == -1 or signal == 1:
        tool_context.state["stop_signal"] = 0

    return score

def stop_signal(tool_context :ToolContext):
    signal = int(tool_context.state.get("stop_signal", -1))

    if signal == -1 or signal == 0:
        tool_context.state["stop_signal"] = 1
    

  

    
# Setup Qdrant
clientQ = QdrantClient(":memory:")
client = texttospeech.TextToSpeechClient()
clientG = genai.Client(api_key="AIzaSyAc1bkPAQ9pE-EL1llkaqAoqNDjv1uscVI")

print("MODELLI AAAA")
response=clientG.models.list(config={'page_size': 5})
print(response.page)

# Create collection
print("Creating collection...")
clientQ.recreate_collection(
    collection_name="my_docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
score=0.0
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
    with open("C:/Users/krysp/Desktop/Hackathon/gdg2025/sochratic_agent/output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "C:/Users/krysp/Desktop/Hackathon/gdg2025/sochratic_agent/output.mp3"')
    
    await play_mp3_async("C:/Users/krysp/Desktop/Hackathon/gdg2025/sochratic_agent/output.mp3")

    try:
        os.remove("C:/Users/krysp/Desktop/Hackathon/gdg2025/sochratic_agent/output.mp3")
        print(f"File C:/Users/krysp/Desktop/Hackathon/gdg2025/sochratic_agent/output.mp3 eliminato.")
    except Exception as e:
        print(f"Errore durante l'eliminazione del file C:/Users/krysp/Desktop/Hackathon/gdg2025/sochratic_agent/output.mp3: {e}")    

def record_audio():
    duration=10 
    fs=44100
    filename="audio.wav"
    print("Registrazione in corso...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print(f"Audio salvato in: {filename}")
    stt_tool(filename)
    return filename

def stt_tool(filepath, output_txt="trascrizione.txt"):
    print("Caricamento file e trascrizione...")
    try:
        audio_file = genai.upload_file(path=filepath)
        myfile = clientG.files.upload(file=filepath)

        response = clientG.models.generate_content(
            model="gemini-1.5-pro-001",
            contents=["Describe this audio clip", myfile]
        )

        trascrizione = response.text

        # Salva nel file .txt
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(trascrizione)

        # Rimuovi il file da GenAI
        genai.delete_file(audio_file.name)

        print("Testo trascritto:")
        print(trascrizione)
        print(f"Trascrizione salvata in: {output_txt}")

        return trascrizione
    except Exception as e:
        print("Errore durante la trascrizione:", e)

question_agent = Agent(
    model='gemini-1.5-pro-001', # Assicurati che questo modello sia compatibile con il tuo ambiente
    name="questioner",
    description="This agent asks a question to the user about the French Revolution.",
    # Modifica l'istruzione: Rimuovi la parte sull'attesa della risposta e il controllo
    instruction="You are interested in just one thing. Ask the user one clear and specific question about the French Revolution.",
    output_key="question"
    # Rimuovi completamente i commenti e le linee relative a tools o before_agent_callback per record_audio
    # tools=[] # O lascia vuoto se non usi altri tool qui
    # before_agent_callback=None # Assicurati che non ci sia
)



imprecision_detector = Agent(
    model='gemini-2.0-flash-001',
    name="imprecision_detector",
    description="This agent identifies inaccuracies or imprecise statements in the user's answer.",
    instruction="""
        You are an imprecision detector. 
        Wait for the answer of the user. You receive the user's answer and the original question: {question}.
        Use the retrieval tool to verify facts. 
        Return to the user his sentence highlighting with bold characters the wrong parts of his answer. 
        Do **not** correct or explain the errors. Set the list of errors in your state and turn yourself off.
    """,
    output_key="errors",
    tools=[qdrant_retrieve]
)


fitness_agent= Agent(
    model="gemini-2.0-flash-001",
    name="fitness_agent",
    description="This agent evaluates the user answer giving it a score.",
    instruction="""
        You are a fitness agent, your task is to evaluate the user's answer and provide a score based on its quality. Your only output **MUST** the score of the answer you received: an evalutation between 0 and 1 as float of how much the errors in the answer {errors} are significant, where 0 is a very bad answer and 1 is a very good answer.
        You have the other following tasks, but you don't have to show them to the user nor setting them in the output - they are just for you, do your task in the order they are listed: 
        1. Call the penalty function to increase the penalty score.
        2. Provide the generated score of the answer you received as the ouput.
        3. Get the previous score using the last score function.
        4. Compare the two scores, if the variation between the previous and the current score is very small (a difference of 0.1 or 0.05) use the stop function to alert the following agents.
        5. If the score is almost equal to 1.0 (with a 0.05 or 0.1 threashold) use the stop function to alert the following agents.
        """,
    output_key="score",
    tools=[penalty_function, last_score, stop_signal,],
    # confronto con le risposte corrette, se la risposta è corretta e completa come quella da punteggio massimo
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


stopping_agent = Agent(
    model='gemini-2.0-flash-001',
    name="stopping_agent",
    description="This agent understands if the user answer is good enough and stop the pipeline if needed.",
    instruction="""
        You are the stopping agent, responsible for deciding whether to halt the execution of the pipeline based on a stop signal and the quality of the user's answer.

        If the value of the stop signal: If the stop function returns False, you must  return control to the coordinator agent requesting him to start the pipeline agent. In this case, you should not evaluate the answer or use any tools.

        However, if the stop function return True you must assess the quality of the user answer using the score ({score}) value. This score reflects how good the answer is: values close to 1 indicate a correct or acceptable answer, while values close to 0 indicate a poor or incorrect one.

        Use the evaluation tool, if the return is True the answer is good enough, you must return control to the coordinator agent asking him to restart the application by invoking the questioning agent, which will restart the application and generate new questions for the user.

        Otherwise use the evaluation tool, if the return is False, meaning the answer is not good enough, you must provide the user with clear feedback on what went wrong. This includes comparing their answer to the correct one using the appropriate tool, and helping them understand their mistake so they can improve. When return the control to the coordinator agent, you must ask him to stop the application because the user has to study again.

        When you return the control to the coordinator agent, you must provide him with a message that includes:
        - The action to perform: continue with the pipeline, restarting from a new question or stop the applicatio.
        - A brief feedback on the reason of the action.
        """,
    tools=[qdrant_retrieve, stop_function, evaluation_tool],  
    output_key="stop_flag",
    
)


pipeline_agent = SequentialAgent(
    name="pipeline_agent",
    description="Executes a sequence of detection, fitness and sochratic questioning.",
    sub_agents=[imprecision_detector, fitness_agent, sochratic_agent,],
)


root_agent = Agent(
    name="coordinator_agent",
    model='gemini-2.0-flash',
    name="agent",
    model='chat-bison-001',
    description="Coordinates all sub-agents.",
    instruction="""
        You are the coordinator agent, and your primary role is to manage the execution of the entire process, ensuring that each step is followed in sequence and that conditions for stopping or restarting the process are handled correctly.

        Your job consists in managing the following application:

        When the user greets you (e.g., "hi", "hello", or similar), your first action is to retrieve and deliver a question using the questioner agent. The questioner agent should only be invoked at the beginning of the process or when explicitly requested by the stopping agent. 

        Only the first time that the user asnwer the question coming from the questioner agent you must forward his answer to the pipeline agent. The pipeline agent will then handle the user's answer according to its sequential pattern and propose a new question. 

        Otherwise when the question to which the user is answering comes from the pipeline agent you must call the stopping agent every time to determine whether the process should be restarted, stopped, or continued. If the stopping agent indicates that the process should continue, you will invoke the pipeline agent for the next step without triggering the questioner agent. If the stopping agent indicates that the process should restart, you will give the user the feedback received from the stopping agent and call the questioner agent to ask a new question and restart the application from the first step. If the stopping agent indicates that the process should stop, you will terminate the process and notify the user with the message from the stopping agent.

        It is crucial that you do not invoke the questioner agent unless explicitly instructed by the stopping agent to restart the application or initiate a new question. You must not bypass or skip the sequence of steps, and all answers must be handled sequentially, with no exceptions.
    """,
    tools=[agent_tool.AgentTool(agent=question_agent), agent_tool.AgentTool(agent=pipeline_agent), agent_tool.AgentTool(agent=stopping_agent)],
)