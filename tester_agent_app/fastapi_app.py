import os
import json
import asyncio
import glob
from datetime import datetime

from pathlib import Path
from dotenv import load_dotenv

from google.genai.types import (
    Part,
    Content,
)

from google.adk.runners import Runner
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService

from fastapi import FastAPI, WebSocket, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from tester_agent.agent import root_agent

#
# ADK Streaming
#

# Load Gemini API Key
load_dotenv()

APP_NAME = "ADK Streaming example"
session_service = InMemorySessionService()

# Path to completed exams PDFs
GROUND_TRUTH_DIR = Path("C:/Users/krysp/Desktop/Hackathon/gdg2025/tester_agent_app/tester_agent/completed_exams")


def start_agent_session(session_id: str):
    """Starts an agent session"""

    # Create a Session
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=session_id,
        session_id=session_id,
    )

    # Create a Runner
    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    # Set response modality = TEXT
    run_config = RunConfig(response_modalities=["TEXT"])

    # Create a LiveRequestQueue for this session
    live_request_queue = LiveRequestQueue()

    # Start agent session
    live_events = runner.run_live(
        session=session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )
    return live_events, live_request_queue


async def agent_to_client_messaging(websocket, live_events):
    """Agent to client communication"""
    while True:
        async for event in live_events:
            # turn_complete
            if event.turn_complete:
                await websocket.send_text(json.dumps({"turn_complete": True}))
                print("[TURN COMPLETE]")

            if event.interrupted:
                await websocket.send_text(json.dumps({"interrupted": True}))
                print("[INTERRUPTED]")

            # Read the Content and its first Part
            part: Part = (
                event.content and event.content.parts and event.content.parts[0]
            )
            if not part or not event.partial:
                continue

            # Get the text
            text = event.content and event.content.parts and event.content.parts[0].text
            if not text:
                continue

            # Send the text to the client
            await websocket.send_text(json.dumps({"message": text}))
            print(f"[AGENT TO CLIENT]: {text}")
            await asyncio.sleep(0)


async def client_to_agent_messaging(websocket, live_request_queue):
    """Client to agent communication"""
    while True:
        text = await websocket.receive_text()
        content = Content(role="user", parts=[Part.from_text(text=text)])
        live_request_queue.send_content(content=content)
        print(f"[CLIENT TO AGENT]: {text}")
        await asyncio.sleep(0)


#
# FastAPI web app
#

app = FastAPI()

STATIC_DIR = Path("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    """Serves the index.html"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/ground-truth-pdfs")
async def get_ground_truth_pdfs():
    """Returns a list of PDF files in the completed exams directory"""
    try:
        # Get all PDF files in the completex exams directory
        pdf_files = []
        for pdf_path in GROUND_TRUTH_DIR.glob("*.pdf"):
            last_modified = datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat()
            pdf_files.append({
                "name": pdf_path.name,
                "path": str(pdf_path),
                "lastModified": last_modified
            })
        
        return JSONResponse(content=pdf_files)
    except Exception as e:
        print(f"Error getting PDFs: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/view-pdf")
async def view_pdf(path: str = Query(...)):
    """Serves a PDF file for viewing"""
    try:
        # Validate that the path is within the completed exams directory
        pdf_path = Path(path)
        if GROUND_TRUTH_DIR in pdf_path.parents or pdf_path.parent == GROUND_TRUTH_DIR:
            return FileResponse(
                path=path,
                media_type="application/pdf",
                filename=pdf_path.name
            )
        else:
            return JSONResponse(
                content={"error": "Access denied. File not in completed exams directory."}, 
                status_code=403
            )
    except Exception as e:
        print(f"Error serving PDF: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: int):
    """Client websocket endpoint"""

    # Wait for client connection
    await websocket.accept()
    print(f"Client #{session_id} connected")

    # Start agent session
    session_id = str(session_id)
    live_events, live_request_queue = start_agent_session(session_id)

    # Start tasks
    agent_to_client_task = asyncio.create_task(
        agent_to_client_messaging(websocket, live_events)
    )
    client_to_agent_task = asyncio.create_task(
        client_to_agent_messaging(websocket, live_request_queue)
    )
    await asyncio.gather(agent_to_client_task, client_to_agent_task)

    # Disconnected
    print(f"Client #{session_id} disconnected")