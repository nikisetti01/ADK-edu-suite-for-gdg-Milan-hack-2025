# ADK Made Simple - Agent Examples

This project demonstrates simple agents built using the Google Agent Development Kit (ADK). It accompanies the YouTube tutorial series "ADK Made Simple" hosted on the [AIOriented YouTube channel](https://www.youtube.com/@AIOriented).

Full series playlist: [ADK Made Simple Playlist](https://www.youtube.com/playlist?list=PLWUH7ke1DYK98Di2FF8Ux3IX6qG-mZ3A7)

## Lessons

| Lesson | Link | Description |
| ------ | ---- | ----------- |
| 1      | [Lesson 1: Basics of ADK & Reddit Scout Agent](https://www.youtube.com/watch?v=BiP4tKZKTvU) | Basics of ADK, Building a Reddit news agent with PRAW |
| 2      | [Lesson 2: Multi-Agent Systems & TTS](https://www.youtube.com/watch?v=FODBW9az-sw) | Combining ADK with MCP, Multi-Agent Systems, Text-To-Speech with ElevenLabs, LiteLLM |

## Agents

- **Reddit Scout**: Simulates fetching recent discussion titles from game development subreddits.

## General Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/chongdashu/adk-made-simple
    cd adk-made-simple
    ```

2.  **Create and activate a virtual environment (Recommended):**

    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install general dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Agent-Specific Setup:** Navigate to the specific agent's directory within `agents/` and follow the instructions in its `README.md` (or follow the steps below for the default agent).

## Setup & Running Agents

1.  **Navigate to Agent Directory:**

    ```bash
    cd agents/reddit_scout
    ```

2.  **Set up API Key:**

    - Copy the example environment file:
      ```bash
      cp ../.env.example .env
      ```
    - Edit the `.env` file and add your Google AI API Key. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).
      ```dotenv
      GOOGLE_API_KEY=YOUR_API_KEY_HERE
      ```
    - _Note:_ You might need to load this into your environment depending on your OS and shell (`source .env` or similar) if `python-dotenv` doesn't automatically pick it up when running `adk`.

3.  **Run the Agent:**

    - Make sure your virtual environment (from the root directory) is activated.
    - From the `agents/reddit_scout` directory, run the agent using the ADK CLI, specifying the core code package:
      ```bash
      adk run reddit_scout
      ```
    - Alternatively, from the project root (`adk-made-simple`), you might be able to run:
      ```bash
      adk run agents/reddit_scout
      ```
      _(Check ADK documentation for preferred discovery method)_
    - Asynchronous agents can only be run from the web view, so first `cd` into the `agents` directory and run 
      ```bash
      adk web
      ```
      _(Check ADK documentation for preferred discovery method)_

4.  **Interact:** The agent will start, and you can interact with it in the terminal. Try prompts like:
    - `What's the latest news?`
    - `Give me news from unrealengine`

## Project Structure Overview

```
adk-made-simple/
├── agents/
│   ├── reddit_scout/        # Lesson 1: Reddit Scout Agent
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── async_reddit_scout/  # Lesson 2: Asynchronous Reddit Scout Agent
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── summarizer/          # Lesson 2: Newscaster Summarizer Agent
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── speaker/             # Lesson 2: Speaker Agent
│   │   ├── __init__.py
│   │   └── agent.py
│   └── coordinator/         # Lesson 2: Coordinator Agent combining sub-agents
│       ├── __init__.py
│       └── agent.py
├── .env.example             # Environment variables example
├── .gitignore               # Root gitignore file
├── requirements.txt         # Project dependencies
├── README.md                # This file (Overall Project README)
└── PLAN.md                  # Development plan notes
```
