import os
from contextlib import AsyncExitStack
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

# Sub-agent factories
from async_reddit_scout.agent import create_agent as create_reddit_scout_agent
from summarizer.agent import create_summarizer_agent
from speaker.agent import create_agent as create_speaker_agent

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

async def create_coordinator_agent():
    """Creates the Coordinator agent that delegates to Reddit Scout, Summarizer, and Speaker sub-agents."""

    # Manage multiple exit stacks for async sub-agents
    exit_stack = AsyncExitStack()
    await exit_stack.__aenter__()

    # Instantiate Reddit (async) and enter its exit stack
    reddit_agent, reddit_stack = await create_reddit_scout_agent()
    await exit_stack.enter_async_context(reddit_stack)

    # Instantiate Summarizer (sync)
    summarizer_agent = create_summarizer_agent()

    # Instantiate Speaker (async) and enter its exit stack
    speaker_agent, speaker_stack = await create_speaker_agent()
    await exit_stack.enter_async_context(speaker_stack)

    # Define a multi-model coordinator LLM
    coordinator_llm = LiteLlm(model="gemini/gemini-1.5-pro-latest", api_key=os.environ.get("GOOGLE_API_KEY"))

    # Create the Coordinator agent
    coordinator = Agent(
        name="coordinator_agent",
        description="Coordinates finding Reddit posts, summarizing titles, and converting text to speech.",
        model=coordinator_llm,
        instruction=(
            "You manage three sub-agents: Reddit Scout, Summarizer, and Speaker."
            "\n1. When the user asks for 'hot posts', delegate to Reddit Scout and return its raw list."
            "\n2. If the user then asks for a 'summary', delegate the Reddit Scout's exact output to Summarizer and return its summary."
            "\n3. If the user asks you to 'speak' or 'read', determine if they want the summary (if available) or the original list, then delegate the appropriate text to Speaker and return its result (URL)."
            "\n4. For other queries, respond directly without delegation."
        ),
        sub_agents=[reddit_agent, summarizer_agent, speaker_agent]
    )

    return coordinator, exit_stack

root_agent = create_coordinator_agent()