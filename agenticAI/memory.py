
from agno.db.sqlite import SqliteDb
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()

db = SqliteDb()

def build_agent():
    return Agent(
        name="My Agent",
        model=Groq(id="openai/gpt-oss-120b"),
        tools=[DuckDuckGoTools()],
        instructions="You are a helpful AI assistant.",
        markdown=True,
        add_datetime_to_context=True,
        db=db,
        add_history_to_context=True
    )

groq_agent = build_agent()

groq_agent.print_response(
    "Good places to visit in Australia",
    session_id="travel-chat",
)

groq_agent.print_response(
    "What is the capital of that country?",
    session_id="travel-chat",
)


