
import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools

load_dotenv()


def build_agent():
    

    return Agent(
        model=Groq(id="openai/gpt-oss-120b"),
        tools=[YFinanceTools()],
        markdown=True,
        add_datetime_to_context=True,
        description=(
            "You are a financial research analyst who researches "
            "stock prices, company fundamentals, analyst recommendations, "
            "and financial news."
        ),
        instructions=[
            "Format responses using Markdown.",
            "Use tables to display financial data where useful.",
            "Use available tools to retrieve current financial information.",
            "Distinguish verified data from estimates and opinions.",
            "Explain important financial metrics in simple language.",
            "Never invent stock prices, financial metrics, or analyst ratings.",
            "State the relevant date for time-sensitive information.",
            "Do not guarantee investment returns.",
            "Clearly explain investment risks and uncertainties.",
            "Do not present financial information as personalized "
            "investment advice without sufficient context.",
        ],
    )

agent=build_agent()
agent.print_response("share the NVDA stock price and analytical reccomendation")


