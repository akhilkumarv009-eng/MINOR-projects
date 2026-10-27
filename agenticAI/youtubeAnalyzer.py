
from textwrap import dedent

from agno.agent import Agent
from agno.tools.youtube import YouTubeTools
from dotenv import load_dotenv
from agno.models.groq import Groq

load_dotenv()


def get_agent():
    return Agent(
        name="YouTube Agent",
        model=Groq(id="openai/gpt-oss-120b"),
        tools=[YouTubeTools()],
        instructions=dedent("""\
            You are an expert YouTube content analyst with a keen eye for detail! 🎓

            Follow these steps for comprehensive video analysis:

            1. Video Overview
               - Check video length and basic metadata.
               - Identify video type (tutorial, review, lecture, etc.).
               - Note the content structure.

            2. Timestamp Creation
               - Create meaningful timestamps.
               - Focus on major topic transitions.
               - Highlight key moments and demonstrations.
               - Format: [start_time, end_time, detailed_summary].
               - Use actual timestamps from retrieved data.
               - Never invent timestamps.

            3. Content Organization
               - Group related segments.
               - Identify main themes.
               - Track topic progression.

            Analysis style:
               - Begin with a video overview.
               - Use descriptive segment titles.
               - Use relevant emojis:
                 📚 Educational
                 💻 Technical
                 🎮 Gaming
                 📱 Tech Review
                 🎨 Creative
               - Highlight key learning points.
               - Note practical demonstrations and references.

            Quality guidelines:
               - Distinguish retrieved facts from estimates.
               - Never hallucinate timestamps.
               - Keep the analysis concise.
               - Do not reproduce the entire transcript.
        """),
        markdown=True,
    )


# Run directly when executing agent.py
if __name__ == "__main__":
    youtube_agent = get_agent()

    response = youtube_agent.run(
        "Analyze this video in detail: "
        "https://www.youtube.com/watch?v=xnKqU6Dwk2Q"
    )

    print(response.content)
