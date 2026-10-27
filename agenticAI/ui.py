
import streamlit as st
from agent import get_agent

st.title("YouTube Video Analyzer")

agent = get_agent()

# input box
video_url = st.text_input("Enter Youtube Video Link")  # str
button = st.button("Analyze Video")  # True/False

if video_url and button:
    with st.spinner("Analyzing video...."):
        response = agent.run(
            f"Analyze this video: {video_url}"
        )

    st.markdown(response.content)
