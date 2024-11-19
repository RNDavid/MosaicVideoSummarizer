import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import whisper
from pytubefix import YouTube
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI model
model = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")

# Streamlit app
st.title("YouTube Video Summarizer")

# User inputs
youtube_video = st.text_input("Enter YouTube Video URL", value="paste your youtube link here and hit ENTER to start")
extra_terms = st.text_input("If you'd like to summarize a specific topic, type it below. Single words work best.")

# Process the YouTube video
if youtube_video and extra_terms:
    with st.spinner("Processing the video..."):
        # Download and transcribe the audio if not already done
        if not os.path.exists("transcription.txt"):
            youtube = YouTube(youtube_video)
            audio = youtube.streams.filter(only_audio=True).first()

            # Load the Whisper model
            whisper_model = whisper.load_model("base")

            with tempfile.TemporaryDirectory() as tmpdir:
                file = audio.download(output_path=tmpdir)
                transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

                with open("transcription.txt", "w") as file:
                    file.write(transcription)

        # Read the transcription
        with open("transcription.txt") as file:
            transcription = file.read()

        # Generate the summary
        prompt_template = """Summarize the following text: {transcription}. Please also call out any reference to {extra_terms}."""
        prompt = PromptTemplate(template=prompt_template, input_variables=["transcription", "extra_terms"])
        summary_chain = LLMChain(prompt=prompt, llm=model)

        output = summary_chain.invoke({'transcription': transcription, 'extra_terms': extra_terms})

        if 'text' in output:
            summary_text = output['text']

            # Save the summary
            summary_file_path = 'summary_output.txt'
            with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                summary_file.write(summary_text)

            # Display the summary
            st.text_area("Summary", summary_text, height=300)

            # Provide download links
            with open("transcription.txt", "rb") as file:
                btn = st.download_button(
                    label="Download Transcription",
                    data=file,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

            with open(summary_file_path, "rb") as file:
                btn = st.download_button(
                    label="Download Summary",
                    data=file,
                    file_name="summary_output.txt",
                    mime="text/plain"
                )
