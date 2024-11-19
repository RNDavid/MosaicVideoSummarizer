
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import tempfile
import whisper
from pytubefix import YouTube
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

api_key= os.getenv("OPENAI_API_KEY")

YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=qSWvn5G1cdg"

model = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")


# Let's do this only if we haven't created the transcription file yet.
if not os.path.exists("transcription.txt"):
    youtube = YouTube(YOUTUBE_VIDEO)
    audio = youtube.streams.filter(only_audio=True).first()

    # Let's load the base model. This is not the most accurate
    # model but it's fast.
    whisper_model = whisper.load_model("base")

    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

        with open("transcription.txt", "w") as file:
            file.write(transcription)


with open("transcription.txt") as file:
    transcription = file.read()

transcription[:100]

loader = TextLoader("transcription.txt")
text_documents = loader.load()
text_documents

extra_terms="Lotte"
prompt_template="""Summarize the following text: {transcription}.  Please also call out any reference to {extra_terms}."""
llm=model
prompt=PromptTemplate(template=prompt_template, input_variables=["transcription", "extra_terms"])

#summary_chain = LLMChain(llm=model, prompt="Summarize the following text: {input_text}")
summary_chain = LLMChain(prompt=prompt, llm=llm)
#summary_chain = prompt | llm

output=summary_chain.invoke({'transcription': transcription, 'extra_terms': extra_terms})

if 'text' in output:
    summary_text = output['text']
    print("Summary:", summary_text)
 
    # Specify the path where the summary should be saved
    summary_file_path = 'summary_output.txt'

with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
    summary_file.write(summary_text)
    print(f"Summary has been written to {summary_file_path}.")
    

