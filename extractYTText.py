import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
import nltk.data
import nltk
nltk.download('punkt')
import os
#from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
#from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

#OpenAI
api_key= os.environ["OPENAI_API_KEY"]


video_url = "https://www.youtube.com/watch?v=qSWvn5G1cdg"

# Get the video ID from the URL
video_id = video_url.split('v=')[-1].split('&')[0]
print(video_id)
# Retrieve the transcript
transcript = YouTubeTranscriptApi.get_transcript(video_id)

text = ''
for entry in transcript:
    text += entry['text']
print(text)

def chunk_text_into_sentences(text, max_chars):
    # Initialize the sentence tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Tokenize the input text into sentences
    sentences = tokenizer.tokenize(text)

    # Initialize variables
    current_chunk = ""
    chunks = []

    for sentence in sentences:
        # Check if adding the sentence to the current chunk would exceed the character limit
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            # If adding the sentence exceeds the limit, start a new chunk
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
# Check if the transcript length is too long
max_chars_per_chunk = 3000 * 4
if len(text) > max_chars_per_chunk:
    summarized_text_chunks = chunk_text_into_sentences(text, max_chars_per_chunk)
else:
    summarized_text_chunks = [text]

llm = ChatOpenAI(
    model_name="gpt-4", 
    openai_api_key=api_key,
    model_kwargs={"temperature": 0}
)
prompt = ChatPromptTemplate.from_template(
  '''
  Provide a concise summary of the following YouTube video transcript, highlighting the main takeaways:
    {transcript}
    '''
)
chain = LLMChain(llm=llm, prompt=prompt)
summary = ''
for chunk in summarized_text_chunks:
    result = chain.invoke({"transcript": chunk})
    summary += result["text"]  # Extract the summary text from the dictionary

print(summary)

