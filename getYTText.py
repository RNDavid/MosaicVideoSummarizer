import yt_dlp
import assemblyai as aai

URL="https://www.youtube.com/watch?v=qSWvn5G1cdg"
aai.settings.api_key="966c4be2f05441a8a0ee7a74b8595b9e"

with yt_dlp.YoutubeDL() as ydl:
   info = ydl.extract_info(URL, download=False)

for format in info["formats"][::-1]:
    if format["resolution"] == "audio only" and format["ext"] == "m4a":
        url = format["url"]
        break
        
print(url)



# If the API key is not set as an environment variable named
# ASSEMBLYAI_API_KEY, you can also set it like this:
# aai.settings.api_key = "YOUR_API_KEY"

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(url)

print(transcript.text)
