# Voice-Emotion-and-sentimental-analysis-chat-bot
step 1 train the voice emotion dataset and dump the train model to the .h5 file format.
step 2 create the virtual envirnoment by opening the terminal and type the following cmd --python venv env then for the activation type in cmd >cd env
>cd Scripts
>./activate
this will actiavte your virtual envirnoment and now you are envirnoment is ready
> In cmd pip install -r requirements.txt
you should set your gemini api key
>then type python app.py 
my project Functionality

This project is basically an AI-powered empathetic voice chatbot.

It does:

Takes voice input (audio file uploaded in /process_audio).

Converts voice → text using speech_recognition.

Detects emotion from voice using a trained deep learning model (emotion.h5).

Analyzes sentiment of text using TextBlob.

Generates an empathetic AI response using Gemini API (gemini-1.5-flash model).

Converts AI’s response to speech using pyttsx3 (Text-to-Speech).

Stores everything (text, emotion, sentiment, duration, etc.) in SQLite database.

Provides APIs for:

Settings management (/settings, /toggle_auto_speak)

Audio generation (/generate_audio)

Fetch conversations (/get_conversations)

Statistics (/emotion_stats)

Available voices (/available_voices)

🔹 Breakdown of Key Features
1. Database (SQLite)

Stores:

Conversations (user text, bot response, emotion, sentiment, etc.)

User settings (voice type, auto-speak, volume, rate, etc.)

Tables:

conversations → history of chats

user_settings → persistent user preferences

2. Audio Handling

Accepts .webm file upload (recorded via browser mic usually).

Converts to .wav using ffmpeg.

Extracts MFCC audio features (librosa) for emotion classification.

Uses Google Speech Recognition (speech_recognition) to convert audio → text.

3. Emotion + Sentiment

Emotion detection = Deep learning model (emotion.h5 with MFCC input).

Sentiment analysis = TextBlob (positive / negative / neutral).

4. AI Response (Gemini API)

Takes user’s text, detected emotion, and sentiment.

Builds a context-aware prompt.

Calls Gemini model (gemini-1.5-flash) to generate a short empathetic reply.

5. Text-to-Speech (TTS)

Uses pyttsx3 to convert bot’s response to speech.

Supports changing:

Voice type (male / female / default)

Speed (rate)

Volume

Returns base64 encoded audio to client.

6. User Settings

Endpoints:

/settings (GET/POST) → read or update preferences

/toggle_auto_speak (POST) → turn TTS auto-play ON/OFF

/available_voices → list available voices from system

7. APIs Provided

/process_audio → Main endpoint to handle user voice input

/generate_audio → Convert any text → speech

/get_conversations → Retrieve last 50 chats

/emotion_stats → Summary of detected emotions/sentiments 
  
