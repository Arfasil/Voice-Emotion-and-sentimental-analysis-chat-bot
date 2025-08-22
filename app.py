from flask import Flask, request, jsonify, render_template
import sqlite3
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import io
import base64
import tempfile
import os
from datetime import datetime
import speech_recognition as sr
import pyttsx3
import threading
import google.generativeai as genai
from textblob import TextBlob
import json
from werkzeug.utils import secure_filename
import wave
import subprocess
import logging

app = Flask(__name__)

# Configuration
GEMINI_API_KEY = "AIzaSyAjwtPiX4ut2LDIjCpcEHzQDUAb1jZ3dbo"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load emotion detection model
try:
    emotion_model = keras.models.load_model('emotion.h5')
    logger.info("Emotion model loaded successfully")
except Exception as e:
    logger.error(f"Warning: Could not load emotion_model.h5: {e}")
    emotion_model = None

# Emotion labels - Updated to match your training script
EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Initialize text-to-speech engine
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 0.8)
    logger.info("TTS engine initialized successfully")
except Exception as e:
    logger.error(f"Warning: Could not initialize TTS engine: {e}")
    tts_engine = None

# Global settings for auto-speak functionality
user_settings = {
    'auto_speak_enabled': True,
    'speech_volume': 0.8,
    'speech_rate': 150,
    'voice_type': 'default'
}

# Database setup
def init_db():
    conn = sqlite3.connect('chatbot_data.db')
    cursor = conn.cursor()

    # Create conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_text TEXT,
            bot_response TEXT,
            detected_emotion TEXT,
            emotion_confidence REAL,
            sentiment_polarity REAL,
            sentiment_subjectivity REAL,
            sentiment_label TEXT
        )
    ''')

    # Create user settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_name TEXT UNIQUE,
            setting_value TEXT,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Try adding audio_duration if it doesn't exist
    try:
        cursor.execute('ALTER TABLE conversations ADD COLUMN audio_duration REAL')
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # Initialize default settings
    cursor.execute('''
        INSERT OR IGNORE INTO user_settings (setting_name, setting_value)
        VALUES 
            ('auto_speak_enabled', 'true'),
            ('speech_volume', '0.8'),
            ('speech_rate', '150'),
            ('voice_type', 'default')
    ''')

    conn.commit()
    conn.close()

def get_user_settings():
    """Retrieve user settings from database"""
    try:
        conn = sqlite3.connect('chatbot_data.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT setting_name, setting_value FROM user_settings')
        settings = cursor.fetchall()
        
        conn.close()
        
        # Convert to dictionary
        settings_dict = {}
        for name, value in settings:
            if name in ['auto_speak_enabled']:
                settings_dict[name] = value.lower() == 'true'
            elif name in ['speech_volume', 'speech_rate']:
                settings_dict[name] = float(value)
            else:
                settings_dict[name] = value
                
        return settings_dict
    except Exception as e:
        logger.error(f"Error getting user settings: {e}")
        return user_settings

def update_user_setting(setting_name, setting_value):
    """Update user setting in database"""
    try:
        conn = sqlite3.connect('chatbot_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_settings (setting_name, setting_value, last_updated)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (setting_name, str(setting_value)))
        
        conn.commit()
        conn.close()
        
        # Update global settings
        if setting_name == 'auto_speak_enabled':
            user_settings[setting_name] = setting_value
        elif setting_name in ['speech_volume', 'speech_rate']:
            user_settings[setting_name] = float(setting_value)
        else:
            user_settings[setting_name] = setting_value
            
        return True
    except Exception as e:
        logger.error(f"Error updating user setting: {e}")
        return False

def configure_tts_engine():
    """Configure TTS engine with current settings"""
    if tts_engine is None:
        return
        
    try:
        # Get current settings
        current_settings = get_user_settings()
        
        # Update TTS engine properties
        tts_engine.setProperty('rate', int(current_settings.get('speech_rate', 150)))
        tts_engine.setProperty('volume', float(current_settings.get('speech_volume', 0.8)))
        
        # Set voice type if specified
        voices = tts_engine.getProperty('voices')
        if voices and current_settings.get('voice_type') != 'default':
            voice_type = current_settings.get('voice_type', 'default')
            if voice_type == 'female':
                female_voices = [v for v in voices if 'female' in v.name.lower() or 'woman' in v.name.lower()]
                if female_voices:
                    tts_engine.setProperty('voice', female_voices[0].id)
            elif voice_type == 'male':
                male_voices = [v for v in voices if 'male' in v.name.lower() or 'man' in v.name.lower()]
                if male_voices:
                    tts_engine.setProperty('voice', male_voices[0].id)
                    
    except Exception as e:
        logger.error(f"Error configuring TTS engine: {e}")

def extract_audio_features(audio_data, sr=16000, n_mfcc=40):
    """Extract MFCC features from audio data to match training format"""
    try:
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        # Take mean across time dimension (same as training)
        mfcc_mean = np.mean(mfccs, axis=1)
        # Reshape to match model input: (1, n_mfcc, 1)
        features = mfcc_mean.reshape(1, n_mfcc, 1)
        return features
    except Exception as e:
        logger.error(f"Error extracting MFCC features: {e}")
        return np.zeros((1, n_mfcc, 1))

def convert_audio_to_wav(input_path, output_path):
    """Convert audio to WAV format using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', output_path, '-y'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        try:
            # Fallback: try with different parameters
            cmd = ['ffmpeg', '-i', input_path, output_path, '-y']
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except:
            return False
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return False

def predict_emotion(audio_features):
    """Predict emotion from MFCC features"""
    if emotion_model is None:
        return 'neutral', 0.5

    try:
        # Predict emotion
        prediction = emotion_model.predict(audio_features, verbose=0)
        emotion_id = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Get emotion label
        if emotion_id < len(EMOTION_LABELS):
            emotion = EMOTION_LABELS[emotion_id]
        else:
            emotion = 'neutral'
            
        return emotion, float(confidence)
    except Exception as e:
        logger.error(f"Error predicting emotion: {e}")
        return 'neutral', 0.5

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment label
        if polarity > 0.1:
            sentiment_label = 'positive'
        elif polarity < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
            
        return float(polarity), float(subjectivity), sentiment_label
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return 0.0, 0.0, 'neutral'

def generate_empathetic_response(user_text, emotion, sentiment):
    """Generate empathetic response using Gemini API"""
    try:
        # Create context-aware prompt based on emotion and sentiment
        emotion_context = {
            'happy': "The user sounds happy and excited. Match their positive energy and enthusiasm.",
            'sad': "The user sounds sad or down. Respond with empathy, comfort, and gentle encouragement.",
            'angry': "The user sounds frustrated or angry. Respond calmly and try to de-escalate while being understanding.",
            'fearful': "The user sounds worried or anxious. Respond with reassurance and support.",
            'surprised': "The user sounds surprised or amazed. Respond with curiosity and engagement.",
            'disgust': "The user sounds disgusted or displeased. Respond with understanding and try to address their concerns.",
            'calm': "The user sounds calm and composed. Respond in a balanced, thoughtful manner.",
            'neutral': "The user sounds neutral. Respond naturally and try to engage them positively."
        }
        
        prompt = f"""
        You are an empathetic AI voice assistant having a friendly conversation. 
        
        User's message: "{user_text}"
        Detected emotion from voice: {emotion}
        Text sentiment: {sentiment}
        
        Emotional context: {emotion_context.get(emotion, "Respond naturally and empathetically.")}
        
        Guidelines:
        - Respond in a warm, understanding, and conversational way
        - Keep your response concise (2-3 sentences max)
        - Match the user's emotional state appropriately
        - Be genuine and supportive
        - Since this is voice-based, use natural speech patterns
        - Ask engaging follow-up questions when appropriate
        
        Please provide a response that acknowledges their emotion and continues the conversation naturally.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error generating response with Gemini: {e}")
        # Fallback responses based on emotion
        fallback_responses = {
            'happy': "That's wonderful to hear! Your positive energy really comes through. What's been making you so happy today?",
            'sad': "I can hear that you might be going through a tough time. I'm here to listen and support you however I can.",
            'angry': "I can sense your frustration in your voice. It's completely valid to feel this way. Would you like to talk about what's troubling you?",
            'fearful': "I can hear some worry in your voice. It's okay to feel anxious sometimes, and I'm here to help you work through it.",
            'surprised': "You sound quite surprised! I'd love to hear more about what caught you off guard.",
            'disgust': "I can tell something is really bothering you. Your feelings are completely valid, and I'm here to listen.",
            'calm': "I appreciate your calm and thoughtful tone. It's nice to have a peaceful conversation with you.",
            'neutral': "Thanks for sharing that with me. I'm here and ready to chat about whatever's on your mind."
        }
        return fallback_responses.get(emotion, "I'm here to listen and chat with you. How are you feeling today?")

def text_to_speech(text, should_generate_audio=None):
    """Convert text to speech with auto-speak consideration"""
    # Check if we should generate audio
    if should_generate_audio is None:
        current_settings = get_user_settings()
        should_generate_audio = current_settings.get('auto_speak_enabled', True)
    
    # If auto-speak is disabled, don't generate audio
    if not should_generate_audio:
        return None
        
    if tts_engine is None:
        return None
        
    try:
        # Configure TTS engine with current settings
        configure_tts_engine()
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            
        # Generate speech
        tts_engine.save_to_file(text, temp_path)
        tts_engine.runAndWait()
        
        # Read audio file and encode to base64
        with open(temp_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
        # Clean up temp file
        os.unlink(temp_path)
        
        return audio_base64
        
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return None

def save_conversation(user_text, bot_response, emotion, emotion_confidence, 
                     sentiment_polarity, sentiment_subjectivity, sentiment_label, audio_duration=0):
    """Save conversation data to database"""
    try:
        conn = sqlite3.connect('chatbot_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (user_text, bot_response, detected_emotion, emotion_confidence, 
             sentiment_polarity, sentiment_subjectivity, sentiment_label, audio_duration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_text, bot_response, emotion, emotion_confidence, 
              sentiment_polarity, sentiment_subjectivity, sentiment_label, audio_duration))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        logger.info("Received audio processing request")

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio data received'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        logger.info(f"Processing audio file: {audio_file.filename}")

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_webm:
            audio_file.save(tmp_webm.name)
            webm_path = tmp_webm.name

        # Convert to WAV
        wav_path = webm_path.replace('.webm', '.wav')
        conversion_success = convert_audio_to_wav(webm_path, wav_path)

        if not conversion_success or not os.path.exists(wav_path):
            if os.path.exists(webm_path):
                os.remove(webm_path)
            return jsonify({'error': 'Audio conversion failed'}), 500

        try:
            # Load audio with librosa
            logger.info("Loading audio with librosa...")
            audio_data, sample_rate = librosa.load(wav_path, sr=16000)
            audio_duration = len(audio_data) / sample_rate
            logger.info(f"Audio loaded: duration={audio_duration:.2f}s, sr={sample_rate}")

            # Extract features and predict emotion
            logger.info("Extracting audio features...")
            features = extract_audio_features(audio_data, sample_rate)
            emotion, confidence = predict_emotion(features)
            logger.info(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")

            # Speech recognition
            recognizer = sr.Recognizer()
            user_text = ""

            try:
                with sr.AudioFile(wav_path) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_speech = recognizer.record(source)
                    user_text = recognizer.recognize_google(audio_speech)
                    logger.info(f"Recognized text: {user_text}")
            except sr.UnknownValueError:
                user_text = "I couldn't understand what you said, but I can sense your emotions from your voice."
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                user_text = "Speech recognition service is temporarily unavailable."

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(webm_path):
                os.remove(webm_path)

        # Analyze sentiment and generate response
        polarity, subjectivity, sentiment_label = analyze_sentiment(user_text)
        bot_response = generate_empathetic_response(user_text, emotion, sentiment_label)

        # Generate audio response based on current settings
        audio_response = text_to_speech(bot_response)

        # Save conversation to database
        save_conversation(user_text, bot_response, emotion, confidence,
                          polarity, subjectivity, sentiment_label, audio_duration)

        response_data = {
            'user_text': user_text,
            'bot_response': bot_response,
            'emotion': emotion,
            'emotion_confidence': confidence,
            'sentiment': {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'label': sentiment_label
            },
            'audio_duration': audio_duration,
            'auto_speak_enabled': user_settings.get('auto_speak_enabled', True)
        }

        if audio_response:
            response_data['audio_response'] = audio_response

        return jsonify(response_data)

    except Exception as e:
        import traceback
        logger.error(f"Error processing audio: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/settings', methods=['GET', 'POST'])
def handle_settings():
    """Handle user settings GET/POST requests"""
    if request.method == 'GET':
        try:
            settings = get_user_settings()
            return jsonify(settings)
        except Exception as e:
            logger.error(f"Error getting settings: {e}")
            return jsonify({'error': str(e)}), 500
            
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            # Update settings
            updated_settings = {}
            
            for setting_name, setting_value in data.items():
                if setting_name in ['auto_speak_enabled', 'speech_volume', 'speech_rate', 'voice_type']:
                    if update_user_setting(setting_name, setting_value):
                        updated_settings[setting_name] = setting_value
                    else:
                        return jsonify({'error': f'Failed to update {setting_name}'}), 500
                        
            return jsonify({
                'message': 'Settings updated successfully',
                'updated_settings': updated_settings
            })
            
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/toggle_auto_speak', methods=['POST'])
def toggle_auto_speak():
    """Toggle auto-speak functionality"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)
        
        if update_user_setting('auto_speak_enabled', enabled):
            return jsonify({
                'message': 'Auto-speak setting updated',
                'auto_speak_enabled': enabled
            })
        else:
            return jsonify({'error': 'Failed to update auto-speak setting'}), 500
            
    except Exception as e:
        logger.error(f"Error toggling auto-speak: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    """Generate audio for text on demand"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Force generate audio regardless of auto-speak setting
        audio_response = text_to_speech(text, should_generate_audio=True)
        
        if audio_response:
            return jsonify({
                'audio_response': audio_response,
                'message': 'Audio generated successfully'
            })
        else:
            return jsonify({'error': 'Failed to generate audio'}), 500
            
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_conversations', methods=['GET'])
def get_conversations():
    try:
        conn = sqlite3.connect('chatbot_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM conversations 
            ORDER BY timestamp DESC 
            LIMIT 50
        ''')
        
        conversations = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        conversation_list = []
        for conv in conversations:
            conversation_list.append({
                'id': conv[0],
                'timestamp': conv[1],
                'user_text': conv[2],
                'bot_response': conv[3],
                'detected_emotion': conv[4],
                'emotion_confidence': conv[5],
                'sentiment_polarity': conv[6],
                'sentiment_subjectivity': conv[7],
                'sentiment_label': conv[8],
                'audio_duration': conv[9] if conv[9] is not None else 0
            })
        
        return jsonify(conversation_list)
        
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/emotion_stats', methods=['GET'])
def emotion_stats():
    try:
        conn = sqlite3.connect('chatbot_data.db')
        cursor = conn.cursor()
        
        # Get emotion distribution
        cursor.execute('''
            SELECT detected_emotion, COUNT(*) as count, AVG(emotion_confidence) as avg_confidence
            FROM conversations
            GROUP BY detected_emotion
            ORDER BY count DESC
        ''')
        
        emotion_data = cursor.fetchall()
        emotion_stats = {}
        for row in emotion_data:
            emotion_stats[row[0]] = {
                'count': row[1], 
                'avg_confidence': round(row[2] * 100, 2)
            }
        
        # Get sentiment distribution
        cursor.execute('''
            SELECT sentiment_label, COUNT(*) as count, AVG(sentiment_polarity) as avg_polarity
            FROM conversations
            GROUP BY sentiment_label
            ORDER BY count DESC
        ''')
        
        sentiment_data = cursor.fetchall()
        sentiment_stats = {}
        for row in sentiment_data:
            sentiment_stats[row[0]] = {
                'count': row[1], 
                'avg_polarity': round(row[2], 2)
            }
        
        # Get total conversations
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'emotions': emotion_stats,
            'sentiments': sentiment_stats,
            'total_conversations': total_conversations
        })
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/available_voices', methods=['GET'])
def get_available_voices():
    """Get list of available TTS voices"""
    try:
        if tts_engine is None:
            return jsonify({'error': 'TTS engine not available'}), 500
            
        voices = tts_engine.getProperty('voices')
        voice_list = []
        
        for voice in voices:
            voice_info = {
                'id': voice.id,
                'name': voice.name,
                'gender': 'female' if any(keyword in voice.name.lower() 
                                        for keyword in ['female', 'woman', 'girl']) else 'male',
                'language': getattr(voice, 'languages', ['en'])[0] if hasattr(voice, 'languages') else 'en'
            }
            voice_list.append(voice_info)
            
        return jsonify({'voices': voice_list})
        
    except Exception as e:
        logger.error(f"Error getting available voices: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Load user settings
    user_settings.update(get_user_settings())
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)