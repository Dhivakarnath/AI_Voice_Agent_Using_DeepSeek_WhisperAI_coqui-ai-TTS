import streamlit as st
import whisper
from TTS.api import TTS
import ollama
import os
import pyaudio
import wave
import torch
import tempfile
import time
import asyncio
from pydub import AudioSegment
import torchaudio
import numpy as np
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="AI Voice Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fix for the "no running event loop" error
# Instead of trying to detect a running loop, use try/except for asyncio operations
# Removed: if not hasattr(asyncio, "_get_running_loop"): asyncio._get_running_loop = asyncio.get_running_loop

# Set PyTorch multiprocessing method
try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Set torchaudio backend explicitly to avoid warnings
torchaudio.set_audio_backend("sox_io")

# Available language options
LANGUAGES = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Chinese": "zh",
    "Russian": "ru",
    "Portuguese": "pt",
    "Arabic": "ar"
}

# TTS models for different languages
TTS_MODELS = {
    "en": "tts_models/en/ljspeech/tacotron2-DDC",
    "ta": "tts_models/ta/tamil/vits",
    "hi": "tts_models/hi/hindi/vits",
    "es": "tts_models/es/css10/vits",
    "fr": "tts_models/fr/css10/vits",
    "de": "tts_models/de/thorsten/tacotron2-DDC",
    "zh": "tts_models/zh-CN/baker/tacotron2-DDC-GST",    
    "default": "tts_models/en/ljspeech/tacotron2-DDC"
}

# Custom CSS for UI styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');

body {
    font-family: 'Roboto', sans-serif;
    background-color: #0e1117;
    color: #fafafa;
}

/* Wave animation for recording */
@keyframes wave {
  0% { height: 5px; }
  25% { height: 20px; }
  50% { height: 10px; }
  75% { height: 25px; }
  100% { height: 5px; }
}

.wave-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 60px;
    margin: 20px 0;
}

.wave-bar {
    width: 6px;
    height: 5px;
    background: linear-gradient(45deg, #4f8bf9, #1e88e5);
    margin: 0 3px;
    border-radius: 3px;
    animation: wave 1s ease-in-out infinite;
}

.wave-bar:nth-child(2n) {
    animation-delay: 0.1s;
}

.wave-bar:nth-child(3n) {
    animation-delay: 0.2s;
}

.wave-bar:nth-child(4n) {
    animation-delay: 0.3s;
}

.wave-bar:nth-child(5n) {
    animation-delay: 0.4s;
}

/* Chat container styling */
.chat-container {
    max-width: 90%;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 15px;
    font-size: 16px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.chat-container:hover {
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    transform: translateY(-2px);
}

.user-message {
    background: linear-gradient(135deg, #6e8efb, #376bd9);
    color: white;
    align-self: flex-end;
    animation: fadeIn 0.5s ease-in-out;
    margin-left: auto;
    margin-right: 10px;
    max-width: 80%;
}

.ai-message {
    background: linear-gradient(135deg, #2c3e50, #1a2533);
    color: #eee;
    align-self: flex-start;
    animation: fadeIn 0.5s ease-in-out;
    margin-right: auto;
    margin-left: 10px;
    max-width: 80%;
    border-left: 3px solid #1e88e5;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
  0% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.1); opacity: 0.8; }
  100% { transform: scale(1); opacity: 1; }
}

.pulse-animation {
    animation: pulse 1.5s infinite;
    display: inline-block;
    font-size: 20px;
    margin-left: 8px;
}

.highlight {
    background-color: rgba(255, 255, 0, 0.3);
    padding: 0 3px;
    border-radius: 3px;
    transition: background-color 0.5s ease;
}

/* Controls styling */
.stButton>button {
    background: linear-gradient(135deg, #1e88e5, #4f8bf9);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 12px 24px;
    font-weight: 500;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}

.stButton>button:active {
    transform: translateY(0);
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.stSelectbox > div > div {
    background-color: #1e293b;
    color: white;
    border-radius: 10px;
}

.stSlider > div {
    color: #1e88e5;
}

/* Create a card effect for main sections */
.main-section {
    background-color: #1a1f29;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #171c26;
}

/* Custom styling for chat history section */
.chat-history {
    max-height: 500px;
    overflow-y: auto;
    padding: 10px;
    background-color: #171c26;
    border-radius: 15px;
    margin-bottom: 20px;
}

/* Loading indicator */
@keyframes typing {
  0% { content: ""; }
  25% { content: "."; }
  50% { content: ".."; }
  75% { content: "..."; }
  100% { content: ""; }
}

.typing-animation::after {
  content: "";
  animation: typing 1.5s infinite;
}

/* Voice button pulse effect */
@keyframes voicePulse {
  0% { box-shadow: 0 0 0 0 rgba(78, 137, 248, 0.7); }
  70% { box-shadow: 0 0 0 15px rgba(78, 137, 248, 0); }
  100% { box-shadow: 0 0 0 0 rgba(78, 137, 248, 0); }
}

.voice-button {
    animation: voicePulse 2s infinite;
    border-radius: 50%;
    width: 80px;
    height: 80px;
    display: flex;
    justify-content: center;
    align-items: center;
    background: linear-gradient(135deg, #1e88e5, #4f8bf9);
    color: white;
    font-size: 30px;
    margin: 0 auto;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# JavaScript for word highlighting during speech - simplified to avoid errors
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Simple function to highlight words during TTS playback
    function setupHighlighting() {
        const audioElements = document.querySelectorAll('audio');
        audioElements.forEach(audio => {
            audio.addEventListener('play', function() {
                const messageId = this.getAttribute('data-message-id');
                if (messageId) {
                    const words = document.querySelectorAll(`#${messageId} .highlight-word`);
                    if (words.length) {
                        let wordIndex = 0;
                        const duration = this.duration * 1000;
                        const wordsCount = words.length;
                        const interval = duration / wordsCount;
                        
                        const highlightInterval = setInterval(() => {
                            words.forEach(w => w.classList.remove('highlight'));
                            if (wordIndex < wordsCount) {
                                words[wordIndex].classList.add('highlight');
                                wordIndex++;
                            } else {
                                clearInterval(highlightInterval);
                            }
                        }, interval);
                        
                        this.addEventListener('pause', () => clearInterval(highlightInterval), {once: true});
                        this.addEventListener('ended', () => clearInterval(highlightInterval), {once: true});
                    }
                }
            });
        });
    }
    
    // Check for elements periodically (Streamlit dynamically adds elements)
    setInterval(setupHighlighting, 1000);
});
</script>
""", unsafe_allow_html=True)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "recording" not in st.session_state:
    st.session_state.recording = False
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"
if "tts_speed" not in st.session_state:
    st.session_state.tts_speed = 1.0
if "voice_type" not in st.session_state:
    st.session_state.voice_type = "Default"
if "audio_visualizer" not in st.session_state:
    st.session_state.audio_visualizer = []

# Function to generate audio wave visualization placeholder
def generate_audio_wave(bars=20):
    wave_html = '<div class="wave-container">'
    for i in range(bars):
        delay = i * 0.05
        wave_html += f'<div class="wave-bar" style="animation-delay: {delay}s"></div>'
    wave_html += '</div>'
    return wave_html

# Function to create highlight-able text for TTS
def create_highlightable_text(text, message_id):
    words = text.split()
    html = f'<div id="{message_id}">'
    for i, word in enumerate(words):
        html += f'<span class="highlight-word" id="{message_id}-word-{i}">{word}</span> '
    html += '</div>'
    return html

class AIVoiceAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load models lazily to prevent startup errors
        self._whisper_model = None
        self.tts_models = {}
        
        # Audio recording setup
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.p = pyaudio.PyAudio()
    
    @property
    def whisper_model(self):
        # Lazy loading of whisper model
        if self._whisper_model is None:
            st.info("Loading speech recognition model... This may take a moment.")
            self._whisper_model = whisper.load_model("small").to(self.device)
        return self._whisper_model

    def get_tts_model(self, lang_code):
        # Load or get TTS model for the specified language
        if lang_code not in self.tts_models:
            model_name = TTS_MODELS.get(lang_code, TTS_MODELS["default"])
            try:
                self.tts_models[lang_code] = TTS(model_name=model_name, gpu=(self.device=="cuda"))
            except Exception as e:
                st.error(f"Error loading TTS model for {lang_code}: {e}")
                # Fallback to English if other language fails
                if lang_code != "en" and "en" in self.tts_models:
                    return self.tts_models["en"]
                else:
                    # Try to load English model if no models are loaded
                    self.tts_models["en"] = TTS(model_name=TTS_MODELS["en"], gpu=(self.device=="cuda"))
                    return self.tts_models["en"]
        return self.tts_models[lang_code]

    def visualize_audio(self, audio_data, num_bars=20):
        # Convert audio data to numpy array
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            # Normalize and chunk the audio data for visualization
            audio_array = np.abs(audio_array)
            max_val = np.max(audio_array) if np.max(audio_array) > 0 else 1
            audio_array = audio_array / max_val
            
            # Chunk the array for visualization
            chunk_size = len(audio_array) // num_bars
            if chunk_size > 0:
                chunks = [np.max(audio_array[i:i+chunk_size]) for i in range(0, len(audio_array), chunk_size)][:num_bars]
                return chunks
            return [0.1] * num_bars
        except Exception as e:
            print(f"Error in audio visualization: {e}")
            return [0.1] * num_bars

    def record_audio(self, duration=5):
        try:
            stream = self.p.open(format=self.audio_format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
            frames = []
            
            # Recording animation placeholder
            wave_placeholder = st.empty()
            for i in range(int(self.rate / self.chunk * duration)):
                try:
                    data = stream.read(self.chunk)
                    frames.append(data)
                    
                    # Update visualization every few chunks to save processing
                    if i % 5 == 0:
                        current_audio = b"".join(frames)
                        visualization = self.visualize_audio(current_audio)
                        st.session_state.audio_visualizer = visualization
                        wave_html = generate_audio_wave()
                        wave_placeholder.markdown(wave_html, unsafe_allow_html=True)
                except IOError as e:
                    # Handle audio input errors gracefully
                    print(f"Audio input error: {e}")
                    continue
                    
            stream.stop_stream()
            stream.close()
            wave_placeholder.empty()
            return b"".join(frames)
        except Exception as e:
            st.error(f"Error recording audio: {e}")
            return b""

    def transcribe_audio(self, audio_data, language_code="en"):
        if not audio_data:
            return ""
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            temp_path = tmp_audio.name
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(audio_data)
        try:
            result = self.whisper_model.transcribe(temp_path, language=language_code)
            return result["text"].strip()
        except Exception as e:
            st.error(f"Error in transcription: {e}")
            return ""
        finally:
            try:
                os.remove(temp_path)
            except:
                pass

    def generate_ai_response(self, text, language_code="en"):
        message_id = f"msg-{len(st.session_state.conversation)}"
        st.session_state.conversation.append({"role": "user", "content": text, "id": message_id})
        
        response_placeholder = st.empty()
        ai_response = ""
        
        # Show typing animation
        response_placeholder.markdown(
            f"<div class='chat-container ai-message'><span class='typing-animation'>Thinking</span></div>",
            unsafe_allow_html=True
        )
        
        try:
            # Use a safer approach to stream from ollama
            try:
                ollama_stream = ollama.chat(
                    model="deepseek-r1:7b",
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.conversation],
                    stream=True,
                    options={"use_gpu": True}
                )
                
                # For streamed response with typing effect
                for chunk in ollama_stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        response_text = chunk['message']['content']
                        ai_response += response_text
                        response_placeholder.markdown(
                            f"<div class='chat-container ai-message'>{ai_response}<span class='pulse-animation'>🔊</span></div>",
                            unsafe_allow_html=True
                        )
                        time.sleep(0.03)  # Slightly faster typing effect
            except Exception as e:
                # Fallback to non-streaming if streaming fails
                st.warning(f"Streaming response failed, falling back to standard response: {e}")
                response = ollama.chat(
                    model="deepseek-r1:7b",
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.conversation],
                    options={"use_gpu": True}
                )
                ai_response = response['message']['content']
                response_placeholder.markdown(
                    f"<div class='chat-container ai-message'>{ai_response}<span class='pulse-animation'>🔊</span></div>",
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error in AI response: {e}")
            ai_response = "Sorry, something went wrong with the AI model. Please try again or check if the DeepSeek R1 model is properly installed in Ollama."
            response_placeholder.markdown(
                f"<div class='chat-container ai-message'>{ai_response}</div>",
                unsafe_allow_html=True
            )
            
        response_id = f"msg-{len(st.session_state.conversation)}"
        st.session_state.conversation.append({"role": "assistant", "content": ai_response, "id": response_id})
        return ai_response, response_id

    def speak_response(self, text, language_code="en", speed=1.0):
        try:
            # FIX: Check if text is too short and pad it if necessary
            if len(text.strip()) < 10:
                # Add padding spaces to short text to avoid kernel size error
                padded_text = text.strip() + " " * 20
            else:
                padded_text = text
            
            tts_model = self.get_tts_model(language_code)
            speech_file = os.path.join(tempfile.gettempdir(), f"response_{int(time.time())}.wav")
            
            # Split very long text into paragraphs to process separately
            # This avoids potential memory issues with very long inputs
            if len(padded_text) > 1000:
                paragraphs = self.split_into_paragraphs(padded_text)
                audio_segments = []
                
                for para in paragraphs:
                    if len(para.strip()) < 5:  # Skip very short paragraphs
                        continue
                        
                    # Add padding to each paragraph if needed
                    if len(para.strip()) < 10:
                        para = para.strip() + " " * 20
                        
                    temp_file = os.path.join(tempfile.gettempdir(), f"temp_{int(time.time())}_{len(para)}.wav")
                    try:
                        tts_model.tts_to_file(text=para, file_path=temp_file, speed=speed)
                        audio_segments.append(AudioSegment.from_wav(temp_file))
                    except Exception as e:
                        print(f"Error in TTS for paragraph: {e}")
                        continue
                    finally:
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        except:
                            pass
                
                if audio_segments:
                    combined = audio_segments[0]
                    for segment in audio_segments[1:]:
                        combined += segment
                    combined.export(speech_file, format="wav")
                else:
                    return None
            else:
                try:
                    tts_model.tts_to_file(text=padded_text, file_path=speech_file, speed=speed)
                except Exception as e:
                    # Try with even more padding if still fails
                    st.warning(f"TTS padding needed: {e}")
                    extra_padded_text = padded_text + " " * 50
                    tts_model.tts_to_file(text=extra_padded_text, file_path=speech_file, speed=speed)
            
            return speech_file
        except Exception as e:
            st.error(f"Error in text-to-speech: {e}")
            # Fallback to default model if language-specific one fails
            if language_code != "en":
                try:
                    return self.speak_response(text, "en", speed)
                except:
                    return None
            return None
            
    def split_into_paragraphs(self, text, max_length=200):
        """Split long text into paragraphs to make TTS processing more manageable"""
        # First try to split by new lines
        paragraphs = text.split('\n')
        result = []
        
        for para in paragraphs:
            # If paragraph is still too long, split by sentences
            if len(para) > max_length:
                sentences = para.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
                current = ""
                
                for sentence in sentences:
                    if len(current) + len(sentence) <= max_length:
                        current += sentence + " "
                    else:
                        if current:
                            result.append(current.strip())
                        current = sentence + " "
                
                if current:
                    result.append(current.strip())
            else:
                result.append(para)
                
        # If no paragraphs were created (e.g., no newlines and no sentence breaks),
        # then just chunk by max_length
        if not result:
            for i in range(0, len(text), max_length):
                result.append(text[i:i+max_length])
                
        return result

# Initialize AI Agent
@st.cache_resource
def get_agent():
    return AIVoiceAgent()

agent = get_agent()

# App header with gradient background
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3c72, #2a5298); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
    <h1 style="color: white; text-align: center;">🎙️ AI Voice Agent</h1>
    <p style="color: white; text-align: center; opacity: 0.8;">Talk to your AI assistant with natural voice interactions</p>
</div>
""", unsafe_allow_html=True)

# Create a two-column layout
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    
    # Language selection
    selected_language = st.selectbox(
        "Select Language",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.selected_language),
        key="language_selector"
    )
    st.session_state.selected_language = selected_language
    language_code = LANGUAGES[selected_language]
    
    # Voice speed slider
    tts_speed = st.slider(
        "Voice Speed",
        min_value=0.5,
        max_value=1.5,
        value=st.session_state.tts_speed,
        step=0.1,
        key="speed_slider"
    )
    st.session_state.tts_speed = tts_speed
    
    # Optional: Voice type selection (if available)
    voice_types = ["Default", "Female", "Male"]
    voice_type = st.selectbox(
        "Voice Type",
        options=voice_types,
        index=voice_types.index(st.session_state.voice_type),
        key="voice_selector"
    )
    st.session_state.voice_type = voice_type
    
    # Recording duration
    record_duration = st.slider(
        "Recording Duration (seconds)",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
        key="duration_slider"
    )
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation = []
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display current status
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Status")
    st.markdown(f"**Active Model:** DeepSeek R1 7B")
    st.markdown(f"**Language:** {selected_language}")
    st.markdown(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Conversation statistics
    if st.session_state.conversation:
        user_msgs = len([m for m in st.session_state.conversation if m["role"] == "user"])
        ai_msgs = len([m for m in st.session_state.conversation if m["role"] == "assistant"])
        
        st.markdown(f"**Messages:** {len(st.session_state.conversation)}")
        st.markdown(f"**User Messages:** {user_msgs}")
        st.markdown(f"**AI Responses:** {ai_msgs}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col1:
    # Chat history with animations
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    
    if not st.session_state.conversation:
        st.markdown("""
        <div style="text-align: center; padding: 50px 20px; color: #888;">
            <span style="font-size: 40px;">👋</span>
            <p>Your conversation will appear here. Click the microphone button below to start talking!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.conversation:
            role_class = "user-message" if msg["role"] == "user" else "ai-message"
            
            if msg["role"] == "assistant":
                # Create highlightable text for TTS
                highlightable_text = create_highlightable_text(msg["content"], msg["id"])
                st.markdown(f"<div class='chat-container {role_class}'>{highlightable_text}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-container {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Audio wave visualization during recording
    wave_container = st.empty()
    
    # Center the recording button
    st.markdown('<div style="display: flex; justify-content: center; margin: 20px 0;">', unsafe_allow_html=True)
    
    # Recording button with animation
    if st.button("🎤 Start Listening", key="record_button", use_container_width=True):
        wave_container.markdown(generate_audio_wave(), unsafe_allow_html=True)
        
        with st.spinner("Listening..."):
            audio_data = agent.record_audio(duration=record_duration)
            
        wave_container.empty()
        
        with st.spinner("Transcribing..."):
            user_text = agent.transcribe_audio(audio_data, language_code=language_code)
            
        if user_text:
            st.markdown(f"<div class='chat-container user-message'>{user_text}</div>", unsafe_allow_html=True)
            
            with st.spinner("Generating response..."):
                ai_text, response_id = agent.generate_ai_response(user_text, language_code=language_code)
                
            with st.spinner("Converting to speech..."):
                audio_path = agent.speak_response(ai_text, language_code=language_code, speed=tts_speed)
                
                if audio_path:
                    try:
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                        
                        # Add data-message-id attribute to link audio with text for highlighting
                        audio_html = f"""
                        <audio data-message-id="{response_id}" controls autoplay>
                          <source src="data:audio/wav;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/wav">
                        </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error playing audio: {e}")
        else:
            st.warning("I didn't catch that. Please try speaking again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Text input as an alternative to voice
    st.markdown('<div class="main-section">', unsafe_allow_html=True)
    user_input = st.text_input("Or type your message:", key="text_input")
    
    if user_input:
        st.markdown(f"<div class='chat-container user-message'>{user_input}</div>", unsafe_allow_html=True)
        
        with st.spinner("Generating response..."):
            ai_text, response_id = agent.generate_ai_response(user_input, language_code=language_code)
            
        with st.spinner("Converting to speech..."):
            audio_path = agent.speak_response(ai_text, language_code=language_code, speed=tts_speed)
            
            if audio_path:
                try:
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
                    
                    # Add data-message-id attribute
                    audio_html = f"""
                    <audio data-message-id="{response_id}" controls autoplay>
                      <source src="data:audio/wav;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/wav">
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error playing audio: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with app information
st.markdown("""
<div style="background-color: #171c26; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center;">
    <p style="margin: 0; opacity: 0.7;">DeepSeek R1 AI Voice Assistant • Powered by Whisper & Ollama</p>
</div>
""", unsafe_allow_html=True)