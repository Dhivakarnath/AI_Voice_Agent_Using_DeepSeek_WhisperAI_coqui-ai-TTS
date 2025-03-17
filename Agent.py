#Completely Offline Approach

import whisper
from TTS.api import TTS
import ollama
import os
import pyaudio
import wave
import torch
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import tempfile

class AIVoiceAgent:
    def __init__(self):
        """ Initializes the AI Voice Agent """
        print("\n🔄 Initializing AI Voice Agent...")

        # ✅ Ensure Whisper uses GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Using device:', device)
        self.whisper_model = whisper.load_model("small").to("cuda") # Load Whisper on GPU 
        
        # ✅ Initialize Coqui TTS for offline text-to-speech
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

        # ✅ Conversation history for agentic behavior
        self.full_transcript = [
            {"role": "system", "content": "You are DeepSeek R1. Answer concisely within 300 characters."},
        ]

        # ✅ Set up microphone stream
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper prefers 16kHz
        self.chunk = 1024
        self.p = pyaudio.PyAudio()

        # ✅ Set up a writable temporary directory
        self.temp_dir = "D:/AI_Voice_Agent/temp"
        os.makedirs(self.temp_dir, exist_ok=True)

    def start_transcription(self):
        """ Listens and transcribes user speech, then generates AI response """
        print("\n🎤 Listening... Speak now.")
        audio_data = self.record_audio()
        text = self.transcribe_audio(audio_data)

        if text:
            print(f"\n🗣 User: {text}")
            self.generate_ai_response(text)

    def record_audio(self, duration=5):
        """ Records audio from the microphone """
        stream = self.p.open(format=self.audio_format,
                             channels=self.channels,
                             rate=self.rate,
                             input=True,
                             frames_per_buffer=self.chunk)

        frames = []
        for _ in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        return b"".join(frames)

    def transcribe_audio(self, audio_data):
        """ Converts recorded audio to text using Whisper (English only) """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=self.temp_dir) as tmp_audio:
            temp_path = tmp_audio.name  # Store the file path before closing

            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(audio_data)

        try:
            result = self.whisper_model.transcribe(temp_path, language="en")  # ✅ Force English transcription
            return result["text"].strip()
        except Exception as e:
            print(f"❌ Error in transcription: {e}")
            return ""
        finally:
            os.remove(temp_path)  # ✅ Manually delete temp file to prevent permission errors

    def generate_ai_response(self, text):
        """ Sends transcribed text to DeepSeek R1 for response generation """
        self.full_transcript.append({"role": "user", "content": text})

        ollama_stream = ollama.chat(
            model="deepseek-r1:7b",
            messages=self.full_transcript,
            stream=True,
            options={"use_gpu": True},  # ✅ Force Ollama to use GPU
        )

        print("\n🤖 DeepSeek R1: ", end="")  # Print in the same line
        full_text = ""

        for chunk in ollama_stream:
            response_text = chunk['message']['content']
            full_text += response_text

        print(full_text)  # ✅ Print entire response at once
        self.speak_response(full_text)

        self.full_transcript.append({"role": "assistant", "content": full_text})
        self.start_transcription()  # Continue conversation loop

    def speak_response(self, text):
        """ Convert AI response to speech and play it using pydub """
        speech_file = os.path.join(self.temp_dir, "response.wav")
        self.tts.tts_to_file(text=text, file_path=speech_file)

        # ✅ Play the generated speech
        audio = AudioSegment.from_wav(speech_file)
        play(audio)

# ✅ Run the AI Voice Agent
if __name__ == "__main__":
    ai_voice_agent = AIVoiceAgent()
    ai_voice_agent.start_transcription()
