import asyncio
import queue
import time
import logging
import numpy as np
import sounddevice as sd
import whisper
import librosa
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize
import langid
from gtts import gTTS
import io
import pygame
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from webrtcvad import Vad
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SPEECH_RATE = 160
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1024
MAX_AUDIO_QUEUE = 10
MIN_VOICE_ACTIVITY_DB = -40
VOICE_ACTIVITY_TIMEOUT = 2.0
ENERGY_THRESHOLD = 400
PAUSE_THRESHOLD = 0.8
SUPPORTED_LANGUAGES = ['en', 'hi', 'ta', 'te']  # English, Hindi, Tamil, Telugu

@dataclass
class VoiceState:
    is_speaking: bool = False
    last_voice_activity: float = 0
    speech_history: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=MAX_AUDIO_QUEUE))
    audio_params: Dict = field(default_factory=lambda: {
        'sample_rate': AUDIO_SAMPLE_RATE,
        'channels': 1,
        'energy_threshold': ENERGY_THRESHOLD
    })
    current_language: str = 'en'

class VoiceInterface:
    def __init__(self, config_path: str = "config/settings.json"):
        self.config = self.load_config(config_path)
        self.state = VoiceState()
        self.running = False
        self.audio_queue = queue.Queue()
        self.raw_audio_buffer = []
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Initialize audio
        pygame.init()
        pygame.mixer.init()
        
        # Initialize Whisper and VAD
        self.stt_model = whisper.load_model("medium")  # Better for Indian languages
        self.vad = Vad(mode=3)  # Aggressive VAD for noisy environments
        
        # Start workers
        asyncio.create_task(self.start_workers())

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from settings.json"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f).get('voice_interface', {})
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {
                'speech_rate': DEFAULT_SPEECH_RATE,
                'audio_sample_rate': AUDIO_SAMPLE_RATE,
                'energy_threshold': ENERGY_THRESHOLD,
                'pause_threshold': PAUSE_THRESHOLD,
                'voice_activity_timeout': VOICE_ACTIVITY_TIMEOUT,
                'default_language': 'en'
            }

    async def start_workers(self):
        """Start background tasks"""
        self.running = True
        asyncio.create_task(self.audio_capture_worker())
        asyncio.create_task(self.stt_processing_worker())
        asyncio.create_task(self.voice_activity_monitor())
        logger.info("Voice interface workers started")

    async def stop_workers(self):
        """Stop background tasks"""
        self.running = False
        pygame.mixer.music.stop()
        logger.info("Voice interface workers stopped")

    async def audio_capture_worker(self):
        """Capture audio with VAD"""
        while self.running:
            try:
                recording = sd.rec(
                    int(0.5 * AUDIO_SAMPLE_RATE),
                    samplerate=AUDIO_SAMPLE_RATE,
                    channels=1,
                    dtype='int16'
                )
                sd.wait()
                
                # Apply VAD
                is_speech = self.vad.is_speech(
                    (recording * 32767).tobytes(),
                    sample_rate=AUDIO_SAMPLE_RATE
                )
                
                if is_speech:
                    audio_np = recording.astype(np.float32) / 32768.0
                    self.audio_queue.put(audio_np)
                    with self.lock:
                        self.state.last_voice_activity = time.time()
                        if not self.state.is_speaking:
                            self.state.is_speaking = True
                            logger.info("Voice activity detected")
                
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Audio capture error: {e}")
                await asyncio.sleep(1)

    async def stt_processing_worker(self):
        """Process audio with Whisper"""
        while self.running:
            try:
                audio_np = await asyncio.get_event_loop().run_in_executor(None, self.audio_queue.get)
                if audio_np is not None:
                    enhanced_audio = self.enhance_audio(audio_np)
                    result = self.stt_model.transcribe(enhanced_audio, fp16=torch.cuda.is_available())
                    text = result['text'].strip()
                    
                    if text:
                        lang, _ = langid.classify(text)
                        if lang in SUPPORTED_LANGUAGES:
                            self.state.current_language = lang
                        logger.info(f"Recognized speech: {text} (lang: {self.state.current_language})")
                        
                        # Encrypt and store
                        encrypted_text = self.cipher.encrypt(text.encode())
                        with self.lock:
                            self.state.speech_history.put(encrypted_text)
                        
                        # Pass to convo_engine
                        from convo_engine import handle_user_input
                        await handle_user_input(text)
                
                self.audio_queue.task_done()
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"STT processing error: {e}")

    async def voice_activity_monitor(self):
        """Monitor voice activity"""
        while self.running:
            time_since_activity = time.time() - self.state.last_voice_activity
            with self.lock:
                if self.state.is_speaking and time_since_activity > self.config.get('voice_activity_timeout'):
                    self.state.is_speaking = False
                    logger.info("Voice activity ended")
            await asyncio.sleep(0.1)

    def enhance_audio(self, audio_np: np.ndarray) -> np.ndarray:
        """Enhance audio for Indian traffic noise"""
        try:
            # Custom noise profile for Indian traffic
            noise_profile = librosa.load("integrated_project/resources/indian_traffic_noise.wav")[0]
            reduced_noise = nr.reduce_noise(y=audio_np, sr=AUDIO_SAMPLE_RATE, y_noise=noise_profile)
            
            audio_segment = AudioSegment(
                (reduced_noise * 32767).astype(np.int16).tobytes(),
                frame_rate=AUDIO_SAMPLE_RATE,
                sample_width=2,
                channels=1
            )
            normalized = normalize(audio_segment)
            return np.frombuffer(normalized.raw_data, np.int16).astype(np.float32) / 32767
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")
            return audio_np

    async def speak(self, text: str):
        """Speak text using gTTS"""
        if not text or not isinstance(text, str):
            logger.warning("Invalid text input for TTS")
            return
        
        try:
            tts = gTTS(text=text, lang=self.state.current_language)
            with io.BytesIO() as f:
                tts.write_to_fp(f)
                f.seek(0)
                pygame.mixer.music.load(f)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
            logger.info(f"[TTS] Speaking: {text}")
            
            # Pause alerts during TTS
            from alert_system import stop_alerts
            await stop_alerts()
        except Exception as e:
            logger.error(f"TTS failed: {e}")

    async def is_user_speaking(self) -> bool:
        """Check if user is speaking"""
        with self.lock:
            return self.state.is_speaking

    async def get_latest_speech(self) -> Optional[str]:
        """Get latest recognized speech"""
        with self.lock:
            try:
                encrypted_text = self.state.speech_history.get_nowait()
                return self.cipher.decrypt(encrypted_text).decode()
            except queue.Empty:
                return None

    async def record_voice_sample(self, duration: float = 5.0) -> Optional[np.ndarray]:
        """Record voice sample in-memory"""
        try:
            logger.info(f"Recording voice sample for {duration} seconds...")
            recording = sd.rec(
                int(duration * AUDIO_SAMPLE_RATE),
                samplerate=AUDIO_SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            return recording
        except Exception as e:
            logger.error(f"Voice recording failed: {e}")
            return None

    async def say_awake_message(self, drowsiness_level: str = "normal"):
        """Deliver context-aware awake message"""
        messages = {
            "normal": {
                'en': ["Great! You're awake. Let's continue.", "Stay alert, you're doing well!"],
                'hi': ["Badhiya! Aap jaag rahe hain. Chalo aage badhein.", "Savdhan rahen, aap achha kar rahe hain!"],
                'ta': ["Nallatu! Nīṅkaḷ veḻiyāka irukkiṟīrkaḷ. Toṭaruvōm.", "Kaṇkaḷ veḻiyil vaittirungal!"],
                'te': ["Bagundi! Meeru melukundaga unnaru. Munduku vellandi.", "Jagrathaga undandi, meeru chala bagunnaru!"]
            },
            "extreme": {
                'en': ["You're awake now, but stay focused!", "Good, you're back! Keep alert."],
                'hi': ["Aap ab jaag gaye, lekin dhyan rakhen!", "Acha, aap wapas aaye! Savdhan rahen."],
                'ta': ["Nīṅkaḷ ippoḻutu veḻiyāka irukkiṟīrkaḷ, āṉāl kaṇkaḷ veḻiyil vaittirungal!", "Nallatu, nīṅkaḷ tirumpi vaṉtīrkaḷ!"],
                'te': ["Meeru ippudu melukundaga unnaru, kaani focus cheyandi!", "Bagundi, meeru tirigi vacharu! Jagrathaga undandi."]
            }
        }
        message = random.choice(messages.get(drowsiness_level, messages["normal"])[self.state.current_language])
        await self.speak(message)

    def get_voice_state(self) -> Dict[str, Any]:
        """Get current voice interface state"""
        with self.lock:
            return {
                "is_speaking": self.state.is_speaking,
                "last_activity": self.state.last_voice_activity,
                "speech_queue_size": self.state.speech_history.qsize(),
                "audio_params": self.state.audio_params,
                "language": self.state.current_language
            }

# Singleton instance
voice_interface = VoiceInterface()

async def start_voice_listener():
    """Initialize the voice interface"""
    await voice_interface.start_workers()

async def stop_voice_listener():
    """Stop the voice interface"""
    await voice_interface.stop_workers()

async def is_user_speaking() -> bool:
    """Check if user is speaking"""
    return await voice_interface.is_user_speaking()

async def get_latest_speech() -> Optional[str]:
    """Get latest recognized speech"""
    return await voice_interface.get_latest_speech()

async def say_awake_message(drowsiness_level: str = "normal"):
    """Deliver awake confirmation"""
    await voice_interface.say_awake_message(drowsiness_level)

async def speak(text: str):
    """Text-to-speech output"""
    await voice_interface.speak(text)

def get_voice_status() -> Dict[str, Any]:
    """Get voice interface status"""
    return voice_interface.get_voice_state()

async def main():
    """Test the voice interface"""
    await start_voice_listener()
    try:
        while voice_interface.running:
            status = get_voice_status()
            print(f"\nVoice Interface Status: Speaking={status['is_speaking']}, "
                  f"Last Activity={time.time() - status['last_activity']:.1f}s, "
                  f"Queue Size={status['speech_queue_size']}, "
                  f"Language={status['language']}")
            
            speech = await get_latest_speech()
            if speech:
                print(f"Recognized: {speech}")
                await say_awake_message("normal")
            
            await asyncio.sleep(1)
    finally:
        await stop_voice_listener()
        print("\nVoice interface stopped")

if __name__ == "__main__":
    asyncio.run(main())