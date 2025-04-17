import speech_recognition as sr
import pyttsx3
import threading
import time
import queue
import whisper
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import sounddevice as sd
import wave
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize
import librosa
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SPEECH_RATE = 160
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1024
MAX_AUDIO_QUEUE = 10
MIN_VOICE_ACTIVITY_DB = -40
VOICE_ACTIVITY_TIMEOUT = 2.0  # seconds
ENERGY_THRESHOLD = 400  # For VAD
PAUSE_THRESHOLD = 0.8  # seconds of silence to end phrase

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

class VoiceInterface:
    def __init__(self):
        # Initialize components
        self.engine = pyttsx3.init()
        self.configure_tts()
        
        # Voice activity detection
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.recognizer.pause_threshold = PAUSE_THRESHOLD
        self.mic = self.configure_microphone()
        
        # Speech-to-text model
        self.stt_model = whisper.load_model("base")  # Can be small/medium/large
        self.running = False
        
        # State management
        self.state = VoiceState()
        self.lock = threading.Lock()
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.raw_audio_buffer = []
        
        # Start workers
        self.start_workers()

    def configure_tts(self):
        """Configure text-to-speech engine"""
        self.engine.setProperty('rate', DEFAULT_SPEECH_RATE)
        self.engine.setProperty('volume', 1.0)
        voices = self.engine.getProperty('voices')
        # Select a more natural sounding voice if available
        for voice in voices:
            if 'english' in voice.languages and 'female' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        else:
            self.engine.setProperty('voice', voices[0].id)

    def configure_microphone(self):
        """Configure microphone with error handling"""
        try:
            mic_list = sr.Microphone.list_microphone_names()
            logger.info(f"Available microphones: {mic_list}")
            
            # Try to find a good quality microphone
            preferred_mics = ['default', 'pulse', 'microphone', 'input']
            for mic_name in preferred_mics:
                for index, name in enumerate(mic_list):
                    if mic_name in name.lower():
                        logger.info(f"Selected microphone: {name}")
                        return sr.Microphone(device_index=index)
            
            return sr.Microphone()
        except Exception as e:
            logger.error(f"Microphone configuration failed: {e}")
            return None

    def start_workers(self):
        """Start background worker threads"""
        self.running = True
        # Audio capture thread
        threading.Thread(target=self.audio_capture_worker, daemon=True).start()
        # STT processing thread
        threading.Thread(target=self.stt_processing_worker, daemon=True).start()
        # Voice activity monitoring
        threading.Thread(target=self.voice_activity_monitor, daemon=True).start()
        logger.info("Voice interface workers started")

    def stop_workers(self):
        """Stop background workers"""
        self.running = False
        logger.info("Voice interface workers stopped")

    def audio_capture_worker(self):
        """Continuously capture audio from microphone"""
        while self.running and self.mic:
            try:
                with self.mic as source:
                    logger.debug("Adjusting for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    logger.debug("Listening...")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    # Convert to numpy array for processing
                    wav_data = audio.get_wav_data()
                    audio_np = np.frombuffer(wav_data, np.int16).astype(np.float32) / 32768.0
                    
                    # Put in queue for processing
                    self.audio_queue.put(audio_np)
                    
                    # Update voice activity
                    with self.lock:
                        self.state.last_voice_activity = time.time()
                        if not self.state.is_speaking:
                            self.state.is_speaking = True
                            logger.info("Voice activity detected")

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audio capture error: {e}")
                time.sleep(1)

    def stt_processing_worker(self):
        """Process audio chunks with STT model"""
        while self.running:
            try:
                audio_np = self.audio_queue.get(timeout=1)
                if audio_np is not None:
                    # Enhance audio quality
                    enhanced_audio = self.enhance_audio(audio_np)
                    
                    # Transcribe
                    result = self.stt_model.transcribe(enhanced_audio, fp16=False)
                    text = result['text'].strip()
                    
                    if text:
                        logger.info(f"Recognized speech: {text}")
                        with self.lock:
                            self.state.speech_history.put(text)
                    
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"STT processing error: {e}")

    def voice_activity_monitor(self):
        """Monitor voice activity and update speaking state"""
        while self.running:
            time_since_activity = time.time() - self.state.last_voice_activity
            with self.lock:
                if self.state.is_speaking and time_since_activity > VOICE_ACTIVITY_TIMEOUT:
                    self.state.is_speaking = False
                    logger.info("Voice activity ended")
            time.sleep(0.1)

    def enhance_audio(self, audio_np: np.ndarray) -> np.ndarray:
        """Apply audio enhancement techniques"""
        try:
            # Noise reduction
            reduced_noise = nr.reduce_noise(y=audio_np, sr=AUDIO_SAMPLE_RATE)
            
            # Normalize volume
            audio_segment = AudioSegment(
                reduced_noise.tobytes(),
                frame_rate=AUDIO_SAMPLE_RATE,
                sample_width=4,  # float32
                channels=1
            )
            normalized = normalize(audio_segment)
            
            return np.frombuffer(normalized.raw_data, np.float32)
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")
            return audio_np

    def speak(self, text: str):
        """Convert text to speech with logging"""
        logger.info(f"[TTS] Speaking: {text}")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS failed: {e}")

    def is_user_speaking(self) -> bool:
        """Check if user is currently speaking"""
        with self.lock:
            return self.state.is_speaking

    def get_latest_speech(self) -> Optional[str]:
        """Get the latest recognized speech"""
        with self.lock:
            try:
                return self.state.speech_history.get_nowait()
            except queue.Empty:
                return None

    def record_voice_sample(self, duration: float = 5.0) -> Optional[str]:
        """Record a voice sample to file"""
        try:
            logger.info(f"Recording voice sample for {duration} seconds...")
            recording = sd.rec(
                int(duration * AUDIO_SAMPLE_RATE),
                samplerate=AUDIO_SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_sample_{timestamp}.wav"
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes((recording * 32767).astype(np.int16))
            
            return filename
        except Exception as e:
            logger.error(f"Voice recording failed: {e}")
            return None

    def say_awake_message(self):
        """Deliver an awake confirmation message"""
        messages = [
            "Great! You're still awake. Let's continue.",
            "Excellent! Stay alert and focused.",
            "Good job staying awake! Keep it up.",
            "You're doing well maintaining your focus.",
            "Alertness confirmed. Let's keep going."
        ]
        self.speak(random.choice(messages))

    def get_voice_state(self) -> Dict[str, Any]:
        """Get current voice interface state"""
        with self.lock:
            return {
                "is_speaking": self.state.is_speaking,
                "last_activity": self.state.last_voice_activity,
                "speech_queue_size": self.state.speech_history.qsize(),
                "audio_params": self.state.audio_params
            }


# Singleton instance
voice_interface = VoiceInterface()

def start_voice_listener():
    """Initialize the voice interface"""
    voice_interface.start_workers()

def stop_voice_listener():
    """Stop the voice interface"""
    voice_interface.stop_workers()

def is_user_speaking() -> bool:
    """Check if user is speaking"""
    return voice_interface.is_user_speaking()

def get_latest_speech() -> Optional[str]:
    """Get latest recognized speech"""
    return voice_interface.get_latest_speech()

def say_awake_message():
    """Deliver awake confirmation"""
    voice_interface.say_awake_message()

def speak(text: str):
    """Text-to-speech output"""
    voice_interface.speak(text)

def get_voice_status() -> Dict[str, Any]:
    """Get voice interface status"""
    return voice_interface.get_voice_state()


# Test code
if __name__ == '__main__':
    start_voice_listener()
    
    try:
        while True:
            print("\nVoice Interface Status:")
            status = get_voice_status()
            print(f"Speaking: {status['is_speaking']}")
            print(f"Last Activity: {time.time() - status['last_activity']:.1f}s ago")
            print(f"Queue Size: {status['speech_queue_size']}")
            
            speech = get_latest_speech()
            if speech:
                print(f"\nRecognized: {speech}")
                say_awake_message()
            
            time.sleep(1)
    except KeyboardInterrupt:
        stop_voice_listener()
        print("\nVoice interface stopped")