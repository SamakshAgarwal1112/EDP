import time
import asyncio
import platform
import logging
import json
import random
import numpy as np
import pygame
import sounddevice as sd
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto
from gtts import gTTS
import io
import hashlib
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ALERT_INTERVAL_NORMAL = 15  # seconds
ALERT_INTERVAL_EXTREME = 5  # seconds
MAX_VOLUME = 1.0
MIN_VOLUME = 0.3
FADE_DURATION = 1.0  # seconds
SAMPLE_RATE = 44100
AMBIENT_NOISE_THRESHOLD = 0.05  # RMS for noise detection
SUPPORTED_LANGUAGES = ['en', 'hi', 'ta', 'te']  # English, Hindi, Tamil, Telugu

class AlertLevel(Enum):
    NONE = auto()
    MILD = auto()
    MODERATE = auto()
    SEVERE = auto()
    EXTREME = auto()

@dataclass
class AlertState:
    active: bool = False
    current_level: AlertLevel = AlertLevel.NONE
    last_alert_time: float = 0
    current_language: str = 'en'
    volume: float = MAX_VOLUME

class AlertSystem:
    def __init__(self, config_path: str = "config/settings.json"):
        self.state = AlertState()
        self.audio_available = False
        self.alert_messages = self.load_alert_messages()
        self.config = self.load_config(config_path)
        self.alert_sounds = {}
        self._initialize_system()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from settings.json"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('alert_system', {})
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {
                'alert_interval_normal': ALERT_INTERVAL_NORMAL,
                'alert_interval_extreme': ALERT_INTERVAL_EXTREME,
                'default_language': 'en',
                'max_volume': MAX_VOLUME,
                'min_volume': MIN_VOLUME
            }

    def _initialize_system(self):
        """Initialize audio and pre-load sounds"""
        try:
            pygame.init()
            pygame.mixer.init()
            self.audio_available = True
            logger.info("Initialized pygame mixer")
        except Exception as e:
            logger.warning(f"pygame init failed: {e}")
            self.audio_available = False

        # Pre-generate and verify alert sounds
        self.alert_sounds = {
            'mild': self.generate_tone(800, 0.5),
            'moderate': self.generate_tone(1000, 0.7),
            'severe': self.generate_tone(1200, 0.9),
            'extreme': self.generate_pulse_tone(1500, 1.0)
        }
        self._verify_sounds()

    def _verify_sounds(self):
        """Verify integrity of generated sounds"""
        for level, sound in self.alert_sounds.items():
            if sound is None:
                logger.error(f"Invalid sound for {level}")
                continue
            sound_bytes = (sound * 32767).astype(np.int16).tobytes()
            checksum = hashlib.sha256(sound_bytes).hexdigest()
            logger.debug(f"Sound {level} checksum: {checksum}")

    def load_alert_messages(self) -> Dict[AlertLevel, Dict[str, list]]:
        """Load multilingual alert messages"""
        return {
            AlertLevel.MILD: {
                'en': ["Please stay alert", "Focus on the road", "Keep your mind active"],
                'hi': ["Kripaya savdhan rahein", "Sadak par dhyan dein", "Thodi si chai pi lijiye"],
                'ta': ["Kaṇkaḷai veḻiyil vaittirungal", "Pātai mītu kavanam seluttungal"],
                'te': ["Dayachesi melukondi", "Road meeda focus cheyandi"]
            },
            AlertLevel.MODERATE: {
                'en': ["Warning: You seem drowsy", "Caution: Low attention detected"],
                'hi': ["Chetavani: Aapko nind aa rahi hai", "Savdhani: Dhyan kam hai"],
                'ta': ["Eccarikkai: Nīṅkaḷ tūkkamāka irukkirīrkaḷ", "Kaṇkaḷ tūṅkukiratu"],
                'te': ["Heccharika: Meeru nidra vastunnaru", "Jagratha: Concentration taggindi"]
            },
            AlertLevel.SEVERE: {
                'en': ["Danger: Severe drowsiness detected", "Pull over immediately"],
                'hi': ["Khatra: Bhayanak nind ka pata chala", "Turant gadi rok dein"],
                'ta': ["Āpaṟṟu: Kaṭumaiyāṉa tūkkam kaṇṭupiṭikkappaṭṭatu", "Uṭaṉē niṟuttuṅkaḷ"],
                'te': ["Pantham: Tీvramaina nidra kanipinchindi", "Veganga vehicle apandi"]
            },
            AlertLevel.EXTREME: {
                'en': ["EMERGENCY! PULL OVER NOW!", "CRITICAL! STOP NOW!"],
                'hi': ["AAPATKAL! ABHI GADI ROK DEIN!", "SANKAT! ABHI RUK JAO!"],
                'ta': ["AKKAṬĀYAM! IPPOḺUTU NIṞUTTUṄKAḶ!", "PĒRICCIL! IPPOḺUTU NIṞUTTU!"],
                'te': ["AVASARAM! IPPUDU VEHICLE APANDI!", "VINASAKARAM! IPPUDE APANDI!"]
            }
        }

    def generate_tone(self, frequency: float, duration: float) -> Optional[np.ndarray]:
        """Generate a modulated tone for attention"""
        try:
            t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
            tone = 0.5 * np.sin(2 * np.pi * frequency * t) * (1 + 0.2 * np.sin(2 * np.pi * 5 * t))
            return tone
        except Exception as e:
            logger.error(f"Failed to generate tone: {e}")
            return None

    def generate_pulse_tone(self, frequency: float, duration: float) -> Optional[np.ndarray]:
        """Generate a pulsing alert tone"""
        try:
            t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
            envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
            tone = envelope * np.sin(2 * np.pi * frequency * t)
            return tone
        except Exception as e:
            logger.error(f"Failed to generate pulse tone: {e}")
            return None

    async def detect_ambient_noise(self) -> float:
        """Detect ambient noise level"""
        try:
            recording = sd.rec(int(0.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()
            rms = np.sqrt(np.mean(recording**2))
            return rms
        except Exception as e:
            logger.warning(f"Ambient noise detection failed: {e}")
            return 0.0

    async def adjust_volume(self):
        """Adjust volume based on ambient noise"""
        noise_level = await self.detect_ambient_noise()
        if noise_level > AMBIENT_NOISE_THRESHOLD:
            self.state.volume = min(MAX_VOLUME, MAX_VOLUME * (1 + noise_level))
        else:
            self.state.volume = self.config.get('min_volume', MIN_VOLUME)
        logger.debug(f"Adjusted volume to {self.state.volume}")

    async def play_alert_sound(self, level: AlertLevel):
        """Play alert sound with dynamic volume"""
        sound_key = {
            AlertLevel.MILD: 'mild',
            AlertLevel.MODERATE: 'moderate',
            AlertLevel.SEVERE: 'severe',
            AlertLevel.EXTREME: 'extreme'
        }.get(level, 'mild')

        sound = self.alert_sounds.get(sound_key)
        if sound is None:
            logger.warning(f"No sound for {level}")
            return

        try:
            await self.adjust_volume()
            sound_stereo = np.column_stack((sound, sound))
            sound_bytes = (sound_stereo * 32767 * self.state.volume).astype(np.int16)
            sound_obj = pygame.sndarray.make_sound(sound_bytes)
            sound_obj.play(fade_ms=int(FADE_DURATION * 1000))
            logger.info(f"Playing {sound_key} alert sound")
        except Exception as e:
            logger.error(f"Failed to play sound: {e}")
            # Fallback to system beep
            if platform.system() == 'Windows':
                import winsound
                winsound.Beep(1000, 500)
            elif platform.system() == 'Darwin':
                os.system('say "ALERT"')
            else:
                os.system('paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga')

    async def speak_alert(self, text: str, lang: str):
        """Speak alert using gTTS for multilingual support"""
        try:
            tts = gTTS(text=text, lang=lang)
            with io.BytesIO() as f:
                tts.write_to_fp(f)
                f.seek(0)
                pygame.mixer.music.load(f)
                pygame.mixer.music.set_volume(self.state.volume)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
            logger.info(f"[TTS] {text} ({lang})")
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            # Fallback to pyttsx3
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                logger.warning("All TTS failed")

    async def mild_alert_loop(self):
        """Mild alert loop with escalation"""
        escalation_count = 0
        while self.state.active and self.state.current_level == AlertLevel.MILD:
            if time.time() - self.state.last_alert_time >= self.config.get('alert_interval_normal'):
                messages = self.alert_messages[AlertLevel.MILD][self.state.current_language]
                if random.random() < 0.1:  # 10% chance of misinterpretation
                    messages = ["Did you say you're sleepy?", "Are you napping?"]
                message = random.choice(messages)
                await self.speak_alert(message, self.state.current_language)
                await self.play_alert_sound(AlertLevel.MILD)
                self.state.last_alert_time = time.time()
                escalation_count += 1
                if escalation_count >= 3:  # Escalate after 3 mild alerts
                    self.state.current_level = AlertLevel.MODERATE
                    logger.info("Escalating to MODERATE alert")
            await asyncio.sleep(1)

    async def extreme_alert_loop(self):
        """Extreme alert loop"""
        while self.state.active and self.state.current_level == AlertLevel.EXTREME:
            if time.time() - self.state.last_alert_time >= self.config.get('alert_interval_extreme'):
                message = random.choice(self.alert_messages[AlertLevel.EXTREME][self.state.current_language])
                await self.speak_alert(message, self.state.current_language)
                await self.play_alert_sound(AlertLevel.EXTREME)
                self.state.last_alert_time = time.time()
            await asyncio.sleep(0.5)

    async def start_alert(self, level: AlertLevel):
        """Start alert at specified level"""
        if self.state.active and self.state.current_level == level:
            return
        await self.stop_alert()
        self.state.active = True
        self.state.current_level = level
        self.state.last_alert_time = time.time()
        logger.info(f"Started {level.name} alert")

    async def stop_alert(self):
        """Stop all alerts"""
        if not self.state.active:
            return
        self.state.active = False
        self.state.current_level = AlertLevel.NONE
        pygame.mixer.music.stop()
        logger.info("All alerts stopped")

    async def handle_alert(self, mode: str, is_speaking: bool, language: str = 'en'):
        """Handle alert with validation and language support"""
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language {language}. Defaulting to 'en'")
            language = 'en'
        self.state.current_language = language

        if is_speaking:
            logger.info("User is speaking - suppressing alerts")
            await self.stop_alert()
            return

        valid_modes = ['normal', 'moderate', 'severe', 'extreme']
        if mode not in valid_modes:
            logger.warning(f"Invalid mode {mode}. Defaulting to normal")
            mode = 'normal'

        alert_level = {
            'normal': AlertLevel.MILD,
            'moderate': AlertLevel.MODERATE,
            'severe': AlertLevel.SEVERE,
            'extreme': AlertLevel.EXTREME
        }[mode]

        await self.start_alert(alert_level)

    def get_alert_state(self) -> Dict[str, Any]:
        """Get current alert system state"""
        return {
            'active': self.state.active,
            'level': self.state.current_level.name,
            'last_alert': self.state.last_alert_time,
            'language': self.state.current_language,
            'volume': self.state.volume
        }

# Singleton instance
alert_system = AlertSystem()

async def handle_alert(mode: str = 'normal', is_speaking: bool = False, language: str = 'en'):
    """Public interface to handle alerts"""
    await alert_system.handle_alert(mode, is_speaking, language)

async def stop_alerts():
    """Public interface to stop all alerts"""
    await alert_system.stop_alert()

def get_alert_status() -> Dict[str, Any]:
    """Get current alert status"""
    return alert_system.get_alert_state()

async def main():
    """Main loop for alert system (Pyodide-compatible)"""
    alert_system.state.active = True
    try:
        while alert_system.state.active:
            if alert_system.state.current_level == AlertLevel.MILD:
                await alert_system.mild_alert_loop()
            elif alert_system.state.current_level == AlertLevel.EXTREME:
                await alert_system.extreme_alert_loop()
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        await stop_alerts()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())