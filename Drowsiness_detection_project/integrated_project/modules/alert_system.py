import time
import threading
import os
import platform
import pyttsx3
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import random
import subprocess
from enum import Enum, auto
import pygame
from pygame import mixer
import numpy as np
import sounddevice as sd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ALERT_INTERVAL_NORMAL = 15  # seconds
ALERT_INTERVAL_EXTREME = 5  # seconds
MAX_VOLUME = 1.0
MIN_VOLUME = 0.3
FADE_DURATION = 1.0  # seconds for audio fade in/out

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
    alert_thread: Optional[threading.Thread] = None
    audio_initialized: bool = False

class AlertSystem:
    def __init__(self):
        # Initialize components
        self.state = AlertState()
        self.lock = threading.Lock()
        self.audio_available = False
        self.alert_messages = self.load_alert_messages()
        
        # Initialize audio systems
        self.init_audio_systems()
        
        # Initialize TTS engine
        self.tts_engine = self.init_tts()
        
        # Pre-load alert sounds
        self.alert_sounds = {
            'mild': self.generate_tone(800, 0.5),
            'moderate': self.generate_tone(1000, 0.7),
            'severe': self.generate_tone(1200, 0.9),
            'extreme': self.generate_pulse_tone(1500, 1.0)
        }

    def init_audio_systems(self):
        """Initialize audio playback systems with fallbacks"""
        try:
            # Try pygame first (best for precise timing)
            pygame.init()
            mixer.init()
            self.audio_available = True
            logger.info("Initialized pygame mixer for audio")
        except Exception as e:
            logger.warning(f"Couldn't initialize pygame mixer: {e}")
            self.audio_available = False

    def init_tts(self):
        """Initialize text-to-speech engine with fallbacks"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', MAX_VOLUME)
            
            # Try to find a good voice
            voices = engine.getProperty('voices')
            preferred_voices = ['Microsoft David', 'Microsoft Zira', 'Alex']
            for voice in voices:
                if any(v in voice.name for v in preferred_voices):
                    engine.setProperty('voice', voice.id)
                    break
            
            return engine
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            return None

    def load_alert_messages(self) -> Dict[AlertLevel, list]:
        """Load alert messages for different levels"""
        return {
            AlertLevel.MILD: [
                "Please stay alert",
                "Stay focused on the road",
                "Let's keep your mind active",
                "How about some fresh air?",
                "Time to check your posture"
            ],
            AlertLevel.MODERATE: [
                "Warning: You seem drowsy",
                "Alert: Signs of fatigue detected",
                "Caution: Your attention seems low",
                "Warning: Decreased alertness detected",
                "Please pay attention"
            ],
            AlertLevel.SEVERE: [
                "Danger: Severe drowsiness detected",
                "Emergency: Pull over immediately",
                "Critical: You must wake up now",
                "Red alert: Stop driving now",
                "Life-threatening situation detected"
            ],
            AlertLevel.EXTREME: [
                "EMERGENCY! PULL OVER NOW!",
                "CRITICAL ALERT! STOP THE VEHICLE!",
                "LIFE THREAT! WAKE UP IMMEDIATELY!",
                "EXTREME DANGER! STOP DRIVING!",
                "COLLISION IMMINENT! STOP NOW!"
            ]
        }

    def generate_tone(self, frequency: float, duration: float) -> Optional[np.ndarray]:
        """Generate a sine wave tone"""
        try:
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * frequency * t)
            return tone
        except Exception as e:
            logger.error(f"Failed to generate tone: {e}")
            return None

    def generate_pulse_tone(self, frequency: float, duration: float) -> Optional[np.ndarray]:
        """Generate a pulsing alert tone"""
        try:
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Create a pulsing effect
            envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # 3Hz pulsation
            tone = envelope * np.sin(2 * np.pi * frequency * t)
            return tone
        except Exception as e:
            logger.error(f"Failed to generate pulse tone: {e}")
            return None

    def play_alert_sound(self, level: AlertLevel):
        """Play appropriate alert sound for level"""
        sound_key = {
            AlertLevel.MILD: 'mild',
            AlertLevel.MODERATE: 'moderate',
            AlertLevel.SEVERE: 'severe',
            AlertLevel.EXTREME: 'extreme'
        }.get(level, 'mild')
        
        sound = self.alert_sounds.get(sound_key)
        if sound is None:
            logger.warning(f"No sound available for level {level}")
            return
        
        try:
            if self.audio_available:
                # Convert numpy array to pygame sound
                sound_stereo = np.column_stack((sound, sound))
                sound_bytes = (sound_stereo * 32767).astype(np.int16)
                sound_obj = pygame.sndarray.make_sound(sound_bytes)
                
                # Play with fade in
                sound_obj.play(fade_ms=int(FADE_DURATION * 1000))
            else:
                # Fallback to system beep
                if platform.system() == 'Windows':
                    import winsound
                    frequency = {
                        'mild': 800,
                        'moderate': 1000,
                        'severe': 1200,
                        'extreme': 1500
                    }[sound_key]
                    winsound.Beep(frequency, int(duration * 1000))
                elif platform.system() == 'Darwin':  # macOS
                    os.system(f'say "ALERT"')
                else:  # Linux
                    os.system('paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga')
        except Exception as e:
            logger.error(f"Failed to play sound: {e}")

    def speak_alert(self, text: str):
        """Speak alert message with TTS"""
        if self.tts_engine is None:
            logger.warning("TTS engine not available")
            return
        
        try:
            logger.info(f"[ALERT-TTS] {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS failed: {e}")

    def mild_alert_loop(self):
        """Mild alert periodic loop"""
        while self.state.active and self.state.current_level == AlertLevel.MILD:
            with self.lock:
                if time.time() - self.state.last_alert_time >= ALERT_INTERVAL_NORMAL:
                    message = random.choice(self.alert_messages[AlertLevel.MILD])
                    self.speak_alert(message)
                    self.play_alert_sound(AlertLevel.MILD)
                    self.state.last_alert_time = time.time()
            time.sleep(1)

    def extreme_alert_loop(self):
        """Extreme alert periodic loop"""
        while self.state.active and self.state.current_level == AlertLevel.EXTREME:
            with self.lock:
                if time.time() - self.state.last_alert_time >= ALERT_INTERVAL_EXTREME:
                    message = random.choice(self.alert_messages[AlertLevel.EXTREME])
                    self.speak_alert(message)
                    self.play_alert_sound(AlertLevel.EXTREME)
                    self.state.last_alert_time = time.time()
            time.sleep(0.5)  # More frequent checking for extreme alerts

    def start_alert(self, level: AlertLevel):
        """Start alert at specified level"""
        with self.lock:
            if self.state.active and self.state.current_level == level:
                return  # Already active at this level
            
            self.stop_alert()  # Stop any existing alerts
            
            self.state.active = True
            self.state.current_level = level
            self.state.last_alert_time = time.time()
            
            # Start appropriate alert thread
            if level == AlertLevel.MILD:
                self.state.alert_thread = threading.Thread(target=self.mild_alert_loop)
            elif level == AlertLevel.EXTREME:
                self.state.alert_thread = threading.Thread(target=self.extreme_alert_loop)
            
            if self.state.alert_thread:
                self.state.alert_thread.daemon = True
                self.state.alert_thread.start()
            
            logger.info(f"Started {level.name} alert")

    def stop_alert(self):
        """Stop any active alerts"""
        with self.lock:
            if not self.state.active:
                return
            
            self.state.active = False
            if self.state.alert_thread and self.state.alert_thread.is_alive():
                self.state.alert_thread.join(timeout=1)
            
            self.state.current_level = AlertLevel.NONE
            logger.info("All alerts stopped")

    def handle_alert(self, mode: str = "normal", is_speaking: bool = False):
        """Handle alert based on drowsiness mode and speaking state"""
        if is_speaking:
            logger.info("User is speaking - suppressing alerts")
            self.stop_alert()
            return
        
        try:
            alert_level = {
                "normal": AlertLevel.MILD,
                "moderate": AlertLevel.MODERATE,
                "severe": AlertLevel.SEVERE,
                "extreme": AlertLevel.EXTREME
            }[mode]
            
            self.start_alert(alert_level)
        except KeyError:
            logger.warning(f"Unknown alert mode: {mode}")
            self.start_alert(AlertLevel.MILD)

    def get_alert_state(self) -> Dict[str, Any]:
        """Get current alert system state"""
        with self.lock:
            return {
                "active": self.state.active,
                "level": self.state.current_level.name,
                "last_alert": self.state.last_alert_time,
                "audio_available": self.audio_available,
                "tts_available": self.tts_engine is not None
            }


# Singleton instance
alert_system = AlertSystem()

def handle_alert(mode: str = "normal", is_speaking: bool = False):
    """Public interface to handle alerts"""
    alert_system.handle_alert(mode, is_speaking)

def stop_alerts():
    """Public interface to stop all alerts"""
    alert_system.stop_alert()

def get_alert_status() -> Dict[str, Any]:
    """Get current alert status"""
    return alert_system.get_alert_state()


# Test code
if __name__ == '__main__':
    print("Testing Alert System...")
    
    # Test mild alert
    print("\nTesting mild alert:")
    handle_alert(mode="normal", is_speaking=False)
    time.sleep(10)
    
    # Test extreme alert
    print("\nTesting extreme alert:")
    handle_alert(mode="extreme", is_speaking=False)
    time.sleep(10)
    
    # Test stop
    print("\nStopping alerts:")
    stop_alerts()
    time.sleep(3)
    
    print("\nAlert system test complete")
    print("Final status:", get_alert_status())