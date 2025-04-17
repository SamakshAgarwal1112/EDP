import random
import threading
import time
from queue import Queue, PriorityQueue
from googletrans import Translator
import pyttsx3
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json
import os
from difflib import get_close_matches
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SPEECH_RATE = 170
MAX_RESPONSE_HISTORY = 10
AUDIO_SAMPLE_RATE = 44100
AUDIO_RECORD_SECONDS = 5
MISINTERPRETATION_PROBABILITY = 0.15  # 15% chance
LANGUAGE_DETECTION_THRESHOLD = 0.7
CONVERSATION_TIMEOUT = 30  # seconds of inactivity before reset

@dataclass
class ConversationState:
    is_active: bool = False
    last_interaction: float = field(default_factory=time.time)
    user_language: str = 'en'
    response_history: List[str] = field(default_factory=list)
    user_profile: Dict = field(default_factory=dict)

class ConversationEngine:
    def __init__(self):
        # Initialize components
        self.translator = Translator()
        self.tts_engine = pyttsx3.init()
        self.configure_tts()
        
        # Conversation management
        self.response_queue = PriorityQueue()
        self.conversation_state = ConversationState()
        self.audio_buffer = []
        self.running = False
        
        # Load knowledge base
        self.knowledge_base = self.load_knowledge_base()
        self.misinterpret_phrases = self.load_misinterpretations()
        self.alert_responses = self.load_alert_responses()
        
        # Thread locks
        self.lock = threading.Lock()
        
        # Start background workers
        self.start_workers()

    def configure_tts(self):
        """Configure text-to-speech engine settings"""
        self.tts_engine.setProperty('rate', DEFAULT_SPEECH_RATE)
        self.tts_engine.setProperty('volume', 1.0)
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('voice', voices[0].id)  # Default voice

    def load_knowledge_base(self) -> Dict[str, List[Dict]]:
        """Load conversation knowledge base from JSON file"""
        try:
            with open('integrated_project/config/conversation_kb.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load knowledge base: {e}")
            return {
                "questions": [
                    {"question": "What is your name?", "response": "My name is AwakeBot! I'm here to keep you alert."},
                    {"question": "How are you feeling?", "response": "I'm always energetic! How about you?"},
                    {"question": "What is the capital of France?", "response": "The capital of France is Paris."},
                    {"question": "Why did you stop talking?", "response": "I'm always listening, just like a good assistant should."},
                    {"question": "Are you sleeping?", "response": "I'm never the one who sleeps here!"},
                ],
                "small_talk": [
                    {"pattern": "hello|hi|hey", "response": "Hello there! How are you feeling?"},
                    {"pattern": "thank you|thanks", "response": "You're welcome! Stay alert out there."},
                    {"pattern": "tired|sleepy|drowsy", "response": "I can help with that! Let's talk to keep you awake."},
                ]
            }

    def load_misinterpretations(self) -> List[str]:
        """Load misinterpretation phrases"""
        return [
            "Did you say banana?",
            "Are you talking to me or the wall?",
            "I thought you said nap time!",
            "Hmm? Sounded like you mumbled something about pizza.",
            "Say that again, I was dreaming... just kidding!",
            "Wait, did you say you wanted to sleep?",
            "I heard something about coffee?",
            "Sounded like you're getting tired there!",
            "Was that a yawn or are you saying something?",
            "Let me guess... you're getting sleepy?"
        ]

    def load_alert_responses(self) -> Dict[str, List[str]]:
        """Load responses for different alert levels"""
        return {
            "normal": [
                "You seem a bit tired. Let's talk to keep you awake!",
                "I noticed you might be getting drowsy. How about we chat?",
                "Stay with me! Let's have a conversation.",
                "I'm here to keep you alert. Tell me about your day.",
                "Let's talk to help you stay focused."
            ],
            "extreme": [
                "ALERT! You seem extremely drowsy! Please respond!",
                "DANGER! You need to wake up immediately!",
                "EMERGENCY ALERT! You appear to be falling asleep!",
                "CRITICAL WARNING! You must stay awake!",
                "RED ALERT! You're showing severe drowsiness signs!"
            ],
            "awake": [
                "Great job staying awake! Keep it up!",
                "You're doing well staying alert!",
                "Excellent! You seem fully awake now.",
                "Good work maintaining your focus!",
                "You're back to being fully alert! Well done."
            ]
        }

    def start_workers(self):
        """Start background worker threads"""
        self.running = True
        # Response processing thread
        threading.Thread(target=self.response_worker, daemon=True).start()
        # Conversation timeout thread
        threading.Thread(target=self.conversation_timeout_checker, daemon=True).start()
        logger.info("Conversation engine workers started")

    def stop_workers(self):
        """Stop background workers"""
        self.running = False
        logger.info("Conversation engine workers stopped")

    def response_worker(self):
        """Process responses from the queue"""
        while self.running:
            try:
                priority, response = self.response_queue.get(timeout=0.5)
                self.speak(response)
                with self.lock:
                    self.conversation_state.response_history.append(response)
                    if len(self.conversation_state.response_history) > MAX_RESPONSE_HISTORY:
                        self.conversation_state.response_history.pop(0)
                self.response_queue.task_done()
            except Exception as e:
                logger.debug(f"Response worker idle: {e}")

    def conversation_timeout_checker(self):
        """Check for conversation timeout"""
        while self.running:
            time_since_last = time.time() - self.conversation_state.last_interaction
            if self.conversation_state.is_active and time_since_last > CONVERSATION_TIMEOUT:
                with self.lock:
                    self.conversation_state.is_active = False
                    self.queue_response("Our conversation timed out. I'm here when you need me.", priority=3)
                logger.info("Conversation timeout reset")
            time.sleep(5)

    def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        try:
            detected = self.translator.detect(text)
            if detected.confidence > LANGUAGE_DETECTION_THRESHOLD:
                return detected.lang
            return 'en'
        except:
            return 'en'

    def translate_text(self, text: str, target_lang: str = 'en') -> str:
        """Translate text to target language"""
        try:
            if self.detect_language(text) != target_lang:
                translated = self.translator.translate(text, dest=target_lang)
                return translated.text
            return text
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text

    def record_audio(self, duration: int = AUDIO_RECORD_SECONDS) -> Optional[str]:
        """Record audio from microphone"""
        try:
            logger.info(f"Recording audio for {duration} seconds...")
            recording = sd.rec(int(duration * AUDIO_SAMPLE_RATE), samplerate=AUDIO_SAMPLE_RATE, channels=1)
            sd.wait()  # Wait until recording is finished
            
            # Save to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"temp_audio_{timestamp}.wav"
            write(filename, AUDIO_SAMPLE_RATE, recording)
            
            return filename
        except Exception as e:
            logger.error(f"Audio recording failed: {e}")
            return None

    def should_misinterpret(self) -> bool:
        """Determine if we should intentionally misinterpret"""
        return random.random() < MISINTERPRETATION_PROBABILITY

    def find_best_match(self, user_input: str, patterns: List[str]) -> Optional[str]:
        """Find the best match for user input in patterns"""
        matches = get_close_matches(user_input.lower(), patterns, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def get_knowledge_response(self, user_input: str) -> Optional[str]:
        """Get response from knowledge base"""
        # Check exact questions first
        for item in self.knowledge_base.get("questions", []):
            if user_input.lower() == item["question"].lower():
                return item["response"]
        
        # Check small talk patterns
        for item in self.knowledge_base.get("small_talk", []):
            patterns = item["pattern"].split("|")
            if self.find_best_match(user_input, patterns):
                return item["response"]
        
        return None

    def generate_response(self, user_input: str) -> str:
        """Generate appropriate response to user input"""
        # Update last interaction time
        with self.lock:
            self.conversation_state.last_interaction = time.time()
            if not self.conversation_state.is_active:
                self.conversation_state.is_active = True
        
        # Translate to English for processing
        english_input = self.translate_text(user_input)
        
        # Occasionally misinterpret for realism
        if self.should_misinterpret():
            return random.choice(self.misinterpret_phrases)
        
        # Check knowledge base
        kb_response = self.get_knowledge_response(english_input)
        if kb_response:
            return kb_response
        
        # Default responses
        if any(word in english_input.lower() for word in ["sleep", "tired", "drowsy"]):
            return "I'm here to help you stay awake! Let's talk more."
        
        return random.choice([
            "Interesting! Tell me more about that.",
            "I see. What else is on your mind?",
            "Let's keep talking to stay alert!",
            "That's fascinating. Could you elaborate?",
            "I'm listening. Please continue."
        ])

    def generate_alert_response(self, alert_level: str) -> str:
        """Generate response for alert condition"""
        responses = self.alert_responses.get(alert_level, [])
        if not responses:
            return "Please stay awake and focused!"
        return random.choice(responses)

    def speak(self, text: str):
        """Convert text to speech with logging"""
        logger.info(f"[BOT SPEAK] {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def queue_response(self, response: str, priority: int = 1):
        """Add response to processing queue with priority"""
        self.response_queue.put((priority, response))

    def process_user_input(self, text: str):
        """Process user text input"""
        response = self.generate_response(text)
        self.queue_response(response)

    def process_alert_condition(self, alert_level: str):
        """Process drowsiness alert condition"""
        if alert_level not in ["normal", "extreme", "awake"]:
            alert_level = "normal"
        
        response = self.generate_alert_response(alert_level)
        priority = 2 if alert_level == "normal" else 1  # Higher priority for extreme alerts
        self.queue_response(response, priority=priority)

    def handle_input(self, text: str):
        """Public method to handle text input"""
        threading.Thread(target=self.process_user_input, args=(text,), daemon=True).start()

    def handle_alert(self, alert_level: str):
        """Public method to handle alert condition"""
        threading.Thread(target=self.process_alert_condition, args=(alert_level,), daemon=True).start()

    def get_conversation_state(self) -> Dict:
        """Get current conversation state"""
        with self.lock:
            return {
                "is_active": self.conversation_state.is_active,
                "last_interaction": self.conversation_state.last_interaction,
                "language": self.conversation_state.user_language,
                "recent_responses": self.conversation_state.response_history[-5:]
            }


# Singleton instance
conversation_engine = ConversationEngine()

def start_conversation_handler():
    """Initialize the conversation engine"""
    conversation_engine.start_workers()

def stop_conversation_handler():
    """Stop the conversation engine"""
    conversation_engine.stop_workers()

def handle_user_input(text: str):
    """Public interface to handle user input"""
    conversation_engine.handle_input(text)

def handle_alert_condition(alert_level: str):
    """Public interface to handle alert conditions"""
    conversation_engine.handle_alert(alert_level)

def get_conversation_status() -> Dict:
    """Get current conversation status"""
    return conversation_engine.get_conversation_state()


# Test code
if __name__ == '__main__':
    start_conversation_handler()
    
    test_inputs = [
        "What is your name?",
        "Are you sleeping?",
        "I'm feeling really tired",
        "Bonjour, comment ça va?",
        "你是谁？",
        "I think I need to pull over",
        "Thank you for helping me"
    ]
    
    for input_text in test_inputs:
        print(f"\n[USER] {input_text}")
        handle_user_input(input_text)
        time.sleep(3)
    
    # Test alert responses
    print("\nTesting alert responses:")
    for level in ["normal", "extreme", "awake"]:
        print(f"\nAlert level: {level}")
        handle_alert_condition(level)
        time.sleep(2)
    
    time.sleep(5)
    stop_conversation_handler()