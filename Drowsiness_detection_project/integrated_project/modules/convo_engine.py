import asyncio
import random
import time
import json
import logging
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import whisper
from gtts import gTTS
import io
import pygame
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SPEECH_RATE = 170
MAX_RESPONSE_HISTORY = 10
AUDIO_SAMPLE_RATE = 44100
AUDIO_RECORD_SECONDS = 5
MISINTERPRETATION_PROBABILITY = 0.15
LANGUAGE_DETECTION_THRESHOLD = 0.7
CONVERSATION_TIMEOUT = 30
SUPPORTED_LANGUAGES = ['en', 'hi', 'ta', 'te']  # English, Hindi, Tamil, Telugu

@dataclass
class ConversationState:
    is_active: bool = False
    last_interaction: float = field(default_factory=time.time)
    user_language: str = 'en'
    response_history: List[str] = field(default_factory=list)
    drowsiness_level: str = 'normal'

class ConversationEngine:
    def __init__(self, config_path: str = "config/settings.json"):
        self.config = self.load_config(config_path)
        self.conversation_state = ConversationState()
        self.running = False
        self.audio_buffer = []
        
        # Initialize NLP models
        self.language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        self.dialogue_model = pipeline("conversational", model="microsoft/DialoGPT-medium")
        self.mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.whisper_model = whisper.load_model("base")
        
        # Initialize audio
        pygame.init()
        pygame.mixer.init()
        
        # Load knowledge base
        self.knowledge_base = self.load_knowledge_base()
        self.misinterpret_phrases = self.load_misinterpretations()
        self.alert_responses = self.load_alert_responses()
        
        # Start background tasks
        self.start_workers()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from settings.json"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f).get('conversation_engine', {})
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {
                'speech_rate': DEFAULT_SPEECH_RATE,
                'misinterpretation_probability': MISINTERPRETATION_PROBABILITY,
                'conversation_timeout': CONVERSATION_TIMEOUT,
                'default_language': 'en'
            }

    def load_knowledge_base(self) -> Dict[str, List[Dict]]:
        """Load multilingual knowledge base"""
        return {
            "questions": [
                {"question": {"en": "What is your name?", "hi": "Aapka naam kya hai?", "ta": "Uṅkaḷ peyar eṉṉa?", "te": "Mee peru emiti?"}, 
                 "response": {"en": "I'm AwakeBot, here to keep you alert!", "hi": "Main AwakeBot hoon, aapko jagaye rakhne ke liye!", "ta": "Nāṉ AwakeBot, uṅkaḷai veḻiyil vaittirukkirēṉ!", "te": "Nenu AwakeBot, mee melukundaga undadaniki!"}},
                {"question": {"en": "How are you feeling?", "hi": "Aap kaisa mehsoos kar rahe hain?", "ta": "Nīṅkaḷ eppaṭi uṇarukiṟīrkaḷ?", "te": "Meeru ela feel avuthunnaru?"}, 
                 "response": {"en": "I'm always energetic! How about you?", "hi": "Main hamesha urjavan hoon! Aap kaise hain?", "ta": "Nāṉ eppōtum āṟṟalmaṉatu! Nīṅkaḷ eppaṭi?", "te": "Nenu eppudu energetic ga untanu! Meeru ela unnaru?"}},
            ],
            "small_talk": [
                {"pattern": {"en": "hello|hi|hey", "hi": "namaste|hello|hi", "ta": "vaṇakkam|hello", "te": "hello|namaste"}, 
                 "response": {"en": "Hello! How are you feeling?", "hi": "Namaste! Aap kaisa mehsoos kar rahe hain?", "ta": "Vaṇakkam! Nīṅkaḷ eppaṭi uṇarukiṟīrkaḷ?", "te": "Hello! Meeru ela feel avuthunnaru?"}},
                {"pattern": {"en": "tired|sleepy|drowsy", "hi": "thak|neend|nind", "ta": "kalaippāka|tūkkam", "te": "alupu|niddura"}, 
                 "response": {"en": "Let's talk to stay awake!", "hi": "Chalo, baat karte hain taaki jagte rahen!", "ta": "Pēci tūṅkāmal iruppōm!", "te": "Matladukundam, melukundaga undadaniki!"}},
            ]
        }

    def load_misinterpretations(self) -> Dict[str, List[str]]:
        """Load multilingual misinterpretation phrases"""
        return {
            'en': ["Did you say you’re sleepy?", "Are you talking about food?", "I heard nap time!"],
            'hi': ["Kya aapne kaha aapko neend aa rahi hai?", "Kya aap khane ki baat kar rahe hain?", "Mujhe laga aapne nap kaha!"],
            'ta': ["Nīṅkaḷ tūṅkirīrkaḷ eṉṟu kūriviṭṭīrkaḷā?", "Nīṅkaḷ uṇavu paṟṟi pēcirukkiṟīrkaḷā?", "Nāṉ tūkkam eṉṟu keṭṭēṉ!"],
            'te': ["Meeru niddura antunnara?", "Meeru food gurinchi matladutunnara?", "Nenu nap time ani vinnanu!"]
        }

    def load_alert_responses(self) -> Dict[str, Dict[str, List[str]]]:
        """Load multilingual alert responses"""
        return {
            "normal": {
                'en': ["You seem tired. Let's chat!", "Stay with me! Tell me a story."],
                'hi': ["Aap thake hue lag rahe hain. Chalo baat karte hain!", "Mere saath raho! Ek kahani sunao."],
                'ta': ["Nīṅkaḷ kalaippāka irukkirīrkaḷ. Pēciṭalām!", "Eṉṉuṭaṉ iruṅkaḷ! Oru katai kūrungal."],
                'te': ["Meeru alupuga kanipistunnaru. Matladukundam!", "Naatho undandi! Oka katha cheppandi."]
            },
            "extreme": {
                'en': ["ALERT! You seem very drowsy! Respond now!", "EMERGENCY! Wake up immediately!"],
                'hi': ["SAVDHAN! Aapko bahut neend aa rahi hai! Abhi jawab do!", "AAPATKAL! Turant jaago!"],
                'ta': ["ECCARIKKAI! Nīṅkaḷ mikka tūkkamāka irukkirīrkaḷ! Ippoḻutē pātiḻiyungal!", "AKKAṬĀYAM! Uṭaṉē veḻiyungal!"],
                'te': ["HECHARRIKA! Meeru chala nidduraga unnaru! Ippude samadhanam cheppandi!", "AVASARAM! Ventane melukondi!"]
            },
            "awake": {
                'en': ["Great job staying awake!", "You're fully alert now!"],
                'hi': ["Bahut badhiya, aap jaag rahe hain!", "Aap ab poori tarah se sakriy hain!"],
                'ta': ["Veḻiyāka iruppatu nallatu!", "Nīṅkaḷ ippoḻutu muḻumaiyāka veḻiyāka irukkiṟīrkaḷ!"],
                'te': ["Melukundaga undadam chala bagundi!", "Meeru ippudu purna drushtitho unnaru!"]
            }
        }

    async def start_workers(self):
        """Start background tasks"""
        self.running = True
        asyncio.create_task(self.conversation_timeout_checker())
        asyncio.create_task(self.response_worker())
        logger.info("Conversation engine workers started")

    async def stop_workers(self):
        """Stop background tasks"""
        self.running = False
        pygame.mixer.music.stop()
        logger.info("Conversation engine workers stopped")

    async def response_worker(self):
        """Process responses asynchronously"""
        while self.running:
            if self.conversation_state.is_active:
                response = await self.generate_response_async()
                if response:
                    await self.speak(response)
                    with self.conversation_state.response_history as history:
                        history.append(response)
                        if len(history) > MAX_RESPONSE_HISTORY:
                            history.pop(0)
            await asyncio.sleep(0.5)

    async def conversation_timeout_checker(self):
        """Check for conversation timeout"""
        while self.running:
            time_since_last = time.time() - self.conversation_state.last_interaction
            if self.conversation_state.is_active and time_since_last > self.config.get('conversation_timeout'):
                self.conversation_state.is_active = False
                await self.speak("Our chat timed out. I'm here when you need me!")
                logger.info("Conversation timeout reset")
            await asyncio.sleep(5)

    async def detect_language(self, text: str) -> str:
        """Detect language using XLM-RoBERTa"""
        try:
            result = self.language_detector(text)[0]
            if result['score'] > LANGUAGE_DETECTION_THRESHOLD and result['label'] in SUPPORTED_LANGUAGES:
                return result['label']
            return self.config.get('default_language', 'en')
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'

    async def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text using mBART"""
        try:
            if target_lang not in SUPPORTED_LANGUAGES:
                target_lang = 'en'
            src_lang = await self.detect_language(text)
            if src_lang == target_lang:
                return text
            
            self.mbart_tokenizer.src_lang = f"{src_lang}_IN"
            inputs = self.mbart_tokenizer(text, return_tensors="pt", padding=True)
            translated = self.mbart_model.generate(
                **inputs, forced_bos_token_id=self.mbart_tokenizer.lang_code_to_id[f"{target_lang}_IN"]
            )
            return self.mbart_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text

    async def record_audio(self) -> Optional[str]:
        """Record audio and transcribe with Whisper"""
        try:
            logger.info(f"Recording audio for {AUDIO_RECORD_SECONDS} seconds...")
            recording = sd.rec(int(AUDIO_RECORD_SECONDS * AUDIO_SAMPLE_RATE), samplerate=AUDIO_SAMPLE_RATE, channels=1)
            sd.wait()
            
            temp_file = "temp_audio.wav"
            write(temp_file, AUDIO_SAMPLE_RATE, recording)
            
            result = self.whisper_model.transcribe(temp_file)
            os.remove(temp_file)  # Delete immediately
            return result["text"]
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return None

    async def should_misinterpret(self) -> bool:
        """Determine if misinterpretation should occur"""
        return random.random() < self.config.get('misinterpretation_probability')

    async def generate_response_async(self, user_input: Optional[str] = None) -> Optional[str]:
        """Generate response asynchronously"""
        if user_input:
            self.conversation_state.last_interaction = time.time()
            self.conversation_state.is_active = True
            user_input = await self.translate_text(user_input, 'en')
            
            if await self.should_misinterpret():
                return random.choice(self.misinterpret_phrases[self.conversation_state.user_language])
            
            for item in self.knowledge_base.get("questions", []):
                if user_input.lower() in [q.lower() for q in item["question"].values()]:
                    return item["response"][self.conversation_state.user_language]
            
            for item in self.knowledge_base.get("small_talk", []):
                patterns = item["pattern"][self.conversation_state.user_language].split("|")
                if any(p in user_input.lower() for p in patterns):
                    return item["response"][self.conversation_state.user_language]
            
            dialogue_response = self.dialogue_model(user_input).generated_responses[0]
            return await self.translate_text(dialogue_response, self.conversation_state.user_language)
        
        return None

    async def generate_alert_response(self, alert_level: str) -> str:
        """Generate alert response"""
        valid_levels = ["normal", "extreme", "awake"]
        if alert_level not in valid_levels:
            alert_level = "normal"
        return random.choice(self.alert_responses[alert_level][self.conversation_state.user_language])

    async def speak(self, text: str):
        """Speak text using gTTS"""
        try:
            tts = gTTS(text=text, lang=self.conversation_state.user_language)
            with io.BytesIO() as f:
                tts.write_to_fp(f)
                f.seek(0)
                pygame.mixer.music.load(f)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
            logger.info(f"[BOT SPEAK] {text}")
        except Exception as e:
            logger.error(f"TTS failed: {e}")

    async def process_user_input(self, text: str):
        """Process user text input"""
        self.conversation_state.user_language = await self.detect_language(text)
        response = await self.generate_response_async(text)
        if response:
            await self.speak(response)

    async def process_alert_condition(self, alert_level: str):
        """Process drowsiness alert condition"""
        self.conversation_state.drowsiness_level = alert_level
        response = await self.generate_alert_response(alert_level)
        await self.speak(response)
        if alert_level in ["normal", "extreme"]:
            self.conversation_state.is_active = True
            await self.speak("Let's keep talking to stay alert!")

    async def handle_input(self, text: Optional[str] = None):
        """Handle text or audio input"""
        if text is None:
            text = await self.record_audio()
        if text:
            await self.process_user_input(text)

    async def handle_alert(self, alert_level: str):
        """Handle alert condition"""
        await self.process_alert_condition(alert_level)

    def get_conversation_state(self) -> Dict:
        """Get current conversation state"""
        return {
            "is_active": self.conversation_state.is_active,
            "last_interaction": self.conversation_state.last_interaction,
            "language": self.conversation_state.user_language,
            "recent_responses": self.conversation_state.response_history[-5:],
            "drowsiness_level": self.conversation_state.drowsiness_level
        }

# Singleton instance
conversation_engine = ConversationEngine()

async def start_conversation_handler():
    """Initialize the conversation engine"""
    await conversation_engine.start_workers()

async def stop_conversation_handler():
    """Stop the conversation engine"""
    await conversation_engine.stop_workers()

async def handle_user_input(text: Optional[str] = None):
    """Public interface to handle user input"""
    await conversation_engine.handle_input(text)

async def handle_alert_condition(alert_level: str):
    """Public interface to handle alert conditions"""
    await conversation_engine.handle_alert(alert_level)

def get_conversation_status() -> Dict:
    """Get current conversation status"""
    return conversation_engine.get_conversation_state()

async def main():
    """Main loop for testing"""
    await start_conversation_handler()
    try:
        test_inputs = [
            "What is your name?",
            "I'm feeling really tired",
            "Namaste, kaise ho?",
            "நான் தூங்குகிறேன்",
            "నీవు ఎవరు?"
        ]
        for input_text in test_inputs:
            print(f"\n[USER] {input_text}")
            await handle_user_input(input_text)
            await asyncio.sleep(5)
        
        for level in ["normal", "extreme", "awake"]:
            print(f"\nAlert level: {level}")
            await handle_alert_condition(level)
            await asyncio.sleep(3)
    finally:
        await stop_conversation_handler()

if __name__ == "__main__":
    asyncio.run(main())