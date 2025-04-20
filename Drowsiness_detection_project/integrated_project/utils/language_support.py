import asyncio
import logging
import time
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, Tuple, List
from deep_translator import MyMemoryTranslator
import fasttext
import os
import json
from pathlib import Path
from cryptography.fernet import Fernet
import aiohttp
import re
from functools import lru_cache
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageSupport:
    def __init__(self, config_path: str = "config/settings.json"):
        self.config = self.load_config(config_path)
        self.language_model = self._load_language_model()
        self.supported_languages = self._get_supported_languages()
        self.translator = MyMemoryTranslator
        self.lock = asyncio.Lock()
        self.translation_cache = defaultdict(dict)
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self._running = True
        self.metrics = {
            'detection_accuracy': 0,
            'detection_time': 0,
            'translation_time': 0,
            'cache_hits': 0,
            'total_requests': 0
        }
        asyncio.create_task(self._cache_maintenance())

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from settings.json"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f).get('language_support', {})
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {
                'supported_languages': [
                    'en', 'hi', 'ta', 'te', 'bn', 'mr', 'kn'  # English, Hindi, Tamil, Telugu, Bengali, Marathi, Kannada
                ],
                'cache_expiry': 86400,  # 24 hours
                'fasttext_model': 'integrated_project/models/lid.176.bin',
                'confidence_threshold': 0.7
            }

    def _load_language_model(self):
        """Load FastText language model with integrity check"""
        model_path = self.config.get('fasttext_model')
        try:
            if not os.path.exists(model_path):
                logger.warning(f"FastText model not found at {model_path}")
                return None
            with open(model_path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            logger.debug(f"Model {model_path} checksum: {checksum}")
            return fasttext.load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load language model: {e}")
            return None

    def _get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages with names"""
        return {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'mr': 'Marathi',
            'kn': 'Kannada',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'zh-cn': 'Chinese (Simplified)',
            'ar': 'Arabic'
        }

    async def _cache_maintenance(self):
        """Periodically clean translation cache"""
        while self._running:
            await asyncio.sleep(1800)  # Run every 30 minutes
            async with self.lock:
                current_time = time.time()
                for lang in list(self.translation_cache.keys()):
                    self.translation_cache[lang] = {
                        k: v for k, v in self.translation_cache[lang].items()
                        if current_time - v['timestamp'] < self.config.get('cache_expiry')
                    }
                logger.info("Performed cache maintenance")

    @lru_cache(maxsize=1000)
    async def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence thresholding"""
        if not self._validate_text(text):
            return 'en', 0.0
        
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        try:
            if self.language_model:
                predictions = self.language_model.predict(text.replace("\n", " "), k=1)
                lang_code = predictions[0][0].replace('__label__', '')
                confidence = float(predictions[1][0])
                lang_code = self._map_language_code(lang_code)
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"https://api.mymemory.translated.net/get?q={text}&langpair=auto|en") as resp:
                        data = await resp.json()
                        lang_code = data.get('detectedLang', 'en')
                        confidence = 0.9
            
            if confidence < self.config.get('confidence_threshold'):
                logger.warning(f"Low confidence ({confidence:.2f}) for {lang_code}; defaulting to English")
                lang_code = 'en'
                confidence = 0.0
            
            detection_time = time.time() - start_time
            self.metrics['detection_time'] = 0.9 * self.metrics['detection_time'] + 0.1 * detection_time
            
            logger.info(f"Detected language: {lang_code} (confidence: {confidence:.2f})")
            return lang_code, confidence
        
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return 'en', 0.0

    def _map_language_code(self, code: str) -> str:
        """Map language codes to standard format"""
        code = code.lower()
        if code == 'zh':
            return 'zh-cn'
        if code in ['pt-br', 'pt-pt']:
            return 'pt'
        return code

    def _validate_text(self, text: str) -> bool:
        """Validate text input"""
        if not isinstance(text, str) or not text.strip():
            logger.warning("Invalid text input")
            return False
        if len(text) > 1000:
            logger.warning("Text too long")
            return False
        if re.search(r'[<>{}]', text):
            logger.warning("Potential injection detected")
            return False
        return True

    async def translate_text(self, text: str, src_lang: Optional[str] = None, 
                          dest_lang: str = 'en', use_cache: bool = True) -> str:
        """Translate text with caching and colloquial handling"""
        if not self._validate_text(text):
            return text
        
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        cache_key = (text, src_lang or 'auto', dest_lang)
        if use_cache:
            async with self.lock:
                if dest_lang in self.translation_cache and text in self.translation_cache[dest_lang]:
                    self.metrics['cache_hits'] += 1
                    return self.cipher.decrypt(self.translation_cache[dest_lang][text]['translation']).decode()
        
        try:
            if not src_lang:
                src_lang, _ = await self.detect_language(text)
            
            if src_lang == dest_lang:
                return text
            
            # Colloquial mappings for Indian context
            colloquial_map = {
                'en': {
                    'Stay awake!': {
                        'hi': 'Jaagte raho!',
                        'ta': 'Veḻiyāka iru!',
                        'te': 'Melukondi undu!',
                        'bn': 'Jāgā thāka!',
                        'mr': 'Jāgā raha!',
                        'kn': 'Echchara iru!'
                    }
                }
            }
            
            if text in colloquial_map.get(src_lang, {}):
                result = colloquial_map[src_lang][text].get(dest_lang, text)
            else:
                translator = self.translator(source=src_lang, target=dest_lang)
                result = translator.translate(text)
            
            async with self.lock:
                self.translation_cache[dest_lang][text] = {
                    'translation': self.cipher.encrypt(result.encode()),
                    'timestamp': time.time()
                }
            
            translation_time = time.time() - start_time
            self.metrics['translation_time'] = 0.9 * self.metrics['translation_time'] + 0.1 * translation_time
            
            return result
        
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text

    async def translate_to_english(self, text: str) -> str:
        """Translate to English"""
        return await self.translate_text(text, dest_lang='en')

    async def translate_from_english(self, text: str, dest_lang: str) -> str:
        """Translate from English"""
        return await self.translate_text(text, src_lang='en', dest_lang=dest_lang)

    async def batch_translate(self, texts: List[str], src_lang: Optional[str] = None,
                           dest_lang: str = 'en') -> List[str]:
        """Translate multiple texts concurrently"""
        tasks = [self.translate_text(text, src_lang, dest_lang) for text in texts]
        return await asyncio.gather(*tasks)

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages"""
        return self.supported_languages

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        return {
            'detection_accuracy': self.metrics['detection_accuracy'],
            'avg_detection_time': self.metrics['detection_time'],
            'avg_translation_time': self.metrics['translation_time'],
            'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['total_requests']),
            'total_requests': self.metrics['total_requests']
        }

    async def shutdown(self):
        """Clean shutdown"""
        self._running = False
        async with self.lock:
            self.translation_cache.clear()
        logger.info("Language support shut down")

# Singleton instance
language_support = LanguageSupport()

async def detect_language(text: str) -> Tuple[str, float]:
    """Detect language"""
    return await language_support.detect_language(text)

async def translate_text(text: str, src_lang: Optional[str] = None, 
                      dest_lang: str = 'en') -> str:
    """Translate text"""
    return await language_support.translate_text(text, src_lang, dest_lang)

async def translate_to_english(text: str) -> str:
    """Translate to English"""
    return await language_support.translate_to_english(text)

async def translate_from_english(text: str, dest_lang: str) -> str:
    """Translate from English"""
    return await language_support.translate_from_english(text, dest_lang)

def get_supported_languages() -> Dict[str, str]:
    """Get supported languages"""
    return language_support.get_supported_languages()

async def shutdown_language_support():
    """Clean shutdown"""
    await language_support.shutdown()

async def main():
    test_phrases = [
        "Bonjour, comment ça va?",
        "मैं जाग रहा हूँ",
        "நான் விழித்திருக்கிறேன்",
        "నేను మెలకువగా ఉన్నాను",
        "Stay awake!",
        "أنا مستيقظ",
        "Estou acordado"
    ]
    
    st.header("Language Support Test")
    for phrase in test_phrases:
        st.write(f"**Original**: {phrase}")
        lang, confidence = await detect_language(phrase)
        lang_name = get_supported_languages().get(lang, lang)
        st.write(f"**Detected**: {lang_name} (confidence: {confidence:.2f})")
        en_text = await translate_to_english(phrase)
        st.write(f"**English**: {en_text}")
        if lang in get_supported_languages():
            back_text = await translate_from_english(en_text, lang)
            st.write(f"**Back Translation**: {back_text}")
        await asyncio.sleep(1)
    
    st.subheader("Performance Metrics")
    for k, v in language_support.get_performance_metrics().items():
        st.write(f"{k.replace('_', ' ').title()}: {v:.4f}")

if __name__ == "__main__":
    asyncio.run(main())