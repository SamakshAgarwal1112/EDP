from googletrans import Translator
import langdetect
from typing import Optional, Dict, Tuple
import logging
from functools import lru_cache
import fasttext
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageSupport:
    def __init__(self):
        # Initialize components
        self.translator = Translator()
        self.language_model = self._load_language_model()
        self.supported_languages = self._get_supported_languages()
        
        # Thread safety
        self.lock = threading.Lock()
        self.translation_cache = defaultdict(dict)
        
        # Performance metrics
        self.metrics = {
            'detection_accuracy': 0,
            'detection_time': 0,
            'translation_time': 0,
            'cache_hits': 0,
            'total_requests': 0
        }
        
        # Start cache maintenance thread
        self._running = True
        threading.Thread(target=self._cache_maintenance, daemon=True).start()

    def _load_language_model(self):
        """Load FastText language identification model"""
        try:
            # Try to load local model first
            if os.path.exists('lid.176.bin'):
                return fasttext.load_model('lid.176.bin')
            
            logger.warning("Local language model not found, using langdetect")
            return None
        except Exception as e:
            logger.error(f"Failed to load language model: {e}")
            return None

    def _get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages with names"""
        return {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh-cn': 'Chinese (Simplified)',
            'ja': 'Japanese',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }

    def _cache_maintenance(self):
        """Periodically clean the translation cache"""
        while self._running:
            time.sleep(3600)  # Run hourly
            with self.lock:
                # Remove entries older than 24 hours
                current_time = time.time()
                for lang in list(self.translation_cache.keys()):
                    self.translation_cache[lang] = {
                        k: v for k, v in self.translation_cache[lang].items()
                        if current_time - v['timestamp'] < 86400
                    }
                logger.info("Performed cache maintenance")

    @lru_cache(maxsize=1000)
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of input text with confidence score"""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        try:
            if self.language_model:
                # Use FastText for more accurate detection
                predictions = self.language_model.predict(text.replace("\n", " "), k=1)
                lang_code = predictions[0][0].replace('__label__', '')
                confidence = float(predictions[1][0])
                
                # Map to Google Translate codes if needed
                lang_code = self._map_language_code(lang_code)
            else:
                # Fallback to langdetect
                lang_code = langdetect.detect(text)
                confidence = 0.9  # Default confidence for langdetect
            
            detection_time = time.time() - start_time
            self.metrics['detection_time'] = 0.9 * self.metrics['detection_time'] + 0.1 * detection_time
            
            logger.info(f"Detected language: {lang_code} (confidence: {confidence:.2f})")
            return lang_code, confidence
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return 'en', 0.0  # Default to English with low confidence

    def _map_language_code(self, code: str) -> str:
        """Map language codes to standard format"""
        code = code.lower()
        if code == 'zh':
            return 'zh-cn'
        if code == 'pt-br':
            return 'pt'
        return code

    def translate_text(self, text: str, src_lang: Optional[str] = None, 
                      dest_lang: str = 'en', use_cache: bool = True) -> str:
        """Translate text between languages with caching"""
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        # Check cache first
        cache_key = (text, src_lang, dest_lang)
        if use_cache:
            with self.lock:
                if dest_lang in self.translation_cache and text in self.translation_cache[dest_lang]:
                    self.metrics['cache_hits'] += 1
                    return self.translation_cache[dest_lang][text]['translation']
        
        try:
            # Auto-detect source language if not provided
            if not src_lang:
                src_lang, _ = self.detect_language(text)
            
            # Skip translation if source and target are same
            if src_lang == dest_lang:
                return text
            
            # Perform translation
            translated = self.translator.translate(text, src=src_lang, dest=dest_lang)
            result = translated.text
            
            # Update cache
            with self.lock:
                self.translation_cache[dest_lang][text] = {
                    'translation': result,
                    'timestamp': time.time()
                }
            
            translation_time = time.time() - start_time
            self.metrics['translation_time'] = 0.9 * self.metrics['translation_time'] + 0.1 * translation_time
            
            logger.info(f"Translated '{text[:30]}...' from {src_lang} to {dest_lang}")
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text  # Return original text on failure

    def translate_to_english(self, text: str) -> str:
        """Convenience method to translate any text to English"""
        return self.translate_text(text, dest_lang='en')

    def translate_from_english(self, text: str, dest_lang: str) -> str:
        """Convenience method to translate from English to target language"""
        return self.translate_text(text, src_lang='en', dest_lang=dest_lang)

    def batch_translate(self, texts: List[str], src_lang: Optional[str] = None,
                       dest_lang: str = 'en') -> List[str]:
        """Translate multiple texts in parallel"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                lambda x: self.translate_text(x, src_lang, dest_lang),
                texts
            ))
        return results

    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names"""
        return self.supported_languages

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return {
            'detection_accuracy': self.metrics['detection_accuracy'],
            'avg_detection_time': self.metrics['detection_time'],
            'avg_translation_time': self.metrics['translation_time'],
            'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['total_requests']),
            'total_requests': self.metrics['total_requests']
        }

    def shutdown(self):
        """Clean shutdown of the language support"""
        self._running = False


# Singleton instance
language_support = LanguageSupport()

# Public interface functions
def detect_language(text: str) -> Tuple[str, float]:
    """Detect language of input text with confidence score"""
    return language_support.detect_language(text)

def translate_text(text: str, src_lang: Optional[str] = None, 
                  dest_lang: str = 'en') -> str:
    """Translate text between languages"""
    return language_support.translate_text(text, src_lang, dest_lang)

def translate_to_english(text: str) -> str:
    """Translate any text to English"""
    return language_support.translate_to_english(text)

def translate_from_english(text: str, dest_lang: str) -> str:
    """Translate from English to target language"""
    return language_support.translate_from_english(text, dest_lang)

def get_supported_languages() -> Dict[str, str]:
    """Get supported languages"""
    return language_support.get_supported_languages()

def shutdown_language_support():
    """Clean shutdown"""
    language_support.shutdown()


# Test code
if __name__ == '__main__':
    test_phrases = [
        "Bonjour, comment ça va?",
        "你今天怎么样？",
        "Hola, estás despierto?",
        "Wie spät ist es?",
        "I'm awake.",
        "أنا مستيقظ",  # Arabic
        "मैं जाग रहा हूँ",  # Hindi
        "Estou acordado"  # Portuguese
    ]

    print("Language Support Test\n" + "="*40)
    
    for phrase in test_phrases:
        print(f"\nOriginal: {phrase}")
        
        # Detect language
        lang, confidence = detect_language(phrase)
        lang_name = get_supported_languages().get(lang, lang)
        print(f"Detected: {lang_name} (confidence: {confidence:.2f})")
        
        # Translate to English
        en_text = translate_to_english(phrase)
        print(f"English: {en_text}")
        
        # Translate back to original
        if lang in get_supported_languages():
            back_text = translate_from_english(en_text, lang)
            print(f"Back Translation: {back_text}")
        
        time.sleep(1)  # Avoid rate limiting

    # Print performance metrics
    print("\nPerformance Metrics:")
    for k, v in language_support.get_performance_metrics().items():
        print(f"{k.replace('_', ' ').title()}: {v:.4f}")

    shutdown_language_support()