import asyncio
import logging
from modules import eye_tracker, voice_interface, alert_system, convo_engine
from utils import language_support, dataset_loader
from main import DrowsinessOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_system():
    orchestrator = DrowsinessOrchestrator()
    
    try:
        # Start system
        logger.info("Starting system test...")
        run_task = asyncio.create_task(orchestrator.run())
        
        # Test 1: Eye Tracker
        await asyncio.sleep(2)
        status = eye_tracker.get_drowsy_state()
        logger.info(f"Eye Tracker Test: Score={status['score']}, Level={status['level']}")
        
        # Test 2: Voice Interface
        await voice_interface.speak("Test alert in Hindi")
        await asyncio.sleep(2)
        speech = await voice_interface.get_latest_speech()
        logger.info(f"Voice Interface Test: Recognized speech={speech}")
        
        # Test 3: Alert System
        await alert_system.handle_alert("mild", False, "hi", "Jaagte raho!")
        await asyncio.sleep(2)
        logger.info("Alert System Test: Mild alert triggered")
        
        # Test 4: Conversation Engine
        await convo_engine.handle_user_input("I'm awake")
        await asyncio.sleep(2)
        logger.info("Conversation Engine Test: Processed user input")
        
        # Test 5: Language Support
        lang, conf = await language_support.detect_language("मैं जाग रहा हूँ")
        translated = await language_support.translate_to_english("मैं जाग रहा हूँ")
        logger.info(f"Language Support Test: Detected={lang} ({conf:.2f}), Translated={translated}")
        
        # Test 6: Dataset Loader
        loader = dataset_loader.DatasetLoader()
        X_val, _, y_val, _ = await loader.load_uta_rldd()
        logger.info(f"Dataset Loader Test: Loaded {len(X_val)} UTA RLDD frames")
        
        # Test 7: System Status
        status = await orchestrator.get_system_status()
        logger.info(f"System Status Test: Running={status['running']}, Language={status['language']}")
        
        # Stop system
        await orchestrator._shutdown_system()
        logger.info("System test completed successfully")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        await orchestrator._shutdown_system()
        raise

if __name__ == "__main__":
    asyncio.run(test_system())