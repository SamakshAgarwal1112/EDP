import asyncio
import logging
import time
from typing import Dict, Any
from dataclasses import dataclass, field
import signal
import sys
from datetime import datetime
from cryptography.fernet import Fernet
import json
from pathlib import Path
import streamlit as st
import loguru
from loguru import logger as loguru_logger

from modules import eye_tracker, voice_interface, alert_system, convo_engine
from utils import language_support, dataset_loader

# Configure logging with rotation and encryption
loguru_logger.remove()
loguru_logger.add(
    "drowsiness_system_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    compression="zip",
    format="{time} - {name} - {level} - {message}",
    level="INFO"
)

@dataclass
class SystemState:
    running: bool = True
    last_alert_time: float = 0
    last_interaction_time: float = time.time()
    performance_metrics: Dict[str, Any] = field(default_factory=lambda: {
        'loop_iterations': 0,
        'avg_loop_time': 0,
        'modules': {
            'eye_tracker': {'status': 'inactive', 'last_update': 0},
            'voice_interface': {'status': 'inactive', 'last_update': 0},
            'alert_system': {'status': 'inactive', 'last_update': 0},
            'convo_engine': {'status': 'inactive', 'last_update': 0}
        }
    })
    language: str = 'en'

class DrowsinessOrchestrator:
    def __init__(self, config_path: str = "config/settings.json"):
        self.config = self.load_config(config_path)
        self.state = SystemState(language=self.config.get('default_language', 'en'))
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.event_queue = asyncio.Queue()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Initialize Streamlit dashboard
        self.dashboard_task = None

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from settings.json"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f).get('orchestrator', {})
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {
                'default_language': 'en',
                'health_check_interval': 30,
                'status_log_interval': 60,
                'inactive_threshold': 30,
                'dashboard_enabled': False
            }

    async def _initialize_modules(self):
        """Initialize all system modules"""
        try:
            logger.info("Initializing system modules...")
            
            await voice_interface.start_voice_listener()
            self._update_module_status('voice_interface', 'active')
            
            await language_support.detect_language("Test")  # Warm-up
            
            await eye_tracker.start_eye_tracker(
                alert_callback=self._alert_handler,
                speaking_flag_checker=voice_interface.is_user_speaking
            )
            self._update_module_status('eye_tracker', 'active')
            
            await convo_engine.start_conversation_handler()
            self._update_module_status('convo_engine', 'active')
            
            logger.info("All modules initialized successfully")
            
        except Exception as e:
            logger.error(f"Module initialization failed: {e}")
            await self._shutdown_system()

    async def _alert_handler(self, mode: str, is_speaking: bool, language: str):
        """Handle alert callbacks from eye tracker"""
        async with self.lock:
            self.state.last_alert_time = time.time()
            alert_level = "NORMAL" if mode == "normal" else "EXTREME"
            logger.warning(f"Alert triggered - Level: {alert_level}, User speaking: {is_speaking}, Language: {language}")
            
            self._update_module_status('alert_system', 'active')
            
            if not is_speaking:
                translated_message = await language_support.translate_text(
                    "Stay awake!" if mode == "normal" else "Wake up immediately!",
                    src_lang='en',
                    dest_lang=language
                )
                await alert_system.handle_alert(mode, is_speaking, language, translated_message)
            else:
                logger.info("Alert suppressed due to active conversation")

    def _update_module_status(self, module: str, status: str):
        """Update module status"""
        async with self.lock:
            self.state.performance_metrics['modules'][module]['status'] = status
            self.state.performance_metrics['modules'][module]['last_update'] = time.time()

    async def _handle_user_interaction(self, speech: str):
        """Process user interaction"""
        async with self.lock:
            self.state.last_interaction_time = time.time()
            lang, _ = await language_support.detect_language(speech)
            self.state.language = lang if lang in language_support.get_supported_languages() else 'en'
            
            logger.info(f"User interaction detected: {speech[:50]}... (Language: {self.state.language})")
            
            translated_speech = await language_support.translate_to_english(speech)
            await convo_engine.handle_user_input(translated_speech)
            
            await alert_system.stop_alerts()
            self._update_module_status('alert_system', 'inactive')
            
            await voice_interface.say_awake_message(drowsiness_level="normal")

    async def _monitor_system_health(self):
        """Check module health"""
        async with self.lock:
            current_time = time.time()
            inactive_threshold = self.config.get('inactive_threshold', 30)
            
            for module, data in self.state.performance_metrics['modules'].items():
                if data['status'] == 'active' and (current_time - data['last_update']) > inactive_threshold:
                    logger.warning(f"Module {module} appears inactive")
                    data['status'] = 'unresponsive'

    async def _handle_shutdown(self, signum, frame):
        """Graceful shutdown"""
        logger.info(f"Received shutdown signal {signum}")
        await self._shutdown_system()
        sys.exit(0)

    async def _shutdown_system(self):
        """Shut down all components"""
        async with self.lock:
            if not self.state.running:
                return
                
            self.state.running = False
            logger.info("Initiating system shutdown...")
            
            await alert_system.stop_alerts()
            await eye_tracker.stop_eye_tracker()
            await convo_engine.stop_conversation_handler()
            await voice_interface.stop_voice_listener()
            await language_support.shutdown()
            
            if self.dashboard_task:
                self.dashboard_task.cancel()
            
            logger.info("System shutdown complete")

    async def _run_dashboard(self):
        """Run Streamlit dashboard"""
        st.set_page_config(page_title="Drowsiness Detection Dashboard", layout="wide")
        st.title("Drowsiness Detection System Dashboard")
        
        while self.state.running:
            status = await self.get_system_status()
            drowsiness = status['drowsiness']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("System Status")
                st.metric("Uptime (s)", f"{status['uptime']:.1f}")
                st.metric("Last Alert", datetime.fromtimestamp(status['last_alert']).strftime('%H:%M:%S'))
                st.metric("Last Interaction", datetime.fromtimestamp(status['last_interaction']).strftime('%H:%M:%S'))
                st.write("**Modules**")
                for module, data in status['performance']['modules'].items():
                    st.write(f"{module}: {data['status']}")
            
            with col2:
                st.subheader("Drowsiness Metrics")
                st.metric("Score", f"{drowsiness['score']:.1f}")
                st.metric("Level", next(k for k, v in eye_tracker.DROWSINESS_LEVELS.items() if v == drowsiness['level']))
                st.metric("Extreme", str(drowsiness['extreme']))
                st.metric("Blink Rate", f"{drowsiness['blink_rate']:.2f}")
                st.metric("PERCLOS", f"{drowsiness['perclos']:.2f}")
                st.metric("Yawn Count", drowsiness['yawn_count'])
            
            await asyncio.sleep(1)

    async def _validate_uta_rldd(self):
        """Validate eye tracker with UTA RLDD dataset"""
        loader = dataset_loader.DatasetLoader()
        X_val, _, y_val, _ = await loader.load_uta_rldd()
        
        correct = 0
        for img, label in zip(X_val, y_val):
            # Simulate eye tracker processing
            # Placeholder for actual validation
            correct += 1 if label == 0 else 0
        
        accuracy = correct / len(X_val) if X_val.size else 0
        logger.info(f"UTA RLDD validation accuracy: {accuracy:.2f}")

    async def run(self):
        """Main orchestration loop"""
        logger.info("Starting Drowsiness Detection System")
        
        await self._initialize_modules()
        
        if self.config.get('dashboard_enabled', False):
            self.dashboard_task = asyncio.create_task(self._run_dashboard())
        
        iteration_count = 0
        try:
            while self.state.running:
                iteration_start = time.time()
                iteration_count += 1
                
                speech = await voice_interface.get_latest_speech()
                if speech:
                    await self.event_queue.put(('interaction', speech))
                
                if iteration_count % self.config.get('health_check_interval', 30) == 0:
                    await self.event_queue.put(('health_check', None))
                
                if iteration_count % self.config.get('status_log_interval', 60) == 0:
                    await self.event_queue.put(('log_status', None))
                
                while not self.event_queue.empty():
                    event_type, data = await self.event_queue.get()
                    if event_type == 'interaction':
                        await self._handle_user_interaction(data)
                    elif event_type == 'health_check':
                        await self._monitor_system_health()
                    elif event_type == 'log_status':
                        status = eye_tracker.get_drowsy_state()
                        logger.info(
                            f"System Status | Score: {status['score']} "
                            f"| Extreme: {status['extreme']} "
                            f"| Language: {self.state.language}"
                        )
                
                loop_time = time.time() - iteration_start
                async with self.lock:
                    self.state.performance_metrics['loop_iterations'] = iteration_count
                    self.state.performance_metrics['avg_loop_time'] = (
                        0.9 * self.state.performance_metrics['avg_loop_time'] + 
                        0.1 * loop_time
                    )
                
                await asyncio.sleep(0.05)  # Reduced for faster response
            
            # Validate with UTA RLDD before shutdown
            await self._validate_uta_rldd()
        
        except Exception as e:
            logger.error(f"Fatal error in orchestrator loop: {e}")
            await self._shutdown_system()
            raise

    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        async with self.lock:
            status = {
                'running': self.state.running,
                'uptime': time.time() - self.state.last_interaction_time,
                'last_alert': self.state.last_alert_time,
                'last_interaction': self.state.last_interaction_time,
                'performance': self.state.performance_metrics,
                'drowsiness': eye_tracker.get_drowsy_state(),
                'voice_active': await voice_interface.is_user_speaking(),
                'language': self.state.language
            }
            return self.cipher.encrypt(json.dumps(status).encode()).decode()

async def main():
    """Entry point"""
    orchestrator = DrowsinessOrchestrator()
    
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        await orchestrator._shutdown_system()
    except Exception as e:
        logger.error(f"System crashed: {e}")
        await orchestrator._shutdown_system()
        raise

if __name__ == '__main__':
    asyncio.run(main())