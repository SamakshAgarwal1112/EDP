import time
import threading
import logging
from typing import Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import signal
import sys
from datetime import datetime

from modules import eye_tracker, voice_interface, alert_system, convo_engine
from utils import language_support

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drowsiness_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

class DrowsinessOrchestrator:
    def __init__(self):
        self.state = SystemState()
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Initialize modules
        self._initialize_modules()

    def _initialize_modules(self):
        """Initialize all system modules"""
        try:
            logger.info("Initializing system modules...")
            
            # Start voice interface first as it's needed for other modules
            voice_interface.start_voice_listener()
            self._update_module_status('voice_interface', 'active')
            
            # Initialize language support
            language_support.detect_language("Test")  # Warm-up
            
            # Start eye tracker with callbacks
            eye_tracker.start_eye_tracker(
                alert_callback=self._alert_handler,
                is_speaking_func=voice_interface.is_user_speaking
            )
            self._update_module_status('eye_tracker', 'active')
            
            # Start conversation engine
            convo_engine.start_conversation_handler()
            self._update_module_status('convo_engine', 'active')
            
            logger.info("All modules initialized successfully")
            
        except Exception as e:
            logger.error(f"Module initialization failed: {e}")
            self._shutdown_system()

    def _alert_handler(self, mode: str, is_speaking: bool):
        """Handle alert callbacks from eye tracker"""
        with self.lock:
            self.state.last_alert_time = time.time()
            
            # Log the alert event
            alert_level = "NORMAL" if mode == "normal" else "EXTREME"
            logger.warning(f"Alert triggered - Level: {alert_level}, User speaking: {is_speaking}")
            
            # Update module status
            self._update_module_status('alert_system', 'active')
            
            # Handle alert in a separate thread
            self.executor.submit(alert_system.handle_alert, mode=mode, is_speaking=is_speaking)

    def _update_module_status(self, module: str, status: str):
        """Update module status in system state"""
        with self.lock:
            self.state.performance_metrics['modules'][module]['status'] = status
            self.state.performance_metrics['modules'][module]['last_update'] = time.time()

    def _handle_user_interaction(self, speech: Optional[str] = None):
        """Process user interaction and update system state"""
        with self.lock:
            self.state.last_interaction_time = time.time()
            
            if speech:
                logger.info(f"User interaction detected: {speech[:50]}...")
                
                # Handle speech input
                self.executor.submit(convo_engine.handle_input, speech)
                
                # Stop any active alerts since user is responding
                alert_system.stop_alerts()
                self._update_module_status('alert_system', 'inactive')

    def _monitor_system_health(self):
        """Check health of all system components"""
        with self.lock:
            current_time = time.time()
            inactive_threshold = 30  # seconds
            
            for module, data in self.state.performance_metrics['modules'].items():
                if data['status'] == 'active' and (current_time - data['last_update']) > inactive_threshold:
                    logger.warning(f"Module {module} appears inactive")
                    data['status'] = 'unresponsive'

    def _handle_shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        logger.info(f"Received shutdown signal {signum}")
        self._shutdown_system()
        sys.exit(0)

    def _shutdown_system(self):
        """Shut down all system components"""
        with self.lock:
            if not self.state.running:
                return
                
            self.state.running = False
            logger.info("Initiating system shutdown...")
            
            # Shutdown modules in reverse initialization order
            alert_system.stop_alerts()
            eye_tracker.stop_eye_tracker()
            convo_engine.stop_conversation_handler()
            voice_interface.stop_voice_listener()
            language_support.shutdown_language_support()
            
            self.executor.shutdown(wait=True)
            logger.info("System shutdown complete")

    def run(self):
        """Main system orchestration loop"""
        logger.info("Starting Drowsiness Detection System")
        
        try:
            iteration_count = 0
            while self.state.running:
                iteration_start = time.time()
                iteration_count += 1
                
                # Check for user speech
                speech = voice_interface.get_latest_speech()
                if speech:
                    self._handle_user_interaction(speech)
                
                # Check system health periodically
                if iteration_count % 30 == 0:
                    self._monitor_system_health()
                
                # Log system status periodically
                if iteration_count % 60 == 0:
                    status = eye_tracker.get_drowsy_state()
                    logger.info(
                        f"System Status | Score: {status['score']} "
                        f"| Extreme: {status['extreme']} "
                        f"| User speaking: {voice_interface.is_user_speaking()}"
                    )
                
                # Calculate loop timing
                loop_time = time.time() - iteration_start
                with self.lock:
                    self.state.performance_metrics['loop_iterations'] = iteration_count
                    # Exponential moving average for loop time
                    self.state.performance_metrics['avg_loop_time'] = (
                        0.9 * self.state.performance_metrics['avg_loop_time'] + 
                        0.1 * loop_time
                    )
                
                time.sleep(0.1)  # Prevent CPU overutilization
                
        except Exception as e:
            logger.error(f"Fatal error in orchestrator loop: {e}")
            self._shutdown_system()
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        with self.lock:
            status = {
                'running': self.state.running,
                'uptime': time.time() - self.state.last_interaction_time,
                'last_alert': self.state.last_alert_time,
                'last_interaction': self.state.last_interaction_time,
                'performance': self.state.performance_metrics,
                'drowsiness': eye_tracker.get_drowsy_state(),
                'voice_active': voice_interface.is_user_speaking()
            }
            return status


def main():
    """Entry point for the drowsiness detection system"""
    orchestrator = DrowsinessOrchestrator()
    
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        orchestrator._shutdown_system()
    except Exception as e:
        logger.error(f"System crashed: {e}")
        orchestrator._shutdown_system()
        raise


if __name__ == '__main__':
    main()