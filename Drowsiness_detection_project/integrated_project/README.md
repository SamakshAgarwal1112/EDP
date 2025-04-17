# 🚗 AI-Powered Drowsiness Detection & Prevention System

![System Architecture Diagram](docs/system_architecture.png) *(Example diagram placeholder)*

A **real-time, multilingual intelligent system** that detects drowsiness using computer vision, engages users with conversational AI, and provides graduated alerts to prevent accidents.

---

## 🌟 Key Features

### 👁️ Computer Vision Module
- Real-time eye state classification (open/closed) using CNN
- PERCLOS (Percentage of Eye Closure) measurement
- Blink rate detection and head pose estimation
- Adaptive thresholding for different lighting conditions

### 🗣️ Voice Interaction System
- Speech-to-text with Whisper AI (multilingual support)
- Context-aware text-to-speech responses
- Voice activity detection for alert suppression
- Dynamic volume adjustment based on environment noise

### ⚠️ Smart Alert System
| Alert Level | Trigger Condition | Response |
|-------------|-------------------|----------|
| Mild | Drowsiness Score > 20 | Periodic verbal reminders |
| Moderate | Score > 40 | Increased frequency + sound alerts |
| Extreme | Score > 60 | Loud repeated alarms + emergency protocol |

### 🧠 Conversational Engine
- Dynamic response generation based on user state
- Intentional misinterpretation (20% chance) for engagement
- Multilingual support with automatic translation
- Context-aware dialog management

---

## 🏗️ System Architecture
drowsiness_detection_project/
├── datasets/ # Training datasets
│ ├── MRL_Dataset/ # 80,000+ eye images
│ ├── DDD/ # Driver Drowsiness Dataset
│ └── UTA_RLDD/ # Real-life driving videos
├── models/ # Pretrained models
│ ├── cnn_eye_model.h5 # Eye state classifier
│ └── whisper/ # STT models
├── integrated_project/ # Main application
│ ├── main.py # System orchestrator
│ ├── modules/ # Core components
│ │ ├── eye_tracker.py # CV pipeline
│ │ ├── voice_interface.py # Audio processing
│ │ ├── convo_engine.py # Dialog management
│ │ └── alert_system.py # Alert logic
│ ├── utils/ # Support utilities
│ │ ├── language_support.py # Translation services
│ │ └── dataset_loader.py # Data preprocessing
│ └── config/
│ └── settings.json # Runtime configuration



---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Webcam & microphone
- NVIDIA GPU recommended for real-time performance

```bash
# Clone repository
git clone https://github.com/yourrepo/drowsiness-detection.git
cd drowsiness-detection/integrated_project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python download_models.py

🚦 Running the System
Basic Usage

python main.py --camera 0 --mic 1

Command Line Options

Parameter	    Description	               Default
--camera	 Video source index	              0
--mic	      Audio input index	              1
--lang	     Default language	              en
--debug  	Enable debug mode	            False

Expected Output
[INFO] System initialized - Starting modules...
[CV] Eye tracker active (30 FPS)
[AUDIO] Voice interface ready (Whisper base)
[ALERT] Current status: Normal (Score: 15)


📊 Dataset Preparation
Supported Formats
Images: .jpg, .png (24x24 grayscale for eye model)

Videos: .mp4, .avi (for behavioral analysis)

Structure:

datasets/
  ├── train/
  │   ├── open_eyes/
  │   └── closed_eyes/
  └── test/
      ├── open_eyes/
      └── closed_eyes/


Use the dataset loader:
from utils.dataset_loader import load_mrl_eye_dataset
X_train, X_test, y_train, y_test = load_mrl_eye_dataset("datasets/MRL_Dataset")


🛠️ Customization Guide
Configuration Options
Edit config/settings.json:

{
  "eye_tracker": {
    "extreme_drowsy_score": 45,
    "blink_rate_threshold": 0.15
  },
  "alert_system": {
    "extreme_alert_interval": 3
  }
}

Adding New Languages
Add language code to conversation_engine.languages.supported

Provide translations in conversation_kb.json

Test with: python main.py --lang es

📈 Performance Metrics

Component	       Latency   	Accuracy
Eye Tracker	         15ms	     94%
Speech Recognition	 300ms	     89%
Alert Response	     50ms	      -

