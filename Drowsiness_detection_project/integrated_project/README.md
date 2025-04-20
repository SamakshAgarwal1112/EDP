Getting Started

Stuff You Need
Laptop: Windows, macOS, or Linux with at least 4GB RAM (8GB is better).
Webcam: Built-in or USB (like your laptop’s default camera).
Microphone: Built-in or earphones with a mic.
Internet: For downloading files and translations (works offline after setup).
Disk Space: About 5GB for models, sounds, and libraries.



Python: Version 3.8 or 3.9 (not newer ones like 3.10).

Step-by-Step Setup

1. Download the Project
Go to the GitHub repo (I’ll share the link, or check our class group).
Click the green “Code” button and select “Download ZIP”.
Unzip the file to a folder (e.g., C:\Projects\drowsiness-detection).
Open the integrated_project folder inside it.

2. Install Python

Check if you have Python 3.8 or 3.9:
Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux).
Run: python --version
If it’s not 3.8 or 3.9, download from python.org.
Pick 3.8 or 3.9, and check “Add Python to PATH” during installation.
Verify again with python --version.
Tip: If you have Python 3.10 or higher, uninstall it first to avoid issues.

3. Set Up a Virtual Environment

In the integrated_project folder, open a terminal:
Windows: Right-click, select “Open Command Prompt here” (or use VS Code).
macOS/Linux: Use Terminal and cd to the folder.



Create a virtual environment:

python -m venv .venv

Activate it:

Windows: .venv\Scripts\activate
macOS/Linux: source .venv/bin/activate

You’ll see (.venv) in the terminal, meaning it’s ready.
Why? This keeps the project’s libraries separate, so your laptop’s Python stays clean.

4. Install Libraries

In the integrated_project folder (with (.venv) active), run:

pip install -r requirements.txt

This installs everything (takes 5-10 minutes). If it’s slow, try:

pip install --no-cache-dir -r requirements.txt
Fix: If you see “pip not found”, run:
python -m ensurepip --upgrade
python -m pip install --upgrade pip

5. Get Models, Sounds, and Configs

You need some files to make the system work. I’ve put them in a Google Drive link (check our class group or ask me). Download and place them like this:

Models (for eye and language detection):
cnn_eye_model_v2.h5
lid.176.bin
Put in: integrated_project/models/

Sounds (for alerts):

mild_alert_india.wav
extreme_alert_india.wav
Put in: integrated_project/sounds/

Noise Profile (for traffic noise):
indian_traffic_noise.wav
Put in: integrated_project/resources/

Configs (already in the ZIP, but check):
settings.json

conversation_kb.json
Should be in: integrated_project/config/

Note: Create models/, sounds/, and resources/ folders if they’re missing.

Optional Datasets (for testing accuracy):
MRL Eye Dataset (~1GB) and UTA RLDD (~2GB) from the same Google Drive.
Put in: integrated_project/datasets/MRL_Dataset/ and integrated_project/datasets/UTA_RLDD/.



You can skip these for basic use; the system runs without them.
Fix: If you can’t find the files, ping me, and I’ll share the link again.

6. Test Everything
Run the test script to make sure it’s working:
python test_system.py

You should see:

2025-04-19 10:00:00 - test_system - INFO - Starting system test...
[INFO] Eye Tracker Test: Score=10, Level=normal
[INFO] Voice Interface Test: Recognized speech=Hello
[INFO] Language Support Test: Detected=hi (0.95), Translated=I'm awake
[INFO] System test completed successfully

If it fails, check the Troubleshooting section below.

Running the System
Basic Command
Sit in front of your webcam, plug in your mic, and run:
python main.py --camera 0 --mic 1 --lang hi

What Happens

Your webcam turns on, watching your eyes.
If you close your eyes too long, you’ll hear “Jaagte raho!” or other Hindi alerts.
Speak (e.g., “Main jaag raha hoon”), and it’ll respond to keep you awake.
Open http://localhost:8501 in your browser to see a dashboard with your drowsiness score.


How to Find Camera/Mic Numbers:

Webcam: Run python -c "import cv2; print([i for i in range(3) if cv2.VideoCapture(i).isOpened()])".
Mic: Run python -c "import sounddevice; print(sounddevice.query_devices())".



Customizing It

Switch to Tamil
Try Tamil alerts like “Veḻiyāka iru!”:
python main.py --lang ta

Add Your Own Alert Sound

Record a .wav file (e.g., “Wake up!”) using Audacity or your phone.
Save it in integrated_project/sounds/ (e.g., my_alert.wav).


Edit integrated_project/config/settings.json:

"alert_system": {
  "audio": {
    "custom_sounds": {
      "mild": "sounds/my_alert.wav"
    }
  }
}

Change Alert Messages

Edit integrated_project/config/conversation_kb.json:

{
  "awake_response": {
    "hi": "Awesome, tum jaag rahe ho!"
  }
}

Want Punjabi?
Add "pu" to language_support.languages.supported in settings.json.

Update conversation_kb.json:

{
  "awake_response": {
    "pu": "Vadhia, tu jaag raha hai!"
  }
}

Run: python main.py --lang pu


Troubleshooting
Issue

How to Fix
Webcam not working
Run python -c "import cv2; print([i for i in range(3) if cv2.VideoCapture(i).isOpened()])". Try --camera 1 or check if another app is using the camera.
Mic not working
Run python -c "import sounddevice; print(sounddevice.query_devices())". Try --mic 0 or use earphones.


‘Model not found’
Check cnn_eye_model_v2.h5 and lid.176.bin are in integrated_project/models/.
Runs slow
Close other apps; edit settings.json to set eye_tracker.max_fps to 15.

No translations

Ensure internet is on; set language_support.fallback_language to en in settings.json.
Dashboard not at localhost:8501
Check settings.json for dashboard.enabled: true. Try http://127.0.0.1:8501.
Still Stuck? Look at integrated_project/logs/system.log for clues or ask me for help.


Common Questions

Q: Will it work on my old laptop?
Yep, it’s made for basic laptops with 4GB RAM. It even runs on a Raspberry Pi!

Q: Does it understand Hinglish?
Totally! It catches Hinglish and responds in Hindi or your chosen language.

Q: Is my data safe?
100%. No audio or video is saved, your face is blurred, and everything’s encrypted.

Q: Do I need the datasets?
Nope, you can skip them for basic use. They’re only for checking accuracy.

Q: What if I’m offline?
You need internet to set up and for translations, but it works offline with cached phrases.

Q: How do I know it’s running?
You’ll see logs like [CV] Eye tracker active and a dashboard at http://localhost:8501.

Extra Cool Stuff

Dashboard: Check your drowsiness score live on a webpage.
Vibration Alerts: If you hook up a vibration motor (like with Arduino), it buzzes.
Emergency Mode: Say “help” or “sos” to trigger an alert (more features coming).
Future Plans: I’m thinking of adding driver stats for fleets or phone notifications.



Make It Your Own
New Language: Add Gujarati or Punjabi to settings.json and conversation_kb.json.
Custom Alerts: Record your voice for alerts or use funny sounds.
Tweak Settings: Edit settings.json to change alert timing or drowsiness levels.



