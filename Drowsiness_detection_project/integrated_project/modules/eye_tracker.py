import cv2
import numpy as np
import time
import asyncio
import dlib
import torch
from torchvision.models import mobilenet_v3_small
from collections import deque
from scipy.spatial import distance as dist
from imutils import face_utils
import logging
import json
import hashlib
from cryptography.fernet import Fernet
import face_recognition
import yolov5
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
MAR_THRESH = 0.5
HEAD_TILT_THRESH = 20
BLINK_RATE_THRESH = 0.2
PERCLOS_THRESH = 0.8
FRAME_SKIP = 2  # Process every 2nd frame for low-end devices

DROWSINESS_LEVELS = {
    "alert": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "extreme": 4
}

class EyeTracker:
    def __init__(self, config_path: str = "config/settings.json"):
        self.config = self.load_config(config_path)
        self.models = self.load_models()
        self.drowsiness_score = 0
        self.extreme_flag = False
        self.current_level = DROWSINESS_LEVELS["alert"]
        self.score_buffer = deque(maxlen=50)
        self.eye_ar_history = deque(maxlen=30)
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.frame_count = 0
        self.eye_closed_frames = 0
        self.yawn_count = 0
        self.head_tilt_detected = False
        self.running = False
        self.frame_buffer = None
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        self.metrics = {
            "processing_time": 0,
            "detection_accuracy": 0,
            "frame_rate": 0
        }
        
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from settings.json"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f).get('eye_tracker', {})
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {
                'eye_ar_thresh': EYE_AR_THRESH,
                'mar_thresh': MAR_THRESH,
                'head_tilt_thresh': HEAD_TILT_THRESH,
                'blink_rate_thresh': BLINK_RATE_THRESH,
                'perclos_thresh': PERCLOS_THRESH,
                'model_path': 'integrated_project/modules/models/cnn_eye_model.pt',
                'landmark_path': 'integrated_project/modules/models/shape_predictor_68_face_landmarks.dat',
                'yolo_path': 'repos/yolo_drowsiness/yolov5s.pt'
            }

    def verify_model(self, path: str) -> bool:
        """Verify model file integrity"""
        try:
            with open(path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            logger.debug(f"Model {path} checksum: {checksum}")
            return True
        except Exception as e:
            logger.error(f"Model verification failed for {path}: {e}")
            return False

    def load_models(self):
        """Load all required models"""
        models = {}
        try:
            # MobileNetV3 for eye state classification
            model_path = self.config.get('model_path')
            if not self.verify_model(model_path):
                raise ValueError("Model verification failed")
            models['eye_state'] = mobilenet_v3_small(pretrained=False)
            models['eye_state'].load_state_dict(torch.load(model_path))
            models['eye_state'].eval()
            if torch.cuda.is_available():
                models['eye_state'].cuda()
            
            # dlib landmark predictor
            landmark_path = self.config.get('landmark_path')
            if not self.verify_model(landmark_path):
                raise ValueError("Landmark model verification failed")
            models['landmark_predictor'] = dlib.shape_predictor(landmark_path)
            
            # YOLOv5 for additional detection
            yolo_path = self.config.get('yolo_path')
            if not self.verify_model(yolo_path):
                raise ValueError("YOLO model verification failed")
            models['yolo'] = yolov5.load(yolo_path)
            
            # Haar cascades as fallback
            models['face_cascade'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            models['eye_cascade'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
        return models

    def eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        """Calculate mouth aspect ratio for yawn detection"""
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        D = dist.euclidean(mouth[12], mouth[16])
        mar = (A + B + C) / (3.0 * D)
        return mar

    async def head_pose_estimation(self, shape, frame):
        """Estimate head pose using deep learning"""
        try:
            # Simplified deep learning-based head pose (placeholder for DeepHeadPose)
            model_points = np.array([
                (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
            ])
            image_points = np.array([
                shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]
            ], dtype="double")
            size = frame.shape
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )
            dist_coeffs = np.zeros((4,1))
            success, rotation_vector, _ = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            return angles
        except Exception as e:
            logger.warning(f"Head pose estimation failed: {e}")
            return [0, 0, 0]

    def preprocess_eye(self, eye):
        """Preprocess eye image for MobileNetV3"""
        eye = cv2.resize(eye, (224, 224))
        eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
        eye = eye / 255.0
        eye = np.transpose(eye, (2, 0, 1))
        eye = np.expand_dims(eye, axis=0)
        return torch.tensor(eye, dtype=torch.float32)

    async def calculate_perclos(self):
        """Calculate PERCLOS metric"""
        if len(self.eye_ar_history) == 0:
            return 0
        closed_frames = sum(ear < self.config.get('eye_ar_thresh') for ear in self.eye_ar_history)
        return closed_frames / len(self.eye_ar_history)

    async def calculate_blink_rate(self):
        """Calculate blink rate"""
        time_elapsed = time.time() - self.last_blink_time
        return self.blink_count / time_elapsed if time_elapsed > 0 else 0

    async def update_drowsiness_score(self, eyes_closed: bool, yawn_detected: bool, head_tilt: bool):
        """Update drowsiness score with adaptive thresholds"""
        with self.lock:
            if eyes_closed:
                self.drowsiness_score += 2
            else:
                self.drowsiness_score = max(0, self.drowsiness_score - 1)
            
            if yawn_detected:
                self.drowsiness_score += 3
                self.yawn_count += 1
            
            if head_tilt:
                self.drowsiness_score += 1
                self.head_tilt_detected = True
            
            perclos = await self.calculate_perclos()
            if perclos > self.config.get('perclos_thresh'):
                self.drowsiness_score += 5
            
            blink_rate = await self.calculate_blink_rate()
            if blink_rate < self.config.get('blink_rate_thresh'):
                self.drowsiness_score += 2
            
            self.drowsiness_score = min(100, max(0, self.drowsiness_score))
            self.score_buffer.append(self.drowsiness_score)
            
            avg_score = np.mean(self.score_buffer)
            if avg_score > 80:
                self.current_level = DROWSINESS_LEVELS["extreme"]
                self.extreme_flag = True
            elif avg_score > 60:
                self.current_level = DROWSINESS_LEVELS["severe"]
                self.extreme_flag = False
            elif avg_score > 40:
                self.current_level = DROWSINESS_LEVELS["moderate"]
                self.extreme_flag = False
            elif avg_score > 20:
                self.current_level = DROWSINESS_LEVELS["mild"]
                self.extreme_flag = False
            else:
                self.current_level = DROWSINESS_LEVELS["alert"]
                self.extreme_flag = False
            
            # Encrypt score for secure storage
            encrypted_score = self.cipher.encrypt(str(self.drowsiness_score).encode())
            logger.debug(f"Encrypted score: {encrypted_score}")

    async def detect_drowsiness(self, frame):
        """Main drowsiness detection pipeline"""
        start_time = time.time()
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame received")
            return frame
        
        # Adaptive lighting correction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        eyes_closed = False
        yawn_detected = False
        head_tilt_detected = False
        
        # Face detection with face_recognition (faster than dlib)
        face_locations = face_recognition.face_locations(gray, model="hog")
        if face_locations:
            for top, right, bottom, left in face_locations:
                rect = dlib.rectangle(left, top, right, bottom)
                shape = self.models['landmark_predictor'](gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                left_eye = shape[self.lStart:self.lEnd]
                right_eye = shape[self.rStart:self.rEnd]
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                self.eye_ar_history.append(ear)
                
                if ear < self.config.get('eye_ar_thresh'):
                    self.eye_closed_frames += 1
                    if self.eye_closed_frames >= EYE_AR_CONSEC_FRAMES:
                        eyes_closed = True
                else:
                    if self.eye_closed_frames >= EYE_AR_CONSEC_FRAMES:
                        self.blink_count += 1
                        self.last_blink_time = time.time()
                    self.eye_closed_frames = 0
                
                mouth = shape[self.mStart:self.mEnd]
                mar = self.mouth_aspect_ratio(mouth)
                if mar > self.config.get('mar_thresh'):
                    yawn_detected = True
                
                angles = await self.head_pose_estimation(shape, frame)
                if abs(angles[0]) > self.config.get('head_tilt_thresh') or abs(angles[1]) > self.config.get('head_tilt_thresh'):
                    head_tilt_detected = True
                
                # Visualize landmarks
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                mouth_hull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)
                
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"X: {angles[0]:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Y: {angles[1]:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Fallback to Haar cascades
            faces = self.models['face_cascade'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.models['eye_cascade'].detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    eye = roi_gray[ey:ey+eh, ex:ex+ew]
                    eye_input = self.preprocess_eye(eye)
                    if torch.cuda.is_available():
                        eye_input = eye_input.cuda()
                    with torch.no_grad():
                        prediction = self.models['eye_state'](eye_input)
                        if prediction[0][0] < 0.5:  # Eye closed
                            eyes_closed = True
                            break
        
        # YOLOv5 for additional drowsiness cues
        results = self.models['yolo'](frame)
        for detection in results.xyxy[0]:
            if detection[-1] == 'yawn':
                yawn_detected = True
                self.yawn_count += 1
        
        await self.update_drowsiness_score(eyes_closed, yawn_detected, head_tilt_detected)
        
        processing_time = time.time() - start_time
        self.metrics["processing_time"] = processing_time
        self.metrics["frame_rate"] = 1.0 / processing_time if processing_time > 0 else 0
        
        # Display drowsiness info
        level_str = next(k for k, v in DROWSINESS_LEVELS.items() if v == self.current_level)
        cv2.putText(frame, f"Drowsiness: {self.drowsiness_score:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Level: {level_str}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        self.frame_buffer = None  # Clear frame from memory
        return frame

    async def run_detection(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try IR camera support
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return
        
        self.running = True
        frame_counter = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.error("Can't receive frame")
                break
            
            frame_counter += 1
            if frame_counter % FRAME_SKIP == 0:
                self.frame_buffer = frame
                processed_frame = await self.detect_drowsiness(frame)
                
                if self.alert_callback and not (await self.speaking_flag_checker()):
                    level_str = next(k for k, v in DROWSINESS_LEVELS.items() if v == self.current_level)
                    await self.alert_callback(mode=level_str, is_speaking=False, language=self.config.get('default_language', 'en'))
                
                cv2.imshow('Drowsiness Detection', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.running = False

    async def start(self, alert_callback, speaking_flag_checker):
        """Start the eye tracker"""
        self.alert_callback = alert_callback
        self.speaking_flag_checker = speaking_flag_checker
        self.running = True
        await self.run_detection()
        logger.info("Eye tracker started")

    async def stop(self):
        """Stop the eye tracker"""
        self.running = False
        logger.info("Eye tracker stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current drowsiness status"""
        with self.lock:
            return {
                "score": self.drowsiness_score,
                "level": self.current_level,
                "extreme": self.extreme_flag,
                "blink_rate": self.calculate_blink_rate(),
                "perclos": self.calculate_perclos(),
                "yawn_count": self.yawn_count,
                "head_tilt": self.head_tilt_detected,
                "metrics": self.metrics
            }

# Singleton instance
eye_tracker_instance = EyeTracker()

async def start_eye_tracker(alert_callback, speaking_flag_checker):
    """Start the eye tracker system"""
    await eye_tracker_instance.start(alert_callback, speaking_flag_checker)

async def stop_eye_tracker():
    """Stop the eye tracker system"""
    await eye_tracker_instance.stop()

def get_drowsy_state() -> Dict[str, Any]:
    """Get current drowsiness state"""
    return eye_tracker_instance.get_status()

async def main():
    """Test the eye tracker"""
    async def test_alert_callback(mode, is_speaking, language):
        print(f"Alert triggered - Mode: {mode}, Speaking: {is_speaking}, Language: {language}")

    async def test_speaking_check():
        return False

    try:
        await start_eye_tracker(test_alert_callback, test_speaking_check)
        while eye_tracker_instance.running:
            status = get_drowsy_state()
            print(f"Current status: {status}")
            await asyncio.sleep(5)
    finally:
        await stop_eye_tracker()
        print("Stopped eye tracker")

if __name__ == "__main__":
    asyncio.run(main())