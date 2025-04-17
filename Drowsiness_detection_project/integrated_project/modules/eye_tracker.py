import cv2
import numpy as np
import time
import threading
import dlib
from keras.models import load_model
from collections import deque
from scipy.spatial import distance as dist
from imutils import face_utils
import logging
from typing import Optional, Callable, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EYE_AR_THRESH = 0.25  # Eye Aspect Ratio threshold
EYE_AR_CONSEC_FRAMES = 20  # Consecutive frames threshold
MAR_THRESH = 0.5  # Mouth Aspect Ratio threshold for yawn detection
HEAD_TILT_THRESH = 20  # Degrees threshold for head tilt detection
BLINK_RATE_THRESH = 0.2  # Blinks per second threshold
PERCLOS_THRESH = 0.8  # Percentage of eye closure threshold

# Drowsiness levels
DROWSINESS_LEVELS = {
    "alert": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "extreme": 4
}

# Load models
def load_models():
    """Load all required models with error handling"""
    models = {}
    try:
        # CNN model for eye state classification
        models['eye_state'] = load_model("integrated_project/modules/models/cnn_eye_model.h5")
        
        # dlib's facial landmark predictor
        models['landmark_predictor'] = dlib.shape_predictor(
            "integrated_project/modules/models/shape_predictor_68_face_landmarks.dat"
        )
        
        # Haar cascades as fallback
        models['face_cascade'] = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        models['eye_cascade'] = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise
    
    return models

class EyeTracker:
    def __init__(self):
        self.models = load_models()
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
        
        # Thread-safe variables
        self.lock = threading.Lock()
        self.alert_callback = None
        self.speaking_flag_checker = None
        
        # Performance metrics
        self.metrics = {
            "processing_time": 0,
            "detection_accuracy": 0,
            "frame_rate": 0
        }
        
        # Initialize facial landmark indices
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio for given eye landmarks"""
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        return ear

    def mouth_aspect_ratio(self, mouth):
        """Calculate mouth aspect ratio for yawn detection"""
        # Compute the euclidean distances between the three sets of
        # vertical mouth landmarks (x, y)-coordinates
        A = dist.euclidean(mouth[13], mouth[19])  # 51, 59
        B = dist.euclidean(mouth[14], mouth[18])  # 53, 57
        C = dist.euclidean(mouth[15], mouth[17])  # 55, 56

        # Compute the euclidean distance between the horizontal
        # mouth landmark (x, y)-coordinates
        D = dist.euclidean(mouth[12], mouth[16])  # 49, 55

        # Compute mouth aspect ratio
        mar = (A + B + C) / (3.0 * D)

        return mar

    def head_pose_estimation(self, shape, frame):
        """Estimate head pose using facial landmarks"""
        # 3D model points (approximate)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),      # Left eye left corner
            (225.0, 170.0, -135.0),       # Right eye right corner
            (-150.0, -150.0, -125.0),     # Left Mouth corner
            (150.0, -150.0, -125.0)       # Right mouth corner
        ])

        # 2D image points from landmarks
        image_points = np.array([
            shape[30],                   # Nose tip
            shape[8],                     # Chin
            shape[36],                    # Left eye left corner
            shape[45],                    # Right eye right corner
            shape[48],                    # Left Mouth corner
            shape[54]                     # Right mouth corner
        ], dtype="double")

        # Camera internals
        size = frame.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # Solve PnP
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Calculate Euler angles
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        return angles

    def preprocess_eye(self, eye):
        """Preprocess eye image for CNN model"""
        eye = cv2.resize(eye, (24, 24))
        eye = cv2.equalizeHist(eye)  # Improve contrast
        eye = eye / 255.0
        eye = eye.reshape(24, 24, 1)
        eye = np.expand_dims(eye, axis=0)
        return eye

    def calculate_perclos(self):
        """Calculate PERCLOS (percentage of eye closure) metric"""
        if len(self.eye_ar_history) == 0:
            return 0
        closed_frames = sum(ear < EYE_AR_THRESH for ear in self.eye_ar_history)
        return closed_frames / len(self.eye_ar_history)

    def calculate_blink_rate(self):
        """Calculate blink rate in blinks per second"""
        current_time = time.time()
        time_elapsed = current_time - self.last_blink_time
        if time_elapsed == 0:
            return 0
        return self.blink_count / time_elapsed

    def update_drowsiness_score(self, eyes_closed: bool, yawn_detected: bool, head_tilt: bool):
        """Update drowsiness score based on multiple factors"""
        with self.lock:
            # Base score adjustment
            if eyes_closed:
                self.drowsiness_score += 2
            else:
                self.drowsiness_score = max(0, self.drowsiness_score - 1)
            
            # Additional factors
            if yawn_detected:
                self.drowsiness_score += 3
                self.yawn_count += 1
                
            if head_tilt:
                self.drowsiness_score += 1
                self.head_tilt_detected = True
            
            # Add PERCLOS factor
            perclos = self.calculate_perclos()
            if perclos > PERCLOS_THRESH:
                self.drowsiness_score += 5
            
            # Add blink rate factor
            blink_rate = self.calculate_blink_rate()
            if blink_rate < BLINK_RATE_THRESH:
                self.drowsiness_score += 2
            
            # Cap the score
            self.drowsiness_score = min(100, max(0, self.drowsiness_score))
            self.score_buffer.append(self.drowsiness_score)
            
            # Update drowsiness level
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

    def detect_drowsiness(self, frame):
        """Main drowsiness detection pipeline"""
        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using dlib (more accurate than Haar cascades)
        detector = dlib.get_frontal_face_detector()
        rects = detector(gray, 0)
        
        eyes_closed = False
        yawn_detected = False
        head_tilt_detected = False
        
        for rect in rects:
            # Get facial landmarks
            shape = self.models['landmark_predictor'](gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Extract eye regions and compute EAR
            left_eye = shape[self.lStart:self.lEnd]
            right_eye = shape[self.rStart:self.rEnd]
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            
            # Average the eye aspect ratio together
            ear = (left_ear + right_ear) / 2.0
            self.eye_ar_history.append(ear)
            
            # Check for blink or eye closure
            if ear < EYE_AR_THRESH:
                self.eye_closed_frames += 1
                if self.eye_closed_frames >= EYE_AR_CONSEC_FRAMES:
                    eyes_closed = True
            else:
                if self.eye_closed_frames >= EYE_AR_CONSEC_FRAMES:
                    self.blink_count += 1
                    self.last_blink_time = time.time()
                self.eye_closed_frames = 0
            
            # Detect yawn
            mouth = shape[self.mStart:self.mEnd]
            mar = self.mouth_aspect_ratio(mouth)
            if mar > MAR_THRESH:
                yawn_detected = True
            
            # Estimate head pose
            angles = self.head_pose_estimation(shape, frame)
            if abs(angles[0]) > HEAD_TILT_THRESH or abs(angles[1]) > HEAD_TILT_THRESH:
                head_tilt_detected = True
            
            # Visualize landmarks (for debugging)
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            mouth_hull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)
            
            # Display EAR and MAR values
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display head pose angles
            cv2.putText(frame, f"X: {angles[0]:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Y: {angles[1]:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Fallback to Haar cascades if no faces detected with dlib
        if len(rects) == 0:
            faces = self.models['face_cascade'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.models['eye_cascade'].detectMultiScale(roi_gray)
                
                # Use CNN model for eye state classification
                for (ex, ey, ew, eh) in eyes:
                    eye = roi_gray[ey:ey+eh, ex:ex+ew]
                    eye_input = self.preprocess_eye(eye)
                    prediction = self.models['eye_state'].predict(eye_input)
                    if prediction[0][0] < 0.5:  # Eye closed
                        eyes_closed = True
                        break
        
        # Update drowsiness metrics
        self.update_drowsiness_score(eyes_closed, yawn_detected, head_tilt_detected)
        
        # Calculate processing metrics
        processing_time = time.time() - start_time
        self.metrics["processing_time"] = processing_time
        self.metrics["frame_rate"] = 1.0 / processing_time if processing_time > 0 else 0
        
        return frame

    def run_detection(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return
        
        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.error("Can't receive frame. Exiting...")
                break
            
            self.frame_count += 1
            processed_frame = self.detect_drowsiness(frame)
            
            # Display drowsiness information
            cv2.putText(processed_frame, f"Drowsiness: {self.drowsiness_score:.1f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            level_str = next(k for k, v in DROWSINESS_LEVELS.items() if v == self.current_level)
            cv2.putText(processed_frame, f"Level: {level_str}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Trigger alerts if needed
            if self.alert_callback and not self.speaking_flag_checker():
                if self.current_level >= DROWSINESS_LEVELS["extreme"]:
                    self.alert_callback(mode="extreme", is_speaking=False)
                elif self.current_level >= DROWSINESS_LEVELS["moderate"]:
                    self.alert_callback(mode="normal", is_speaking=False)
            
            # Show the frame
            cv2.imshow('Drowsiness Detection', processed_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.running = False

    def start(self, alert_callback: Callable, speaking_flag_checker: Callable):
        """Start the eye tracker in a separate thread"""
        self.alert_callback = alert_callback
        self.speaking_flag_checker = speaking_flag_checker
        self.thread = threading.Thread(target=self.run_detection)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Eye tracker started")

    def stop(self):
        """Stop the eye tracker"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
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


# Singleton instance for easy access
eye_tracker_instance = EyeTracker()

def start_eye_tracker(alert_callback: Callable, speaking_flag_checker: Callable):
    """Start the eye tracker system"""
    eye_tracker_instance.start(alert_callback, speaking_flag_checker)

def stop_eye_tracker():
    """Stop the eye tracker system"""
    eye_tracker_instance.stop()

def get_drowsy_state() -> Dict[str, Any]:
    """Get current drowsiness state"""
    return eye_tracker_instance.get_status()


# Test code
if __name__ == '__main__':
    def test_alert_callback(mode, is_speaking):
        print(f"Alert triggered - Mode: {mode}, Speaking: {is_speaking}")

    def test_speaking_check():
        return False

    try:
        start_eye_tracker(test_alert_callback, test_speaking_check)
        while True:
            status = get_drowsy_state()
            print(f"Current status: {status}")
            time.sleep(5)
    except KeyboardInterrupt:
        stop_eye_tracker()
        print("Stopped eye tracker")