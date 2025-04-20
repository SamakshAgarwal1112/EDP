import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional
import logging
import asyncio
import albumentations as A
from imblearn.over_sampling import SMOTE
import face_recognition
import hashlib
from cryptography.fernet import Fernet
import matplotlib.pyplot as plt
import json
from pathlib import Path
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, config_path: str = "config/settings.json"):
        self.config = self.load_config(config_path)
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # India-specific augmentation pipeline
        self.augmentation = A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=(-0.3, 0.3)),  # Simulate low-light
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Indian road noise
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.CLAHE(p=0.2),  # Enhance contrast for diverse skin tones
            A.RandomGamma(p=0.2)  # Simulate glare
        ])
        
        self.load_stats = {
            'total_samples': 0,
            'class_distribution': {},
            'load_errors': 0
        }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from settings.json"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f).get('dataset_loader', {})
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {
                'target_size': (48, 48),
                'grayscale': True,
                'augment': True,
                'balance': True,
                'frames_per_video': 30,
                'dataset_path': '../../datasets'
            }

    def verify_file(self, path: str) -> bool:
        """Verify file integrity"""
        try:
            with open(path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            logger.debug(f"File {path} checksum: {checksum}")
            return True
        except Exception as e:
            logger.error(f"File verification failed for {path}: {e}")
            return False

    async def _load_single_image(self, img_path: str, target_size: Tuple[int, int], grayscale: bool = True) -> Optional[np.ndarray]:
        """Load and preprocess a single image with face cropping"""
        for _ in range(2):  # Retry once
            try:
                if not self.verify_file(img_path):
                    raise ValueError(f"Invalid file: {img_path}")
                
                color_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
                img = cv2.imread(img_path, color_mode)
                if img is None:
                    raise ValueError(f"Failed to read image: {img_path}")
                
                # Face detection and cropping
                faces = face_recognition.face_locations(img, model="hog")
                if faces:
                    top, right, bottom, left = faces[0]
                    img = img[top:bottom, left:right]
                
                img = self._smart_resize(img, target_size)
                img = img.astype(np.float32) / 255.0
                
                if not grayscale and len(img.shape) == 2:
                    img = np.stack((img,)*3, axis=-1)
                
                return img
            
            except Exception as e:
                logger.warning(f"Error loading {img_path}: {e}")
                self.load_stats['load_errors'] += 1
                await asyncio.sleep(0.1)
        return None

    def _smart_resize(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image while preserving aspect ratio"""
        h, w = img.shape[:2]
        target_h, target_w = target_size
        scale = min(target_w/w, target_h/h)
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        delta_w = target_w - new_size[0]
        delta_h = target_h - new_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0] * img.shape[2] if len(img.shape) == 3 else 0
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    async def _parallel_load(self, file_paths: List[str], target_size: Tuple[int, int], grayscale: bool = True) -> List[Optional[np.ndarray]]:
        """Load images asynchronously"""
        tasks = [self._load_single_image(path, target_size, grayscale) for path in file_paths]
        return await asyncio.gather(*tasks)

    def _balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using SMOTE"""
        try:
            smote = SMOTE(random_state=42)
            X_reshaped = X.reshape(X.shape[0], -1)
            X_res, y_res = smote.fit_resample(X_reshaped, y)
            return X_res.reshape(-1, *X.shape[1:]), y_res
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original dataset.")
            return X, y

    async def load_mrl_eye_dataset(self, base_path: str = None, target_size: Tuple[int, int] = None,
                                 augment: bool = None, balance: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load MRL Eye Dataset with India-specific preprocessing"""
        base_path = base_path or os.path.join(self.config.get('dataset_path'), 'MRL_Dataset')
        target_size = target_size or self.config.get('target_size')
        augment = augment if augment is not None else self.config.get('augment')
        balance = balance if balance is not None else self.config.get('balance')
        
        data = []
        labels = []
        file_paths = []
        
        logger.info(f"Loading MRL Eye Dataset from: {base_path}")
        
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = 1 if 'open' in file.lower() else 0
                    file_paths.append((os.path.join(root, file), label))
                    self.load_stats['class_distribution'][label] = self.load_stats['class_distribution'].get(label, 0) + 1
        
        paths, labels = zip(*file_paths)
        loaded_images = await self._parallel_load(paths, target_size)
        
        data = [img for img in loaded_images if img is not None]
        labels = [label for img, label in zip(loaded_images, labels) if img is not None]
        self.load_stats['total_samples'] = len(data)
        
        data = np.array(data).reshape(-1, *target_size, 1)
        labels = np.array(labels)
        
        if balance:
            data, labels = self._balance_dataset(data, labels)
        
        if augment:
            augmented_data = []
            augmented_labels = []
            for img, label in zip(data, labels):
                augmented = self.augmentation(image=img.squeeze())['image']
                augmented_data.append(augmented)
                augmented_labels.append(label)
            data = np.concatenate([data, np.array(augmented_data).reshape(-1, *target_size, 1)])
            labels = np.concatenate([labels, np.array(augmented_labels)])
        
        # Encrypt load stats
        encrypted_stats = self.cipher.encrypt(json.dumps(self.load_stats).encode())
        logger.debug(f"Encrypted stats: {encrypted_stats}")
        
        logger.info(f"Loaded {len(data)} samples (Errors: {self.load_stats['load_errors']})")
        logger.info(f"Class distribution: {self.load_stats['class_distribution']}")
        
        return train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    async def load_generic_image_dataset(self, base_path: str = None, label_map: Dict[str, int] = None,
                                      target_size: Tuple[int, int] = None, grayscale: bool = None,
                                      augment: bool = None, balance: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load generic image dataset (e.g., DDD)"""
        base_path = base_path or os.path.join(self.config.get('dataset_path'), 'DDD')
        label_map = label_map or {"awake": 0, "drowsy": 1, "extreme": 2}
        target_size = target_size or self.config.get('target_size')
        grayscale = grayscale if grayscale is not None else self.config.get('grayscale')
        augment = augment if augment is not None else self.config.get('augment')
        balance = balance if balance is not None else self.config.get('balance')
        
        data = []
        labels = []
        file_paths = []
        
        logger.info(f"Loading generic dataset from: {base_path}")
        
        for label_name, label_id in label_map.items():
            label_path = os.path.join(base_path, label_name)
            if not os.path.isdir(label_path):
                logger.warning(f"Label directory not found: {label_path}")
                continue
            for img_file in os.listdir(label_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_paths.append((os.path.join(label_path, img_file), label_id))
                    self.load_stats['class_distribution'][label_id] = self.load_stats['class_distribution'].get(label_id, 0) + 1
        
        paths, labels = zip(*file_paths)
        loaded_images = await self._parallel_load(paths, target_size, grayscale)
        
        data = [img for img in loaded_images if img is not None]
        labels = [label for img, label in zip(loaded_images, labels) if img is not None]
        self.load_stats['total_samples'] = len(data)
        
        channels = 1 if grayscale else 3
        data = np.array(data).reshape(-1, *target_size, channels)
        labels = np.array(labels)
        
        if balance:
            data, labels = self._balance_dataset(data, labels)
        
        if augment:
            augmented_data = []
            augmented_labels = []
            for img, label in zip(data, labels):
                augmented = self.augmentation(image=img.squeeze())['image']
                augmented_data.append(augmented)
                augmented_labels.append(label)
            data = np.concatenate([data, np.array(augmented_data).reshape(-1, *target_size, channels)])
            labels = np.concatenate([labels, np.array(augmented_labels)])
        
        encrypted_stats = self.cipher.encrypt(json.dumps(self.load_stats).encode())
        logger.debug(f"Encrypted stats: {encrypted_stats}")
        
        logger.info(f"Loaded {len(data)} samples (Errors: {self.load_stats['load_errors']})")
        logger.info(f"Class distribution: {self.load_stats['class_distribution']}")
        
        return train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    async def load_uta_rldd(self, base_path: str = None, target_size: Tuple[int, int] = None,
                           frames_per_video: int = None, anonymize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load UTA RLDD video dataset with face anonymization"""
        base_path = base_path or os.path.join(self.config.get('dataset_path'), 'UTA_RLDD')
        target_size = target_size or self.config.get('target_size')
        frames_per_video = frames_per_video or self.config.get('frames_per_video')
        
        data = []
        labels = []
        label_map = {"awake": 0, "drowsy": 1, "extreme": 2}
        
        logger.info(f"Loading UTA RLDD dataset from: {base_path}")
        
        for label_name, label_id in label_map.items():
            label_path = os.path.join(base_path, label_name)
            if not os.path.isdir(label_path):
                logger.warning(f"Label directory not found: {label_path}")
                continue
            for video_file in os.listdir(label_path):
                if video_file.lower().endswith(('.mp4', '.avi')):
                    video_path = os.path.join(label_path, video_file)
                    if not self.verify_file(video_path):
                        self.load_stats['load_errors'] += 1
                        continue
                    frames = await self._load_video_frames(video_path, target_size, frames_per_video, anonymize)
                    data.extend(frames)
                    labels.extend([label_id] * len(frames))
                    self.load_stats['class_distribution'][label_id] = self.load_stats['class_distribution'].get(label_id, 0) + len(frames)
        
        data = np.array(data).reshape(-1, *target_size, 1)
        labels = np.array(labels)
        self.load_stats['total_samples'] = len(data)
        
        data, labels = self._balance_dataset(data, labels)
        
        encrypted_stats = self.cipher.encrypt(json.dumps(self.load_stats).encode())
        logger.debug(f"Encrypted stats: {encrypted_stats}")
        
        logger.info(f"Loaded {len(data)} samples (Errors: {self.load_stats['load_errors']})")
        logger.info(f"Class distribution: {self.load_stats['class_distribution']}")
        
        return train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    async def _load_video_frames(self, video_path: str, target_size: Tuple[int, int], 
                               frames_per_video: int, anonymize: bool) -> List[np.ndarray]:
        """Load and preprocess video frames"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, frames_per_video, dtype=int)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # IR-compatible
                if anonymize:
                    faces = face_recognition.face_locations(frame)
                    for top, right, bottom, left in faces:
                        frame[top:bottom, left:right] = cv2.GaussianBlur(frame[top:bottom, left:right], (99, 99), 30)
                frame = self._smart_resize(frame, target_size)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        
        cap.release()
        return frames

    def plot_sample_images(self, X: np.ndarray, y: np.ndarray, class_names: Dict[int, str], n_samples: int = 5):
        """Plot sample images with Streamlit option"""
        if 'streamlit' in self.config.get('visualization_mode', 'matplotlib'):
            st.header("Dataset Sample Images")
            for cls in np.unique(y):
                cls_indices = np.where(y == cls)[0]
                sample_indices = np.random.choice(cls_indices, n_samples, replace=False)
                st.subheader(class_names[cls])
                cols = st.columns(n_samples)
                for idx, col in zip(sample_indices, cols):
                    with col:
                        st.image(X[idx].squeeze(), clamp=True, caption=f"Sample {idx}")
        else:
            plt.figure(figsize=(15, 8))
            unique_classes = np.unique(y)
            for i, cls in enumerate(unique_classes):
                cls_indices = np.where(y == cls)[0]
                sample_indices = np.random.choice(cls_indices, n_samples, replace=False)
                for j, idx in enumerate(sample_indices):
                    plt_idx = i * n_samples + j + 1
                    plt.subplot(len(unique_classes), n_samples, plt_idx)
                    img = X[idx].squeeze()
                    plt.imshow(img, cmap='gray')
                    plt.title(f"{class_names[cls]}")
                    plt.axis('off')
            plt.tight_layout()
            plt.show()

    def get_load_stats(self) -> Dict[str, Any]:
        """Get encrypted load statistics"""
        with self.lock:
            return self.cipher.decrypt(json.dumps(self.load_stats).encode()).decode()

async def main():
    loader = DatasetLoader()
    
    # Test MRL Eye Dataset
    print("\nLoading MRL Eye Dataset:")
    X_train, X_test, y_train, y_test = await loader.load_mrl_eye_dataset()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    loader.plot_sample_images(X_train, y_train, {0: "Closed", 1: "Open"})
    
    # Test DDD Dataset
    print("\nLoading DDD Dataset:")
    label_map = {"awake": 0, "drowsy": 1, "extreme": 2}
    X_train, X_test, y_train, y_test = await loader.load_generic_image_dataset(label_map=label_map)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    loader.plot_sample_images(X_train, y_train, {v: k for k, v in label_map.items()})
    
    # Test UTA RLDD
    print("\nLoading UTA RLDD Dataset:")
    X_train, X_test, y_train, y_test = await loader.load_uta_rldd()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    loader.plot_sample_images(X_train, y_train, {0: "Awake", 1: "Drowsy", 2: "Extreme"})

if __name__ == "__main__":
    asyncio.run(main())