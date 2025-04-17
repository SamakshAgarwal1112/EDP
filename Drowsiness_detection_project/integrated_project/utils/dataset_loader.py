import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self):
        # Image augmentation pipeline
        self.augmentation = A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        ])

        # Performance metrics
        self.load_stats = {
            'total_samples': 0,
            'class_distribution': {},
            'load_errors': 0
        }

    def _load_single_image(self, img_path: str, target_size: Tuple[int, int], grayscale: bool = True) -> Optional[np.ndarray]:
        """Load and preprocess a single image"""
        try:
            color_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            img = cv2.imread(img_path, color_mode)
            
            if img is None:
                raise ValueError(f"Failed to read image: {img_path}")
            
            # Resize with aspect ratio preservation
            img = self._smart_resize(img, target_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            # Convert grayscale to 3 channels if needed
            if not grayscale and len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)
                
            return img
            
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            self.load_stats['load_errors'] += 1
            return None

    def _smart_resize(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image while preserving aspect ratio"""
        h, w = img.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factors
        scale = min(target_w/w, target_h/h)
        
        # Resize with scaling
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        
        # Pad if needed
        delta_w = target_w - new_size[0]
        delta_h = target_h - new_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        
        color = [0] * img.shape[2] if len(img.shape) == 3 else 0
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    def _parallel_load(self, file_paths: List[str], target_size: Tuple[int, int], grayscale: bool = True) -> List[Optional[np.ndarray]]:
        """Load multiple images in parallel"""
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(
                lambda x: self._load_single_image(x, target_size, grayscale),
                file_paths
            ))
        return results

    def _balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance class distribution using oversampling"""
        ros = RandomOverSampler(random_state=42)
        X_reshaped = X.reshape(X.shape[0], -1)  # Flatten images for sampler
        X_res, y_res = ros.fit_resample(X_reshaped, y)
        return X_res.reshape(-1, *X.shape[1:]), y_res

    def load_mrl_eye_dataset(self, base_path: str, target_size: Tuple[int, int] = (24, 24), 
                           augment: bool = False, balance: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load MRL Eye Dataset with enhanced features"""
        data = []
        labels = []
        file_paths = []

        logger.info(f"Loading MRL Eye Dataset from: {base_path}")
        
        # First pass: collect all valid file paths
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = 1 if 'open' in file.lower() else 0
                    file_paths.append((os.path.join(root, file), label))
                    self.load_stats['class_distribution'][label] = self.load_stats['class_distribution'].get(label, 0) + 1

        # Parallel image loading
        paths, labels = zip(*file_paths)
        loaded_images = self._parallel_load(paths, target_size)
        
        # Filter out failed loads
        data = [img for img in loaded_images if img is not None]
        labels = [label for img, label in zip(loaded_images, labels) if img is not None]
        self.load_stats['total_samples'] = len(data)

        # Convert to numpy arrays
        data = np.array(data).reshape(-1, *target_size, 1)
        labels = np.array(labels)

        # Balance dataset if requested
        if balance:
            data, labels = self._balance_dataset(data, labels)

        # Data augmentation
        if augment:
            augmented_data = []
            augmented_labels = []
            for img, label in zip(data, labels):
                augmented = self.augmentation(image=img.squeeze())['image']
                augmented_data.append(augmented)
                augmented_labels.append(label)
            
            # Combine original and augmented data
            data = np.concatenate([data, np.array(augmented_data).reshape(-1, *target_size, 1)])
            labels = np.concatenate([labels, np.array(augmented_labels)])

        logger.info(f"Successfully loaded {len(data)} samples (Errors: {self.load_stats['load_errors']})")
        logger.info(f"Class distribution: {self.load_stats['class_distribution']}")

        return train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    def load_generic_image_dataset(self, base_path: str, label_map: Dict[str, int], 
                                 target_size: Tuple[int, int] = (64, 64), grayscale: bool = True,
                                 augment: bool = False, balance: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load generic image dataset with flexible structure"""
        data = []
        labels = []
        file_paths = []

        logger.info(f"Loading generic dataset from: {base_path}")
        
        # First pass: collect all valid file paths
        for label_name, label_id in label_map.items():
            label_path = os.path.join(base_path, label_name)
            if not os.path.isdir(label_path):
                logger.warning(f"Label directory not found: {label_path}")
                continue

            for img_file in os.listdir(label_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_paths.append((os.path.join(label_path, img_file), label_id))
                    self.load_stats['class_distribution'][label_id] = self.load_stats['class_distribution'].get(label_id, 0) + 1

        # Parallel image loading
        paths, labels = zip(*file_paths)
        loaded_images = self._parallel_load(paths, target_size, grayscale)
        
        # Filter out failed loads
        data = [img for img in loaded_images if img is not None]
        labels = [label for img, label in zip(loaded_images, labels) if img is not None]
        self.load_stats['total_samples'] = len(data)

        # Convert to numpy arrays
        channels = 1 if grayscale else 3
        data = np.array(data).reshape(-1, *target_size, channels)
        labels = np.array(labels)

        # Balance dataset if requested
        if balance and len(np.unique(labels)) > 1:
            data, labels = self._balance_dataset(data, labels)

        # Data augmentation
        if augment:
            augmented_data = []
            augmented_labels = []
            for img, label in zip(data, labels):
                augmented = self.augmentation(image=img.squeeze())['image']
                augmented_data.append(augmented)
                augmented_labels.append(label)
            
            # Combine original and augmented data
            data = np.concatenate([data, np.array(augmented_data).reshape(-1, *target_size, channels)])
            labels = np.concatenate([labels, np.array(augmented_labels)])

        logger.info(f"Successfully loaded {len(data)} samples (Errors: {self.load_stats['load_errors']})")
        logger.info(f"Class distribution: {self.load_stats['class_distribution']}")

        return train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    def load_video_frames(self, video_path: str, target_size: Tuple[int, int], 
                         frames_per_video: int = 30, grayscale: bool = True) -> np.ndarray:
        """Load frames from video file"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, frames_per_video, dtype=int)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            if i in frame_indices:
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = self._smart_resize(frame, target_size)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

        cap.release()
        channels = 1 if grayscale else 3
        return np.array(frames).reshape(-1, *target_size, channels)

    def plot_sample_images(self, X: np.ndarray, y: np.ndarray, class_names: Dict[int, str], n_samples: int = 5):
        """Plot sample images from dataset"""
        plt.figure(figsize=(15, 8))
        unique_classes = np.unique(y)
        
        for i, cls in enumerate(unique_classes):
            cls_indices = np.where(y == cls)[0]
            sample_indices = np.random.choice(cls_indices, n_samples, replace=False)
            
            for j, idx in enumerate(sample_indices):
                plt_idx = i * n_samples + j + 1
                plt.subplot(len(unique_classes), n_samples, plt_idx)
                img = X[idx].squeeze()
                if len(img.shape) == 2:
                    plt.imshow(img, cmap='gray')
                else:
                    plt.imshow(img)
                plt.title(f"{class_names[cls]}")
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == '__main__':
    loader = DatasetLoader()

    # Test MRL Eye Dataset
    print("\nLoading MRL Eye Dataset:")
    X_train, X_test, y_train, y_test = loader.load_mrl_eye_dataset(
        "../../datasets/MRL_Dataset", 
        target_size=(48, 48),
        augment=True,
        balance=True
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Plot samples
    loader.plot_sample_images(X_train, y_train, {0: "Closed", 1: "Open"})

    # Test generic dataset
    print("\nLoading DDD Dataset:")
    label_map = {"awake": 0, "drowsy": 1, "extreme": 2}
    X_train, X_test, y_train, y_test = loader.load_generic_image_dataset(
        "../../datasets/DDD",
        label_map,
        target_size=(128, 128),
        grayscale=False,
        augment=True
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    loader.plot_sample_images(X_train, y_train, {v: k for k, v in label_map.items()})