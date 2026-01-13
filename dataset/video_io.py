import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os


def load_mov_video(video_path):
    """Load a .mov video file and return frames as numpy array."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1
    
    cap.release()
    
    if frames:
        video_array = np.array(frames)
        print(f"Loaded video: {os.path.basename(video_path)} - {frame_count} frames, shape: {video_array.shape}")
        return video_array
    else:
        raise ValueError(f"No frames found in video: {video_path}")


def load_video_labels(labels_path="dataset_mini/labels.csv"):
    """Load the labels CSV file from dataset_mini folder."""
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    df = pd.read_csv(labels_path)
    print(f"Loaded labels: {len(df)} annotations, {df['videoName'].nunique()} videos")
    
    return df


def get_mini_dataset_info(dataset_dir="dataset_mini"):
    """Get information about the mini dataset."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Find video files
    video_files = list(dataset_path.glob("*.mov"))
    
    # Find labels file
    labels_file = dataset_path / "labels.csv"
    
    info = {
        "video_files": [str(f) for f in video_files],
        "video_count": len(video_files),
        "labels_file": str(labels_file) if labels_file.exists() else None,
        "dataset_dir": str(dataset_path)
    }
    
    print(f"Mini dataset info: {info['video_count']} videos, labels: {'✓' if info['labels_file'] else '✗'}")
    
    return info


class DrivingVideoMiniDataset:
    """Dataset class for working with the mini driving video dataset with train/val/test splits."""
    
    def __init__(self, dataset_dir="dataset_mini", split="train", train_ratio=0.7, val_ratio=0.2, random_seed=42):
        """
        Initialize dataset with train/val/test splits.
        
        Args:
            dataset_dir: Path to dataset_mini folder
            split: "train", "val", or "test"
            train_ratio: Proportion of videos for training (default: 0.7)
            val_ratio: Proportion of videos for validation (default: 0.2)
            random_seed: Random seed for reproducible splits
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Load labels
        labels_path = self.dataset_dir / "labels.csv"
        if labels_path.exists():
            self.labels = pd.read_csv(labels_path)
            self.video_names = self.labels['videoName'].unique()
        else:
            self.labels = None
            self.video_names = []
        
        # Find video files
        all_video_files = list(self.dataset_dir.glob("*.mov"))
        
        # Create train/val/test splits
        np.random.seed(random_seed)
        video_indices = np.random.permutation(len(all_video_files))
        
        n_train = int(len(all_video_files) * train_ratio)
        n_val = int(len(all_video_files) * val_ratio)
        
        if split == "train":
            selected_indices = video_indices[:n_train]
        elif split == "val":
            selected_indices = video_indices[n_train:n_train+n_val]
        elif split == "test":
            selected_indices = video_indices[n_train+n_val:]
        else:
            raise ValueError(f"Invalid split: {split}. Choose from 'train', 'val', 'test'")
        
        self.video_files = [all_video_files[i] for i in selected_indices]
        
        print(f"Initialized DrivingVideoMiniDataset ({split} split):")
        print(f"  - Directory: {self.dataset_dir}")
        print(f"  - Total videos: {len(all_video_files)}")
        print(f"  - {split} videos: {len(self.video_files)}")
        print(f"  - Labels: {len(self.labels) if self.labels is not None else 0} annotations")
    
    def get_video_path(self, video_name):
        """Get the file path for a specific video."""
        video_file = self.dataset_dir / f"{video_name}.mov"
        if video_file.exists():
            return str(video_file)
        else:
            raise FileNotFoundError(f"Video file not found: {video_file}")
    
    def load_video_batch(self, video_name):
        """Load a video as a batch of frames (N_frames, H, W, C)."""
        video_path = self.get_video_path(video_name)
        video_frames = load_mov_video(video_path)
        return video_frames  # Shape: (N_frames, H, W, 3)
    
    def get_video_labels(self, video_name):
        """Get labels for a specific video."""
        if self.labels is None:
            raise ValueError("No labels loaded")
        
        video_labels = self.labels[self.labels['videoName'] == video_name].copy()
        return video_labels
    
    def get_frame_labels(self, video_name, frame_idx):
        """Get labels for a specific frame within a video."""
        video_labels = self.get_video_labels(video_name)
        frame_labels = video_labels[video_labels['frameIndex'] == frame_idx].copy()
        return frame_labels
    
    def __len__(self):
        """Return number of videos in this split."""
        return len(self.video_files)
    
    def __getitem__(self, idx):
        """Get video batch and labels by index."""
        video_file = self.video_files[idx]
        video_name = video_file.stem  # filename without extension
        
        # Load video as batch of frames
        video_batch = self.load_video_batch(video_name)
        
        # Get all labels for this video
        if self.labels is not None:
            video_labels = self.get_video_labels(video_name)
        else:
            video_labels = None
        
        return {
            "video_name": video_name,
            "video_batch": video_batch,  # Shape: (N_frames, H, W, 3)
            "labels": video_labels,
            "video_path": str(video_file),
            "batch_size": len(video_batch),  # Number of frames in this batch
            "split": self.split
        }
    
    @staticmethod
    def create_dataloaders(dataset_dir="dataset_mini", batch_size=1, train_ratio=0.7, val_ratio=0.2, random_seed=42):
        """
        Create train, val, test datasets and dataloaders.
        
        Args:
            dataset_dir: Path to dataset_mini folder
            batch_size: Number of videos per dataloader batch (usually 1 since each video is already a batch)
            train_ratio, val_ratio: Split ratios
            random_seed: Random seed for reproducible splits
            
        Returns:
            dict with train, val, test datasets
        """
        datasets = {
            "train": DrivingVideoMiniDataset(dataset_dir, "train", train_ratio, val_ratio, random_seed),
            "val": DrivingVideoMiniDataset(dataset_dir, "val", train_ratio, val_ratio, random_seed),
            "test": DrivingVideoMiniDataset(dataset_dir, "test", train_ratio, val_ratio, random_seed)
        }
        
        print(f"\nDataset splits created:")
        for split, dataset in datasets.items():
            print(f"  - {split}: {len(dataset)} videos")
        
        return datasets 


