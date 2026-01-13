import os
import shutil
from pathlib import Path
import pandas as pd


def get_dataset_path():
    """Get the path to the local driving video dataset."""
    dataset_path = "dataset/driving-video-with-object-tracking"
    if os.path.exists(dataset_path):
        print(f"Using local dataset at: {dataset_path}")
        return dataset_path
    else:
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")


def clean_output_dir(output_dir="dataset_mini"):
    """Clean the output directory before extraction."""
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"Cleaned existing directory: {output_path}")
    output_path.mkdir(exist_ok=True)
    print(f"Created clean directory: {output_path}")
    return str(output_path)


def extract_video_labels(n_videos=5, output_dir="dataset_mini"):
    """Extract labels for exactly n video sequences, replacing missing ones if needed."""
    csv_path = "dataset/driving-video-with-object-tracking/versions/1/mot_labels.csv"
    
    if not os.path.exists(csv_path):
        print(f"Labels file not found at: {csv_path}")
        return None
    
    # Load the full dataset
    df = pd.read_csv(csv_path, low_memory=False)
    all_videos = df['videoName'].unique()
    
    # Get first n unique video sequences
    selected_videos = all_videos[:n_videos]
    
    # Filter dataset to only include selected videos
    mini_df = df[df['videoName'].isin(selected_videos)].copy()
    
    # Save to file in the output directory
    output_file = os.path.join(output_dir, "labels.csv")
    mini_df.to_csv(output_file, index=False)
    
    print(f"=== MINI LABELS EXTRACTED ===")
    print(f"Selected videos: {list(selected_videos)}")
    print(f"Total annotations: {len(mini_df):,}")
    print(f"Unique images: {mini_df['name'].nunique():,}")
    print(f"Saved to: {output_file}")
    
    return mini_df, all_videos  # Return both mini_df and all available videos


def extract_mini_batch(dataset_path, mini_labels, all_videos, n_videos, output_dir="dataset_mini"):
    """Extract exactly n video files (.mov), replacing missing ones with alternatives."""
    output_path = Path(output_dir)
    # Directory is already created and cleaned by clean_output_dir()
    
    if mini_labels is None:
        print("No mini_labels provided")
        return str(output_path)
    
    # Get unique video names from labels
    target_videos = mini_labels['videoName'].unique()
    
    print(f"Looking for {len(target_videos)} video files (.mov)...")
    
    # Find all available video files first
    all_available_videos = []
    for root, dirs, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith('.mov'):
                video_name = filename.replace('.mov', '')
                all_available_videos.append((video_name, os.path.join(root, filename)))
    
    print(f"Found {len(all_available_videos)} total .mov files in dataset")
    
    # Find and copy target video files
    copied_files = []
    target_found = []
    
    for video_name, source_path in all_available_videos:
        if video_name in target_videos:
            filename = os.path.basename(source_path)
            dest_path = output_path / filename
            shutil.copy2(source_path, dest_path)
            copied_files.append(filename)
            target_found.append(video_name)
            print(f"Copied: {filename}")
    
    # If we have missing videos, find replacements
    missing_count = n_videos - len(target_found)
    if missing_count > 0:
        print(f"Missing {missing_count} videos, finding replacements...")
        
        # Find available videos that aren't already copied
        available_for_replacement = [(vn, sp) for vn, sp in all_available_videos 
                                   if vn not in target_found]
        
        replacements_needed = min(missing_count, len(available_for_replacement))
        for i in range(replacements_needed):
            video_name, source_path = available_for_replacement[i]
            filename = os.path.basename(source_path)
            dest_path = output_path / filename
            shutil.copy2(source_path, dest_path)
            copied_files.append(filename)
            print(f"Replacement copied: {filename}")
    
    print(f"=== EXTRACTION COMPLETE ===")
    print(f"Copied {len(copied_files)} video files (target: {n_videos})")
    print(f"Mini batch saved to: {output_path}")
    return str(output_path)


def generate_mini_dataset(n_videos=10):
    """Main function to generate mini dataset for pipeline setup."""
    dataset_path = get_dataset_path()
    
    # Clean output directory first
    output_dir = clean_output_dir()
    
    # Extract mini labels first
    result = extract_video_labels(n_videos, output_dir)
    if result is None:
        return None, None
    
    mini_labels, all_videos = result
    
    # Extract corresponding video files based on labels
    mini_path = extract_mini_batch(dataset_path, mini_labels, all_videos, n_videos, output_dir)
    
    return mini_labels, mini_path


if __name__ == "__main__":
    generate_mini_dataset()
