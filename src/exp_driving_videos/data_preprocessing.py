
"""
Data preprocessing utilities for driving videos
Includes functions for converting videos to frames and other preprocessing tasks
"""

import cv2
import os
import subprocess
import json
import sys
from pathlib import Path
import config
from typing import Union, Optional, List
import logging

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_video_rotation(video_path):
    """
    Get the rotation metadata from a video file using ffprobe.
    Returns rotation angle in degrees (0, 90, 180, 270) or 0 if not found.
    
    Falls back to checking frame dimensions if ffprobe is not available.
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream_tags=rotate',
            '-of', 'json',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'streams' in data and len(data['streams']) > 0:
                tags = data['streams'][0].get('tags', {})
                rotation = int(tags.get('rotate', 0))
                if rotation > 0:
                    return rotation
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
        logger.debug("ffprobe not available or failed, checking frame dimensions")
    
    # Fallback: check if video dimensions suggest it needs rotation
    # If width < height, it's likely a portrait/rotated video that needs correction
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # If portrait orientation (width < height), assume 90 degree rotation
            if width < height:
                logger.info(f"  Video dimensions {width}x{height} suggest 90° rotation needed")
                return 90
    except Exception:
        pass
    
    return 0

def rotate_frame(frame, rotation):
    """
    Rotate frame based on rotation angle.
    
    Args:
        frame: Input frame (numpy array)
        rotation: Rotation angle in degrees (90, 180, 270)
    
    Returns:
        Rotated frame
    """
    if rotation == 90:
        # Rotate 90 degrees counter-clockwise
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        # Rotate 90 degrees clockwise (or 270 counter-clockwise)
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def extract_frames(video_path, output_dir):
    """
    Extract frames from a video file with automatic rotation correction.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory where frames will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get rotation metadata
    rotation = get_video_rotation(video_path)
    if rotation > 0:
        logger.info(f"  Detected rotation: {rotation} degrees - will correct during extraction")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Apply rotation correction if needed
        if rotation > 0:
            frame = rotate_frame(frame, rotation)
        
        # Save frame with zero-indexed numbering
        frame_filename = f"{output_dir}/frame_{frame_count:05d}.jpg"
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    logger.info(f"  Extracted {frame_count} frames to {output_dir}")



def convert_videos_to_frames(
    video_folder: Union[str, Path], 
    output_folder: Union[str, Path],
    video_extensions: List[str] = ['.mov', '.mp4', '.avi'],
    frame_format: str = '.jpg',
    target_fps: float = 5.0,
    max_frames_per_video: Optional[int] = None
) -> bool:
    """
    Convert video files in a folder to individual frames with frame rate downsampling.
    
    Args:
        video_folder: Path to folder containing video files
        output_folder: Path to folder where frames will be saved
        video_extensions: List of video file extensions to process (default: ['.mov', '.mp4', '.avi'])
        frame_format: Output frame format (default: '.jpg')
        target_fps: Target frame rate for extraction (default: 5.0 fps). 
                   For example, a 30fps video will be downsampled to 5fps by keeping every 6th frame.
        max_frames_per_video: Maximum number of frames to extract per video (None for unlimited)
        
    Returns:
        bool: True if successful, False if any errors occurred
        
    Frame organization:
        output_folder/
        ├── video1_name/
        │   ├── frame_00000.jpg
        │   ├── frame_00001.jpg
        │   └── ...
        └── video2_name/
            ├── frame_00000.jpg
            └── ...
            
    Example:
        For a 40-second video at 30fps (1200 frames):
        - With target_fps=5.0, only 200 frames will be saved (40s * 5fps)
        - Every 6th frame will be kept (30/5 = 6)
    """
    
    video_folder = Path(video_folder)
    output_folder = Path(output_folder)
    
    # Validate input folder
    if not video_folder.exists():
        logger.error(f"Video folder does not exist: {video_folder}")
        return False
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output frames will be saved to: {output_folder}")
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_folder.glob(f"*{ext}"))
        video_files.extend(video_folder.glob(f"*{ext.upper()}"))  # Include uppercase
    
    if not video_files:
        logger.warning(f"No video files found in {video_folder} with extensions {video_extensions}")
        return False
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    success_count = 0
    
    for video_path in video_files:
        
        logger.info(f"Processing: {video_path.name}")
        # extract_frames(video_path, output_folder / video_path.stem)
        
        
        try:
            # Create output subfolder for this video
            video_name = video_path.stem  # filename without extension
            video_output_folder = output_folder / video_name
            video_output_folder.mkdir(exist_ok=True)
            
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                continue
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Get video rotation metadata
            rotation = get_video_rotation(video_path)
            if rotation > 0:
                logger.info(f"  Detected rotation: {rotation} degrees - will correct during extraction")
            
            # Calculate frame skip interval to achieve target fps
            # For example: 30fps video -> 5fps target = keep every 6th frame
            frame_skip = max(1, int(round(original_fps / target_fps)))
            
            logger.info(f"  Video info: {total_frames} frames, {original_fps:.2f} FPS")
            logger.info(f"  Downsampling: {original_fps:.2f} fps -> {target_fps:.2f} fps (keeping every {frame_skip} frame)")
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Only save frames at the target interval
                if frame_count % frame_skip == 0:
                    # Stop if max_frames_per_video limit reached
                    if max_frames_per_video is not None and extracted_count >= max_frames_per_video:
                        break
                    
                    # Apply rotation correction if needed
                    if rotation > 0:
                        frame = rotate_frame(frame, rotation)
                    
                    # Save frame (zero-indexed)
                    frame_filename = f"frame_{extracted_count:05d}{frame_format}"
                    frame_path = video_output_folder / frame_filename
                    
                    if cv2.imwrite(str(frame_path), frame):
                        extracted_count += 1
                    else:
                        logger.error(f"Failed to save frame: {frame_path}")
                
                frame_count += 1
            
            cap.release()
            
            actual_fps = extracted_count / (total_frames / original_fps) if total_frames > 0 else 0
            logger.info(f"  Extracted {extracted_count} frames from {video_path.name} (effective fps: {actual_fps:.2f})")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            continue
    
    logger.info(f"Processing complete: {success_count}/{len(video_files)} videos processed successfully")
    return success_count == len(video_files)

def convert_mov_to_frames(video_folder: Union[str, Path], frame_folder: Union[str, Path]) -> bool:
    """
    Simplified function to convert .mov videos to frames (backward compatibility)
    
    Args:
        video_folder: Path to folder containing .mov files
        frame_folder: Path to folder where frames will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    return convert_videos_to_frames(
        video_folder=video_folder,
        output_folder=frame_folder,
        video_extensions=['.mov']
    )

def get_video_info(video_path: Union[str, Path]) -> dict:
    """
    Get information about a video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Video information including frames, fps, duration, resolution
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        return {"error": f"Video file not found: {video_path}"}
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {"error": f"Could not open video: {video_path}"}
    
    info = {
        "filename": video_path.name,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": 0,
        "format": video_path.suffix
    }
    
    if info["fps"] > 0:
        info["duration_seconds"] = info["total_frames"] / info["fps"]
    
    cap.release()
    
    return info

def generate_frame_depth_maps(
    frame_folder: Union[str, Path],
    depth_output_folder: Union[str, Path],
    model_name: str = "depth-anything/DA3-Large",
    batch_size: int = 4,
    device: str = "auto",
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png'],
    max_frames: Optional[int] = None
) -> bool:
    """
    Generate depth maps for frames using Depth Anything V3
    
    Args:
        frame_folder: Path to folder containing frame images
        depth_output_folder: Path to folder where depth maps will be saved
        model_name: Depth Anything V3 model to use (DA3-Large, DA3-Base, DA3-Small, DA3-Giant)
        batch_size: Number of frames to process in parallel
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        image_extensions: List of image file extensions to process
        max_frames: Maximum number of frames to process (None for all)
        
    Returns:
        bool: True if successful, False if any errors occurred
        
    Output organization:
        depth_output_folder/
        ├── video1_name/
        │   ├── frame_00001_depth.npy
        │   ├── frame_00001_depth.png  (visualization)
        │   └── ...
        └── video2_name/
            └── ...
    """
    
    try:
        # Import the depth map generator
        sys.path.append(str(PROJECT_ROOT / "external" / "Depth-Anything-3"))
        from src.external_depth_anything.depth_map_generator import generate_depth_maps
        
        frame_folder = Path(frame_folder)
        depth_output_folder = Path(depth_output_folder)
        
        # Validate input folder
        if not frame_folder.exists():
            logger.error(f"Frame folder does not exist: {frame_folder}")
            return False
        
        # Create output folder
        depth_output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Depth maps will be saved to: {depth_output_folder}")
        
        # Auto-detect device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            logger.info(f"Auto-detected device: {device}")
        
        # Check if frame_folder contains subdirectories (organized by video) or just frames
        subdirs = [d for d in frame_folder.iterdir() if d.is_dir()]
        
        if subdirs:
            # Process each video subdirectory separately
            logger.info(f"Found {len(subdirs)} video subdirectories to process")
            success_count = 0
            
            for subdir in subdirs:
                video_name = subdir.name
                logger.info(f"Processing frames for video: {video_name}")
                
                # Create corresponding output subdirectory
                video_depth_folder = depth_output_folder / video_name
                video_depth_folder.mkdir(exist_ok=True)
                
                try:
                    # Generate depth maps for this video's frames
                    generate_depth_maps(
                        input_dir=subdir,
                        output_dir=video_depth_folder,
                        model_name=model_name,
                        max_images=max_frames,
                        batch_size=batch_size,
                        device=device,
                        use_fp16=(device == "cuda"),
                        image_extensions=tuple(f"*{ext}" for ext in image_extensions)
                    )
                    
                    logger.info(f"✅ Completed depth maps for {video_name}")
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {video_name}: {str(e)}")
                    continue
            
            logger.info(f"Depth map generation complete: {success_count}/{len(subdirs)} videos processed")
            return success_count == len(subdirs)
            
        else:
            # Process all frames in the folder directly
            logger.info("Processing all frames in the folder directly")
            
            try:
                generate_depth_maps(
                    input_dir=frame_folder,
                    output_dir=depth_output_folder,
                    model_name=model_name,
                    max_images=max_frames,
                    batch_size=batch_size,
                    device=device,
                    use_fp16=(device == "cuda"),
                    image_extensions=tuple(f"*{ext}" for ext in image_extensions)
                )
                
                logger.info("✅ Depth map generation completed successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to generate depth maps: {str(e)}")
                return False
    
    except ImportError as e:
        logger.error(f"❌ Failed to import depth map generator: {e}")
        logger.error("Make sure Depth Anything V3 is properly set up")
        return False
    
    except Exception as e:
        logger.error(f"❌ Unexpected error in depth map generation: {str(e)}")
        return False

def convert_frames_to_depth_maps(
    frame_folder: Union[str, Path],
    depth_folder: Union[str, Path],
    model_size: str = "large",
    device: str = "auto"
) -> bool:
    """
    Simple function to convert existing frames to depth maps using Depth Anything V3
    
    Args:
        frame_folder: Path to folder containing frame images
        depth_folder: Path to folder where depth maps will be saved
        model_size: Model size ('small', 'base', 'large', 'giant')
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        bool: True if successful, False if any errors occurred
        
    Usage:
        convert_frames_to_depth_maps("path/to/frames", "path/to/depth_maps")
    """
    
    # Model name mapping for convenience
    model_mapping = {
        "small": "depth-anything/DA3-Small",
        "base": "depth-anything/DA3-Base", 
        "large": "depth-anything/DA3-Large",
        "giant": "depth-anything/DA3-Giant"
    }
    
    model_name = model_mapping.get(model_size.lower(), "depth-anything/DA3-Large")
    
    logger.info(f"🔍 Converting frames to depth maps...")
    logger.info(f"📂 Source frames: {frame_folder}")
    logger.info(f"💾 Output depth maps: {depth_folder}")
    logger.info(f"🤖 Using model: {model_name}")
    
    # Use the existing comprehensive function
    return generate_frame_depth_maps(
        frame_folder=frame_folder,
        depth_output_folder=depth_folder,
        model_name=model_name,
        batch_size=4,  # Safe default for most GPUs
        device=device,
        image_extensions=['.jpg', '.jpeg', '.png', '.bmp'],
        max_frames=None  # Process all frames
    )

def process_video_to_depth_pipeline(
    video_folder: Union[str, Path],
    output_base_folder: Union[str, Path],
    model_name: str = "depth-anything/DA3-Large",
    target_fps: float = 5.0,
    max_frames_per_video: Optional[int] = None,
    batch_size: int = 4,
    device: str = "auto"
) -> bool:
    """
    Complete pipeline: Convert videos to frames then generate depth maps
    
    Args:
        video_folder: Path to folder containing video files
        output_base_folder: Base path where frames and depth maps will be saved
        model_name: Depth Anything V3 model to use
        target_fps: Target frame rate for extraction (default: 5.0 fps)
        max_frames_per_video: Maximum number of frames to extract per video
        batch_size: Batch size for depth map generation
        device: Device to use for depth estimation
        
    Returns:
        bool: True if successful, False if any errors occurred
        
    Output structure:
        output_base_folder/
        ├── frames/
        │   ├── video1/
        │   └── video2/
        └── depth_maps/
            ├── video1/
            └── video2/
    """
    
    output_base_folder = Path(output_base_folder)
    frames_folder = output_base_folder / "frames"
    depth_folder = output_base_folder / "depth_maps"
    
    logger.info("🎬 Starting complete video-to-depth pipeline...")
    logger.info(f"Input videos: {video_folder}")
    logger.info(f"Output frames: {frames_folder}")
    logger.info(f"Output depth maps: {depth_folder}")
    
    # Step 1: Convert videos to frames
    logger.info("📝 Step 1: Converting videos to frames...")
    if not convert_videos_to_frames(
        video_folder=video_folder,
        output_folder=frames_folder,
        target_fps=target_fps,
        max_frames_per_video=max_frames_per_video
    ):
        logger.error("Failed to convert videos to frames")
        return False
    
    # Step 2: Generate depth maps
    logger.info("🔍 Step 2: Generating depth maps...")
    if not generate_frame_depth_maps(
        frame_folder=frames_folder,
        depth_output_folder=depth_folder,
        model_name=model_name,
        batch_size=batch_size,
        device=device
    ):
        logger.error("Failed to generate depth maps")
        return False
    
    logger.info("🎉 Complete pipeline finished successfully!")
    return True

# Example usage
if __name__ == "__main__":
    # Example: Convert videos using config paths
    try:
        # You can modify these paths as needed
        data_root_folder = config.get_dataset_path("driving_mini")  # or specify custom path
        video_folder = data_root_folder / "videos"  # Assuming videos are in this subfolder
        output_folder = data_root_folder / "frames"
        depth_folder = data_root_folder / "depth_maps"
        
        print("🎬 CauVid Data Preprocessing Pipeline")
        print("=" * 50)
        print(f"Video source: {video_folder}")
        print(f"Frame output: {output_folder}")
        print(f"Depth output: {depth_folder}")
        print()
        
        # Choose what to run:
        run_frame_extraction = False  # Set to True if you need to extract frames from videos
        run_depth_estimation = True   # Convert existing frames to depth maps
        run_complete_pipeline = False # Set to True to run both steps together
        
        if run_complete_pipeline:
            # Option 1: Run complete pipeline (videos -> frames -> depth maps)
            print("🚀 Running complete video-to-depth pipeline...")
            success = process_video_to_depth_pipeline(
                video_folder=video_folder,
                output_base_folder=data_root_folder / "processed",
                model_name="depth-anything/DA3-Large",  # or DA3-Base, DA3-Small
                target_fps=5.0,  # Extract at 5 fps (e.g., 30fps video -> every 6th frame)
                max_frames_per_video=None,  # No limit, extract all at target fps
                batch_size=4,  # Adjust based on your GPU memory
                device="auto"  # auto-detect best device
            )
        else:
            # Option 2: Run steps separately
            if run_frame_extraction:
                print("📽️ Converting videos to frames...")
                success = convert_mov_to_frames(video_folder, output_folder)
                if success:
                    print("✅ Frame extraction completed!")
                else:
                    print("⚠️ Some videos failed to convert. Check logs for details.")
            
            if run_depth_estimation:
                print("🔍 Converting frames to depth maps...")
                
                # Simple way - just convert frames to depth maps
                depth_success = convert_frames_to_depth_maps(
                    frame_folder=output_folder,   # Where your frames are stored
                    depth_folder=depth_folder,    # Where to save depth maps
                    model_size="large",           # "small", "base", "large", or "giant"
                    device="auto"                 # auto-detect best device
                )
                
                if depth_success:
                    print("✅ Depth map generation completed!")
                    print(f"   📂 Depth maps saved to: {depth_folder}")
                else:
                    print("⚠️ Depth map generation failed. Check logs for details.")
        
        print("\n🎉 Data preprocessing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure to set up proper video and output folders in your config.")
        print("Also ensure Depth Anything V3 is properly installed and configured.")

