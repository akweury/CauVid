"""
Generate depth maps for COCO images using Depth Anything V3
Processes images from storage/coco/selected/val2017 and saves depth maps
"""

import os
import sys
from pathlib import Path
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Import Depth Anything 3
try:
    from depth_anything_3.api import DepthAnything3
except ImportError as e:
    print("❌ Error: Failed to import Depth Anything 3")
    print(f"Import error: {e}")
    print("\nPlease ensure you're using the Python 3.10 environment (.venv310)")
    print("Run: .venv310\\Scripts\\Activate.ps1")
    sys.exit(1)


def generate_depth_maps(
    input_dir,
    output_dir,
    model_name="depth-anything/DA3-Large",
    max_images=None,
    batch_size=8,
    device="cuda",
    use_fp16=False,
    image_extensions=("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
):
    """
    Generate depth maps for images in a directory using Depth Anything V3
    
    Args:
        input_dir: Path to input images
        output_dir: Path to save depth maps
        model_name: Model to use (DA3-Large, DA3-Base, DA3-Small, DA3-Giant)
        max_images: Maximum number of images to process (None for all)
        batch_size: Number of images to process in parallel (higher = more GPU usage)
        device: Device to use (cuda, cpu)
        use_fp16: Use mixed precision (faster on modern GPUs)
        image_extensions: Tuple of image file extensions to process
    """
    
    # Setup paths
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Depth Anything V3 - Depth Map Generation (GPU Optimized)")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output directory (absolute): {output_dir.absolute()}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Mixed precision (FP16): {use_fp16}")
    
    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        device = "cpu"
        use_fp16 = False
    
    device = torch.device(device)
    
    # Print GPU info
    print("\n" + "="*60)
    if device.type == "cuda":
        print("🚀 GPU ACCELERATION ENABLED")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Available GPU Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
    else:
        print("⚠️  RUNNING ON CPU (This will be slow!)")
    print("="*60)
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(str(input_dir / ext)))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images to process")
    print(f"Processing in {(len(image_files) + batch_size - 1) // batch_size} batches")
    
    # Load model
    print("\n" + "="*60)
    print("Loading Depth Anything V3 model...")
    print("="*60)
    
    try:
        model = DepthAnything3.from_pretrained(model_name)
        model = model.to(device=device)
        model.eval()  # Set to evaluation mode
        
        # Enable mixed precision if requested
        if use_fp16 and device.type == "cuda":
            model = model.half()
            print(f"✓ Model loaded successfully (FP16 mode)")
            print(f"GPU Memory after model load: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        else:
            print(f"✓ Model loaded successfully")
            if device.type == "cuda":
                print(f"GPU Memory after model load: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nAvailable models:")
        print("  - depth-anything/DA3-Giant")
        print("  - depth-anything/DA3-Large")
        print("  - depth-anything/DA3-Base")
        print("  - depth-anything/DA3-Small")
        return
    
    # Process images in batches
    print("\n" + "="*60)
    print("Processing images in batches...")
    print("="*60)
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Process in batches
        for batch_start in tqdm(range(0, len(image_files), batch_size), desc="Batch progress"):
            batch_end = min(batch_start + batch_size, len(image_files))
            batch_paths = image_files[batch_start:batch_end]
            
            try:
                # Print GPU memory before batch processing
                if device.type == "cuda" and batch_start % (batch_size * 5) == 0:
                    print(f"\n[Batch {batch_start//batch_size + 1}] GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB (current/peak)")
                
                # Run inference on batch
                prediction = model.inference(batch_paths)
                
                # Save each image in the batch
                for i, img_path in enumerate(batch_paths):
                    img_name = Path(img_path).stem
                    
                    # Get depth map for this image
                    depth = prediction.depth[i]
                    
                    # Save depth map as numpy array
                    depth_npz_path = output_dir / f"{img_name}_depth.npz"
                    np.savez_compressed(depth_npz_path, depth=depth)
                    print(f"  ✓ Saved: {depth_npz_path.name}")
                    
                    # Save depth map as normalized image for visualization
                    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    depth_normalized = (depth_normalized * 255).astype(np.uint8)
                    depth_img = Image.fromarray(depth_normalized)
                    depth_img_path = output_dir / f"{img_name}_depth.png"
                    depth_img.save(depth_img_path)
                    print(f"  ✓ Saved: {depth_img_path.name}")
                    
                    # Optionally save confidence map if available
                    if prediction.conf is not None:
                        conf = prediction.conf[i]
                        conf_normalized = (conf * 255).astype(np.uint8)
                        conf_img = Image.fromarray(conf_normalized)
                        conf_img_path = output_dir / f"{img_name}_conf.png"
                        conf_img.save(conf_img_path)
                        print(f"  ✓ Saved: {conf_img_path.name}")
                
                # Clear GPU cache after each batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\n❌ Error processing batch {batch_start}-{batch_end}: {e}")
                continue
    
    print("\n" + "="*60)
    print("✅ Depth map generation complete!")
    print("="*60)
    print(f"Processed {len(image_files)} images")
    print(f"Depth maps saved to: {output_dir.absolute()}")
    
    # Count saved files
    saved_files = list(output_dir.glob('*'))
    print(f"\nTotal files in output directory: {len(saved_files)}")
    print(f"  - .npz files: {len(list(output_dir.glob('*.npz')))} (raw depth values)")
    print(f"  - .png files: {len(list(output_dir.glob('*.png')))} (visualizations)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth maps for COCO images using Depth Anything V3"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory containing images (default: storage/coco/selected/val2017)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for depth maps (default: storage/coco/depth_maps)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/DA3-Large",
        choices=[
            "depth-anything/DA3-Giant",
            "depth-anything/DA3-Large",
            "depth-anything/DA3-Base",
            "depth-anything/DA3-Small",
            "depth-anything/DA3Metric-Large",
            "depth-anything/DA3Mono-Large",
        ],
        help="Depth Anything V3 model to use"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images to process in parallel (default: 8c, increase for more GPU usage)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Enable mixed precision (FP16) - faster but may have compatibility issues"
    )
    
    args = parser.parse_args()
    
    # Set default paths using config
    if args.input_dir is None:
        coco_path = config.get_coco_path(remote=False)
        args.input_dir = coco_path / "selected" / "val2017"
    
    if args.output_dir is None:
        args.output_dir = config.storage / "coco" / "depth_maps"
    
    # Check if input directory exists
    if not Path(args.input_dir).exists():
        print(f"❌ Input directory does not exist: {args.input_dir}")
        print("\nPlease ensure the COCO dataset is downloaded and images are in:")
        print(f"  {args.input_dir}")
        sys.exit(1)
    
    # Generate depth maps
    generate_depth_maps(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        max_images=args.max_images,
        batch_size=args.batch_size,
        device=args.device,
        use_fp16=args.use_fp16
    )


if __name__ == "__main__":
    main()
