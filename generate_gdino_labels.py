"""
Grounding DINO Label Generator for YOLO Dataset

This script generates YOLO format labels for logo detection using Grounding DINO model.
It processes images from a YOLO dataset and creates filtered bounding box labels.

Features:
- Batch image preloading with FIFO queue
- YOLO format output (class_id x_center y_center width height, normalized)
- Multiple filtering strategies:
  1. Size filtering: remove boxes that are too large or too small
  2. Aspect ratio filtering: remove overly elongated boxes
  3. Nested box filtering: keep smaller box when one is contained in another
  4. Skip images with zero detections

Usage:
    python generate_gdino_labels.py --config_file CONFIG --checkpoint_path CHECKPOINT \
                                     --dataset_path DATASET --text_prompt "logo" \
                                     --box_threshold 0.3 --test_mode --test_samples 50
"""

import argparse
import os
import random
from pathlib import Path
from queue import Queue
from threading import Thread
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


class ImageLoader:
    """Preload images in background thread with FIFO queue"""
    
    def __init__(self, image_paths, queue_size=10):
        self.image_paths = image_paths
        self.queue = Queue(maxsize=queue_size)
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def _load_worker(self):
        """Worker thread for loading images"""
        for img_path in self.image_paths:
            try:
                image_pil = Image.open(img_path).convert("RGB")
                image_tensor, _ = self.transform(image_pil, None)
                self.queue.put((img_path, image_pil, image_tensor))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                self.queue.put((img_path, None, None))
        self.queue.put((None, None, None))  # Sentinel value
    
    def start(self):
        """Start background loading thread"""
        self.thread = Thread(target=self._load_worker, daemon=True)
        self.thread.start()
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.queue.get()
        if item[0] is None:  # Sentinel value
            raise StopIteration
        return item


def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    """Load Grounding DINO model"""
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cuda"):
    """Run Grounding DINO inference"""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)
    
    # Filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]
    
    return boxes_filt


def filter_by_size(boxes, min_size=0.01, max_size=0.95):
    """Filter boxes by area (normalized by image area)"""
    if len(boxes) == 0:
        return boxes, []
    
    areas = boxes[:, 2] * boxes[:, 3]  # width * height
    mask = (areas >= min_size) & (areas <= max_size)
    return boxes[mask], mask.tolist()


def filter_by_aspect_ratio(boxes, max_ratio=5.0):
    """Filter boxes with extreme aspect ratios"""
    if len(boxes) == 0:
        return boxes, []
    
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    ratios = torch.maximum(widths / (heights + 1e-6), heights / (widths + 1e-6))
    mask = ratios <= max_ratio
    return boxes[mask], mask.tolist()


def filter_nested_boxes(boxes):
    """Remove larger box when one box contains another, keep smaller one"""
    if len(boxes) <= 1:
        return boxes, list(range(len(boxes)))
    
    # Convert from center format to corner format
    boxes_xyxy = boxes.clone()
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    areas = boxes[:, 2] * boxes[:, 3]
    keep_mask = torch.ones(len(boxes), dtype=torch.bool)
    
    # Sort by area (smallest first)
    sorted_indices = torch.argsort(areas)
    
    for i in range(len(sorted_indices)):
        if not keep_mask[sorted_indices[i]]:
            continue
            
        box_i = boxes_xyxy[sorted_indices[i]]
        
        for j in range(i + 1, len(sorted_indices)):
            if not keep_mask[sorted_indices[j]]:
                continue
                
            box_j = boxes_xyxy[sorted_indices[j]]
            
            # Check if box_i is inside box_j
            if (box_i[0] >= box_j[0] and box_i[1] >= box_j[1] and
                box_i[2] <= box_j[2] and box_i[3] <= box_j[3]):
                keep_mask[sorted_indices[j]] = False  # Remove larger box
    
    return boxes[keep_mask], keep_mask.tolist()


def boxes_to_yolo_format(boxes, class_id=0):
    """Convert boxes to YOLO format: class_id x_center y_center width height"""
    if len(boxes) == 0:
        return []
    
    yolo_labels = []
    for box in boxes:
        x_center, y_center, width, height = box.tolist()
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_labels


def visualize_boxes(image_pil, boxes, save_path):
    """Draw boxes on image and save"""
    if len(boxes) == 0:
        return
    
    W, H = image_pil.size
    draw = ImageDraw.Draw(image_pil)
    
    for idx, box in enumerate(boxes):
        # Convert from normalized center format to pixel corner format
        x_center, y_center, width, height = box.tolist()
        x_center *= W
        y_center *= H
        width *= W
        height *= H
        
        x0 = int(x_center - width / 2)
        y0 = int(y_center - height / 2)
        x1 = int(x_center + width / 2)
        y1 = int(y_center + height / 2)
        
        # Random color for each box
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        
        # Draw rectangle
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        
        # Draw label
        label = f"logo_{idx}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Get text bounding box (use textbbox for newer PIL versions)
        try:
            bbox = draw.textbbox((x0, y0), label, font=font)
        except:
            # Simple fallback
            bbox = (x0, y0, x0 + 50, y0 + 15)
        
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), label, fill="white", font=font)
    
    image_pil.save(save_path)


def process_dataset(model, dataset_path, output_path, text_prompt, box_threshold, 
                   text_threshold, min_size, max_size, max_aspect_ratio, 
                   device="cuda", test_mode=False, test_samples=50):
    """Process entire dataset and generate YOLO labels"""
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Process train and val splits
    for split in ['train', 'val']:
        print(f"\nProcessing {split} split...")
        
        image_dir = dataset_path / 'images' / split
        output_label_dir = output_path / 'labels' / split
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization directory in test mode
        vis_dir = None
        if test_mode:
            vis_dir = output_path / 'visualizations' / split
            vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image paths
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        if test_mode:
            image_paths = random.sample(image_paths, min(test_samples, len(image_paths)))
            print(f"Test mode: processing {len(image_paths)} random images")
        
        print(f"Found {len(image_paths)} images in {split}")
        
        # Statistics
        stats = {
            'total': len(image_paths),
            'processed': 0,
            'skipped_zero_detections': 0,
            'filtered_by_size': 0,
            'filtered_by_aspect': 0,
            'filtered_by_nesting': 0,
            'total_boxes_generated': 0
        }
        
        # Start image loader
        loader = ImageLoader(image_paths, queue_size=10).start()
        
        # Process images
        for img_path, image_pil, image_tensor in tqdm(loader, total=len(image_paths)):
            if image_pil is None:
                continue
            
            # Run inference
            boxes = get_grounding_output(
                model, image_tensor, text_prompt, 
                box_threshold, text_threshold, device
            )
            
            original_count = len(boxes)
            
            # Apply filters
            boxes, size_mask = filter_by_size(boxes, min_size, max_size)
            stats['filtered_by_size'] += original_count - len(boxes)
            
            boxes, aspect_mask = filter_by_aspect_ratio(boxes, max_aspect_ratio)
            stats['filtered_by_aspect'] += len(size_mask) - len(boxes) if size_mask else 0
            
            boxes, nest_mask = filter_nested_boxes(boxes)
            stats['filtered_by_nesting'] += len(aspect_mask) - len(boxes) if aspect_mask else 0
            
            # Skip if no detections
            if len(boxes) == 0:
                stats['skipped_zero_detections'] += 1
                continue
            
            # Convert to YOLO format
            yolo_labels = boxes_to_yolo_format(boxes, class_id=0)
            
            # Save labels
            label_path = output_label_dir / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
            
            # Save visualization in test mode
            if test_mode and vis_dir is not None:
                vis_path = vis_dir / f"{img_path.stem}_vis.jpg"
                visualize_boxes(image_pil.copy(), boxes, vis_path)
            
            stats['processed'] += 1
            stats['total_boxes_generated'] += len(boxes)
        
        # Print statistics
        print(f"\n{split.upper()} Statistics:")
        print(f"  Total images: {stats['total']}")
        print(f"  Processed: {stats['processed']}")
        print(f"  Skipped (zero detections): {stats['skipped_zero_detections']}")
        print(f"  Filtered by size: {stats['filtered_by_size']}")
        print(f"  Filtered by aspect ratio: {stats['filtered_by_aspect']}")
        print(f"  Filtered by nesting: {stats['filtered_by_nesting']}")
        print(f"  Total boxes generated: {stats['total_boxes_generated']}")
        print(f"  Avg boxes per image: {stats['total_boxes_generated'] / max(stats['processed'], 1):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO labels using Grounding DINO")
    
    # Model arguments
    parser.add_argument("--config_file", "-c", type=str, required=True,
                       help="Path to model config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda/cpu)")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", "-d", type=str, required=True,
                       help="Path to YOLO dataset root (containing images/ and labels/)")
    parser.add_argument("--output_path", "-o", type=str, default=None,
                       help="Output path for generated labels (default: dataset_path/labels_gdino)")
    
    # Detection arguments
    parser.add_argument("--text_prompt", "-t", type=str, default="logo",
                       help="Text prompt for detection")
    parser.add_argument("--box_threshold", type=float, default=0.3,
                       help="Box confidence threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                       help="Text confidence threshold")
    
    # Filtering arguments
    parser.add_argument("--min_size", type=float, default=0.001,
                       help="Minimum box size (as fraction of image area)")
    parser.add_argument("--max_size", type=float, default=0.9,
                       help="Maximum box size (as fraction of image area)")
    parser.add_argument("--max_aspect_ratio", type=float, default=5.0,
                       help="Maximum aspect ratio (width/height or height/width)")
    
    # Test mode
    parser.add_argument("--test_mode", action="store_true",
                       help="Test mode: process only random subset of images")
    parser.add_argument("--test_samples", type=int, default=50,
                       help="Number of samples to test in test mode")
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        args.output_path = os.path.join(args.dataset_path, 'labels_gdino')
    
    print("Configuration:")
    print(f"  Model config: {args.config_file}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Output: {args.output_path}")
    print(f"  Text prompt: {args.text_prompt}")
    print(f"  Box threshold: {args.box_threshold}")
    print(f"  Size range: {args.min_size} - {args.max_size}")
    print(f"  Max aspect ratio: {args.max_aspect_ratio}")
    print(f"  Device: {args.device}")
    if args.test_mode:
        print(f"  Test mode: {args.test_samples} samples")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.config_file, args.checkpoint_path, args.device)
    print("Model loaded successfully")
    
    # Process dataset
    process_dataset(
        model=model,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        min_size=args.min_size,
        max_size=args.max_size,
        max_aspect_ratio=args.max_aspect_ratio,
        device=args.device,
        test_mode=args.test_mode,
        test_samples=args.test_samples
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
