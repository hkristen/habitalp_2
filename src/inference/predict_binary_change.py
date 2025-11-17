#!/usr/bin/env python3

from pathlib import Path
import torch
import re
from ..trainers.utils import predict_large_image
from ..trainers.change import BinaryChangeSemanticSegmentationTaskBinaryLoss

def main():
    print("Starting binary change prediction...")
    
    # Configuration
    experiment_dir = Path('/home/hkristen/Nextcloud/HabitAlp2.0/Originaldaten/model_output/direct_change/')
    
    # Only the binary checkpoint
    checkpoint_path = experiment_dir / Path('checkpoints/binary_change-sweep-OneCycleLR_mit_b2_p256/binary_change-sweep-OneCycleLR_mit_b2_p256-epoch=05-step=4170-val_jaccard=0.5140.ckpt')
    
    # Define input paths
    image1_path = '/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2013/flug_2003_rgb_clipped.tif'
    
    # Two different image2 paths for different year ranges
    image2_configs = [
       # {
       #     'path': '/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2013/flug_2013_rgb_clipped.tif',
       #     'year': '2003_2013'
       # },
        {
            'path': '/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2024/flug_2019_2021_rgb_clipped.tif',
            'year': '2013_2020'
        }
    ]
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image 1: {image1_path}")
    
    # Load the model from checkpoint once
    print("Loading model...")
    task = BinaryChangeSemanticSegmentationTaskBinaryLoss.load_from_checkpoint(checkpoint_path)
    task.eval()  # Set to evaluation mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task.to(device)
    
    print(f"Model loaded: {type(task)}")
    print(f"Model is on device: {next(task.parameters()).device}")
    print(f"Model has segmentation head: {hasattr(task, 'segmentation_head')}")
    
    # Extract tile size from checkpoint path
    tile_size = int(re.search(r'_p(\d+)', str(checkpoint_path)).group(1))
    print(f"Using tile size: {tile_size}")
    
    # Iterate over both image2 configurations
    for i, config in enumerate(image2_configs, 1):
        print(f"\n--- Processing year range {i}/{len(image2_configs)}: {config['year']} ---")
        print(f"Image 2: {config['path']}")
        
        # Define output path for this year range
        output_path = experiment_dir / config['year'] / Path('change_prediction_' + config['year'] + '_' + checkpoint_path.stem + '.tif')
        print(f"Output: {output_path}")
        
        # Run prediction
        print("Running prediction on large image...")
        prediction = predict_large_image(
            model=task,
            image1_path=image1_path,
            image2_path=config['path'],
            output_path=output_path,
            tile_size=tile_size,
            overlap=64  # Use overlap to avoid edge artifacts
        )
        
        print(f"Prediction completed for {config['year']}! Output saved to: {output_path}")
    
    print(f"\nAll {len(image2_configs)} predictions completed!")

if __name__ == "__main__":
    main()
