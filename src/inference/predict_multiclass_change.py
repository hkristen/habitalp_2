#!/usr/bin/env python3

from pathlib import Path
import torch
import re
from ..trainers.utils import predict_large_image_multiclass
from ..trainers.change import MultiClassChangeSemanticSegmentationTask

def main():
    print("Starting multiclass change prediction...")
    
    experiment_dir = Path('/home/hkristen/Nextcloud/HabitAlp2.0/Originaldaten/model_output/direct_change/')
    
    # Two multiclass checkpoint paths from the notebook
    checkpoint_paths = [
        experiment_dir / Path('checkpoints/multiclass-sweep-9cls-OneCycleLR_efficientnet-b4_p256/multiclass-sweep-9cls-OneCycleLR_efficientnet-b4_p256-epoch=13-step=14588-val_jaccard=0.2096.ckpt'),
        experiment_dir / Path('checkpoints/multiclass-sweep-9cls-OneCycleLR_efficientnet-b4_p512/multiclass-sweep-9cls-OneCycleLR_efficientnet-b4_p512-epoch=09-step=20840-val_jaccard=0.2061.ckpt')
    ]
    
    # Define input paths
    image1_path = '/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2013/flug_2003_rgb_clipped.tif'
    
    # Two different image2 paths for different year ranges
    image2_configs = [
        {
            'path': '/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2013/flug_2013_rgb_clipped.tif',
            'year': '2003_2013'
        },
        {
            'path': '/home/hkristen/habitalp2/data/processed/orthos_rgb_2003_2024/flug_2019_2021_rgb_clipped.tif',
            'year': '2013_2020'
        }
    ]
    
    print(f"Image 1: {image1_path}")
    
    # Iterate over both checkpoints
    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths, 1):
        print(f"\n--- Processing checkpoint {checkpoint_idx}/{len(checkpoint_paths)} ---")
        print(f"Checkpoint: {checkpoint_path}")
        
        # Extract tile size from checkpoint path
        tile_size = int(re.search(r'_p(\d+)', str(checkpoint_path)).group(1))
        print(f"Using tile size: {tile_size}")
        
        # Load the model from checkpoint
        print("Loading model...")
        task = MultiClassChangeSemanticSegmentationTask.load_from_checkpoint(checkpoint_path, ignore_index=None)
        task.eval()  # Set to evaluation mode
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        task.to(device)
        
        print(f"Model loaded: {type(task)}")
        print(f"Model is on device: {next(task.parameters()).device}")
        
        # Iterate over both image2 configurations
        for config_idx, config in enumerate(image2_configs, 1):
            print(f"\n--- Processing year range {config_idx}/{len(image2_configs)}: {config['year']} ---")
            print(f"Image 2: {config['path']}")
            
            # Define output path for this combination
            output_path = experiment_dir / config['year'] / Path('change_prediction_multiclass_' + config['year'] + '_' + checkpoint_path.stem + '.tif')
            print(f"Output: {output_path}")
            
            # Run prediction
            print("Running multiclass prediction on large image...")
            prediction = predict_large_image_multiclass(
                model=task,
                image1_path=image1_path,
                image2_path=config['path'],
                output_path=output_path,
                tile_size=tile_size,
                overlap=64,  # Use overlap to avoid edge artifacts
                save_probability_map=False  # Set to True if you want probability maps
            )
            
            print(f"Prediction completed for {config['year']}! Output saved to: {output_path}")
        
        # Clear GPU memory
        del task
        torch.cuda.empty_cache()
        print(f"Completed checkpoint {checkpoint_idx}")
    
    print(f"\nAll predictions completed for {len(checkpoint_paths)} checkpoints and {len(image2_configs)} year ranges!")

if __name__ == "__main__":
    main()