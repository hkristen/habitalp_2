"""
Inference script for habitalp2 semantic segementation models (=post-classificaiton).

Supports both Clay (terratorch) and UNet models with YAML configuration files.

Usage:
    # Clay model inference with config file
    python src/inference/inference.py --config configs/clay_data_2013.yaml

    # Override checkpoint
    python src/inference/inference.py --config configs/clay_data_2020.yaml \
        --checkpoint path/to/clay.ckpt

    # Override ROI
    python src/inference/inference.py --config configs/clay_data_2013.yaml \
        --roi path/to/different_roi.gpkg

    # Full extent inference (no ROI file)
    python src/inference/inference.py --config configs/clay_data_2020.yaml --roi none

    # Override output filename
    python src/inference/inference.py --config configs/clay_data_2013.yaml \
        --output my_custom_name.tif
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
import yaml
from terratorch.tasks import SemanticSegmentationTask

BASE_FOLDER = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_FOLDER))

from src.data.datamodules.unet_change import SegmentationDataModule
from src.inference.eval import infer_on_whole_image
from src.trainers.unet_segmentation.unet_segmentation import (
    MultiClassSemanticSegmentationTask,
)

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'



def load_config(config_path):
    """Load configuration from YAML file.

    Args:
        config_path (Path): Path to YAML config file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_clay_model(checkpoint_path, device):
    """Load Clay/terratorch model from checkpoint.

    Args:
        checkpoint_path (Path): Path to fine-tuned model checkpoint (.ckpt).
        device (torch.device): Device to load model on (cuda or cpu).

    Returns:
        SemanticSegmentationTask: Model in eval mode.
    """
    print(f"Loading Clay model from: {checkpoint_path}")
    task = SemanticSegmentationTask.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    task.eval()
    return task


def load_unet_model(checkpoint_path, device):
    """Load UNet model from checkpoint.

    Args:
        checkpoint_path (Path): Path to UNet model checkpoint (.ckpt).
        device (torch.device): Device to load model on (cuda or cpu).

    Returns:
        MultiClassSemanticSegmentationTask: Model in eval mode. Uses strict=False
            if state_dict keys mismatch.
    """
    print(f"Loading UNet model from: {checkpoint_path}")
    try:
        task = MultiClassSemanticSegmentationTask.load_from_checkpoint(checkpoint_path)
    except RuntimeError as e:
        if "Unexpected key(s) in state_dict" in str(e):
            print(f"Warning: {e}")
            print("Attempting to load with strict=False...")
            task = MultiClassSemanticSegmentationTask.load_from_checkpoint(checkpoint_path, strict=False)
        else:
            raise
    task.to(device).eval()
    return task


def main():
    """Main entry point for inference script.

    Loads config, model, and runs whole-image inference with tiling and blending.
    Output filename auto-generated as: {roi_name}_{checkpoint_name}.tif
    """
    parser = argparse.ArgumentParser(
        description="Production inference for Clay and UNet models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clay model inference with 2013 data
  python src/inference/inference.py --config configs/clay_data_2013.yaml

  # Override checkpoint
  python src/inference/inference.py --config configs/clay_data_2020.yaml --checkpoint path/to/model.ckpt

  # Full extent inference (no ROI)
  python src/inference/inference.py --config configs/clay_data_2020.yaml --roi none

  # Custom output filename
  python src/inference/inference.py --config configs/clay_data_2013.yaml --output my_result.tif
        """
    )

    # Required configuration file
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file with data paths and parameters"
    )

    # Optional overrides
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path from config"
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Override ROI path from config. Use 'none' for full extent inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output filename (auto-generated if not specified)"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    args = parser.parse_args()

    # Load configuration file
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Look for config relative to script directory
        script_dir = Path(__file__).parent
        config_path = script_dir / args.config

    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    print("✓ Configuration loaded")

    # Determine model type from config
    model_type = config.get('model_type', 'clay')  # Default to clay if not specified

    # Get model-specific config
    if model_type not in config['models']:
        raise ValueError(f"Model type '{model_type}' not found in config. Available: {list(config['models'].keys())}")

    model_config = config['models'][model_type]

    # Load paths from config with CLI overrides
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(model_config['checkpoint'])
    roi_path = args.roi if args.roi is not None else config.get('roi_path')
    output_dir = Path(config['output']['dir'])

    # Get inference parameters from config
    inference_config = config['inference']
    patch_size = inference_config['patch_size']
    overlap = inference_config['overlap']
    delta = inference_config['delta']
    batch_size = inference_config['batch_size']
    n_classes = inference_config['n_classes']

    # Get input data paths from config
    input_data = {k: Path(v) for k, v in config['input_data'].items()}
    mask_path = Path(config['mask_path'])

    if args.output:
        # Remove .tif extension if user provided it
        output_filename = args.output.removesuffix('.tif')
    else:
        # Smart default: ROI_name + checkpoint_name
        roi_name = Path(roi_path).stem if roi_path and roi_path.lower() != 'none' else "full_extent"
        checkpoint_name = checkpoint_path.stem
        output_filename = f"{roi_name}_{checkpoint_name}"

    # Process ROI path
    output_dir.mkdir(exist_ok=True, parents=True)
    roi_path = None if roi_path and roi_path.lower() == "none" else roi_path

    # Print configuration
    print("=" * 80)
    print(f"{model_type.upper()} Model Inference")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"ROI: {roi_path if roi_path else 'Full extent'}")
    print("Input data:")
    for key, path in input_data.items():
        print(f"  {key}: {path}")
    print(f"Mask: {mask_path}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Overlap: {overlap}px")
    print(f"Delta cropping: {delta}px")
    print(f"Batch size: {batch_size}")
    print(f"Classes: {n_classes}")
    print(f"Output: {output_dir / output_filename}")
    print("=" * 80)

    # Load model
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if model_type == "clay":
        task = load_clay_model(checkpoint_path, device)
    elif model_type == "unet":
        task = load_unet_model(checkpoint_path, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print("✓ Model loaded successfully")

    # Initialize datamodule
    print("\nInitializing datamodule...")
    datamodule = SegmentationDataModule(
        image_paths=input_data,
        mask_path=str(mask_path),
        roi_shape_path=roi_path,
        n_classes=n_classes,
        batch_size=1,
        patch_size=(patch_size, patch_size),
        num_workers=0,
        image_paths_pred=input_data,
        prediction_mask_path=str(mask_path),
    )
    print("✓ Datamodule initialized")

    # Run inference
    print("\nRunning inference...")
    print("-" * 80)

    try:
        infer_on_whole_image(
            datamodule=datamodule,
            task=task,
            experiment_dir=str(output_dir),
            patch_size=patch_size,
            overlap=overlap,
            delta=delta,
            predict_on_test_ds=False,
            output_filename=output_filename,
            inference_batch_size=batch_size,
        )
    except RuntimeError as e:
        if "expected input" in str(e) and "channels" in str(e):
            print("\n" + "=" * 80)
            print("ERROR: Input channel mismatch!")
            print("=" * 80)
            print(f"{e}\n")
            print("This usually means the model was trained with different input bands")
            print("than what's provided in the data folder.")
            print("\nSolution: Either:")
            print("  1. Use a checkpoint trained with the current input bands (rgb, cir, ndsm)")
            print("  2. Update --data_folder to match the bands the model expects")
            print("=" * 80)
            sys.exit(1)
        else:
            raise

    print("-" * 80)
    print("\n✓ Inference complete!")
    print(f"✓ Output saved to: {output_dir / output_filename}")


if __name__ == "__main__":
    main()
