import torch
import argparse
import onnx
import yaml
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List, Any

from models.model import StyleTransfer
from onnx import shape_inference
from hyperparam.hyperparam import Hyperparameter

# Ignore ONNX warnings
import warnings
warnings.filterwarnings(
    "ignore",
    message="Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.",
    category=UserWarning,
    module="torch.onnx"
)


def load_config(config_path: str) -> Hyperparameter:
    """
    Load the configuration from a YAML file and convert it to a Hyperparameter object

    Args:
        config_path: Path to the configuration file

    Returns:
        Parsed Hyperparameter object
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Convert to a Hyperparameter object
    return Hyperparameter(**config)


def export_onnx(
    model: torch.nn.Module,
    dummy_input: Tuple[torch.Tensor, torch.Tensor],
    output_path: str,
    dynamic_axes: Optional[Dict] = None,
    opset: int = 16
) -> bool:
    """
    Export the model to ONNX format

    Args:
        model: PyTorch model
        dummy_input: Dummy input (content image, style image)
        output_path: Path to the output ONNX file
        dynamic_axes: Dynamic axes configuration (optional)
        opset: ONNX operator set version

    Returns:
        Whether the export was successful
    """
    try:
        # Create the output directory
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Perform the ONNX export
        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=output_path,
            input_names=["content", "style"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False
        )

        # Shape inference
        model_onnx = onnx.load(output_path)
        inferred_model = shape_inference.infer_shapes(model_onnx)
        onnx.save(inferred_model, output_path)

        print(f"✅ ONNX export successful: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        return False


def get_groups_config(hyper_param: Hyperparameter) -> List[int]:
    """
    Get the grouped convolution parameters according to the configuration

    Args:
        hyper_param: Hyperparameter configuration

    Returns:
        List of grouped convolution parameters
    """
    # Calculate the groups parameter (if not explicitly set)
    if hyper_param.groups_list:
        return hyper_param.groups_list
    elif hyper_param.groups:
        # If groups is a single integer, repeat it 4 times
        if isinstance(hyper_param.groups, int):
            return [hyper_param.groups] * 4
        return hyper_param.groups
    else:
        # Calculate according to the ratio
        base_channels = [512, 256, 128, 64]
        groups_list = [
            max(1, int(c * r))
            for c, r in zip(base_channels, hyper_param.groups_ratios)
        ]
        print(f"Automatically calculated groups parameters: {groups_list}")
        return groups_list


def initialize_model(hyper_param: Hyperparameter, device: str) -> torch.nn.Module:
    """
    Initialize the model with the correct parameters

    Args:
        hyper_param: Hyperparameter configuration
        device: Device ('cuda' or 'cpu')

    Returns:
        Initialized model
    """
    # Get the groups configuration
    groups_list = get_groups_config(hyper_param)

    # Create the export configuration
    export_config = {
        'export_mode': True,
        'fixed_batch_size': hyper_param.fixed_batch_size,
        'use_fixed_size': hyper_param.use_fixed_size
    }

    # Initialize the model
    model = StyleTransfer(
        image_shape=hyper_param.image_shape,
        style_dim=hyper_param.style_dim,
        style_kernel=hyper_param.style_kernel,
        groups=groups_list,
        export_config=export_config
    ).to(device)

    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str):
    """
    Load the weights from the checkpoint (including compatibility handling)

    Args:
        model: Model
        checkpoint_path: Path to the checkpoint
        device: Device ('cuda' or 'cpu')

    Returns:
        Model with loaded weights
    """
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract the state dictionary (compatible with different formats)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Load the state dictionary (use strict=False to allow missing parameters)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Print debug information
        if missing:
            print(f"❓ Missing parameters: {len(missing)}")
        if unexpected:
            print(f"⚠️ Unexpected parameters: {len(unexpected)}")

        print(f"✅ Checkpoint loaded successfully: {checkpoint_path}")
        return model

    except Exception as e:
        print(f"❌ Failed to load the checkpoint: {str(e)}")
        raise


def main(
    checkpoint_path: str,
    config_path: str,
    output_path: str,
    opset: int = 16
):
    """
    Main export function

    Args:
        checkpoint_path: Path to the checkpoint
        config_path: Path to the configuration file
        output_path: Path to the output ONNX file
        opset: ONNX operator set version
    """
    # Load the configuration
    print(f"🔧 Loading configuration: {config_path}")
    hyper_param = load_config(config_path)

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device.upper()}")

    # Initialize the model
    print(f"🔧 Initializing the model...")
    model = initialize_model(hyper_param, device)

    # Load the weights
    print(f"🔧 Loading the checkpoint: {checkpoint_path}")
    model = load_checkpoint(model, checkpoint_path, device)

    # Set the model to evaluation mode
    model.eval()

    # Generate dummy inputs
    print(f"🔧 Preparing dummy inputs...")
    batch_size = hyper_param.fixed_batch_size if hyper_param.fixed_batch_size else 1
    dummy_content = torch.randn(batch_size, 3, *hyper_param.image_shape, device=device)
    dummy_style = torch.randn(batch_size, 3, *hyper_param.image_shape, device=device)

    # Dynamic axes configuration
    dynamic_axes = None
    if not hyper_param.use_fixed_size:
        dynamic_axes = {
            'content': {2: 'height', 3: 'width'},
            'style': {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'}
        }

        # If fixed_batch_size is not set, the batch size is also dynamic
        if not hyper_param.fixed_batch_size:
            dynamic_axes['content'][0] = 'batch_size'
            dynamic_axes['style'][0] = 'batch_size'
            dynamic_axes['output'][0] = 'batch_size'

    # Perform the export
    print(f"🔧 Exporting the model...")
    success = export_onnx(
        model=model,
        dummy_input=(dummy_content, dummy_style),
        output_path=output_path,
        dynamic_axes=dynamic_axes,
        opset=opset
    )

    if success:
        # Get the size of the exported file
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        print(f"📊 Exported file size: {file_size:.2f} MB")

        # Print a summary of the export mode
        mode_str = "Static mode" if hyper_param.use_fixed_size else "Dynamic mode"
        batch_str = f"Fixed batch size: {batch_size}" if hyper_param.fixed_batch_size else "Dynamic batch size"
        shape_str = f"Fixed spatial dimensions: {hyper_param.image_shape}" if hyper_param.use_fixed_size else "Dynamic spatial dimensions"

        print(f"✅ Export completed: {output_path}")
        print(f"   - Mode: {mode_str}")
        print(f"   - {batch_str}")
        print(f"   - {shape_str}")
        print(f"   - ONNX operator set: {opset}")
    else:
        print(f"❌ Export failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaConv ONNX Export Tool")
    parser.add_argument("--checkpoint", required=True, help="Path to the PyTorch checkpoint (.pt)")
    parser.add_argument("--config", default="configs/lambda100.yaml", help="Path to the configuration file")
    parser.add_argument("--output", required=True, help="Path to the output ONNX file")
    parser.add_argument("--opset", type=int, default=16, help="ONNX operator set version (default: 16)")

    args = parser.parse_args()

    try:
        main(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            output_path=args.output,
            opset=args.opset
        )
    except Exception as e:
        import traceback
        print(f"❌ Export process terminated due to an exception:")
        traceback.print_exc()
        exit(1)