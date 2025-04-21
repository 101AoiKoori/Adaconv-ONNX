"""
Simplified ONNX export script for AdaConv - Compatible with the new version of StyleTransfer

Usage:
    python exporter.py --output model.onnx  # Use default settings
    python exporter.py --output model.onnx --static  # Static mode
    python exporter.py --output model.onnx --dynamic  # Fully dynamic mode
    python exporter.py --output model.onnx --dynamic-batch  # Dynamic batch size
"""

import argparse
import os
import yaml
import torch
from pathlib import Path
from models.model import StyleTransfer
from models.encoder import Encoder
from hyperparam.hyperparam import Hyperparameter
import warnings
warnings.filterwarnings(
        "ignore",
        message="Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.",
        category=UserWarning,
        module="torch.onnx"
    )


def get_default_paths():
    """Get default paths and parameters"""
    # Default paths - Modify according to actual situation
    ckpt_dir = Path("./logs/ckpts")
    config_dir = Path("./configs")
    
    # Check if the directories exist, create them if not
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the latest checkpoint
    checkpoint_path = ckpt_dir / "last.pt"
    if not checkpoint_path.exists():
        # Find other checkpoints
        ckpts = list(ckpt_dir.glob("*.pt"))
        if ckpts:
            checkpoint_path = sorted(ckpts)[-1]  # Use the latest checkpoint
        else:
            raise FileNotFoundError(f"No model checkpoints found. Please make sure there are checkpoint files in the {ckpt_dir} directory.")
    
    # Get the configuration file
    config_path = config_dir / "lambda100.yaml"
    if not config_path.exists():
        # Find other configuration files
        configs = list(config_dir.glob("*.yaml"))
        if configs:
            config_path = configs[0]  # Use the first found configuration
        else:
            raise FileNotFoundError(f"No configuration files found. Please make sure there are YAML configuration files in the {config_dir} directory.")
    
    return str(checkpoint_path), str(config_path)


def load_config(config_path):
    """Load the configuration file"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return Hyperparameter(**config)


def initialize_model(hyper_param, device, export_config=None):
    """Initialize the model, compatible with the new StyleTransfer constructor"""
    # Process the groups parameter
    if hyper_param.groups_list:
        groups = hyper_param.groups_list
    elif hyper_param.groups:
        groups = hyper_param.groups
    else:
        # Calculate the groups according to the ratio
        base_channels = [512, 256, 128, 64]
        groups = [max(1, int(c * r)) for c, r in zip(base_channels, hyper_param.groups_ratios)]
        print(f"Automatically calculated groups: {groups}")
    
    # Create the model
    return StyleTransfer(
        image_shape=hyper_param.image_shape,
        style_dim=hyper_param.style_dim,
        style_kernel=hyper_param.style_kernel,
        groups=groups,
        export_config=export_config
    ).to(device)


def load_checkpoint(model, checkpoint_path, device):
    """Load the checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Checkpoint loaded successfully: {checkpoint_path}")
    return model


def export_onnx(model, dummy_input, output_path, dynamic_axes=None, opset=16):
    """Export the model to ONNX format"""
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=output_path,
        input_names=["content", "style"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL
    )
    print(f"✅ ONNX export successful: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AdaConv ONNX Export Tool - Custom Version")
    
    # Basic parameters
    parser.add_argument("--output", required=True, help="Output ONNX model path (e.g., models/adaconv.onnx)")
    parser.add_argument("--checkpoint", help="Model checkpoint path (.pt file)")
    parser.add_argument("--config", help="Configuration file path (.yaml file)")
    parser.add_argument("--opset", type=int, default=16, help="ONNX operator set version (default: 16)")
    parser.add_argument("--batch-size", type=int, default=1, help="Export batch size (default: 1)")
    
    # Export mode options (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--static", action="store_true", help="Export in fully static mode (fixed batch size and dimensions)")
    mode_group.add_argument("--dynamic", action="store_true", help="Export in fully dynamic mode (dynamic batch size and dimensions)")
    mode_group.add_argument("--dynamic-batch", action="store_true", help="Export in dynamic batch size mode (fixed dimensions)")
    mode_group.add_argument("--dynamic-shape", action="store_true", help="Export in dynamic dimension mode (fixed batch size)")
    
    args = parser.parse_args()
    
    # Create the output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get default paths
    try:
        default_checkpoint, default_config = get_default_paths()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Use command-line arguments or default values
    checkpoint_path = args.checkpoint or default_checkpoint
    config_path = args.config or default_config
    
    # Set the export mode based on the parameters
    is_static = args.static
    dynamic_batch = args.dynamic or args.dynamic_batch
    dynamic_shape = args.dynamic or args.dynamic_shape
    
    # Print export information
    print("\n" + "="*60)
    print("AdaConv ONNX Export Tool - Custom Version")
    print("="*60)
    print(f"Checkpoint file: {checkpoint_path}")
    print(f"Configuration file: {config_path}")
    print(f"Output file: {args.output}")
    print(f"Batch size: {args.batch_size}")
    
    # Print the export mode
    if is_static:
        print("Export mode: Fully static (fixed batch size and dimensions)")
    elif dynamic_batch and dynamic_shape:
        print("Export mode: Fully dynamic (dynamic batch size and dimensions)")
    elif dynamic_batch:
        print("Export mode: Dynamic batch size (fixed dimensions)")
    elif dynamic_shape:
        print("Export mode: Dynamic dimensions (fixed batch size)")
    else:
        print("Export mode: Default (only batch size is dynamic)")
        dynamic_batch = True  # Enable dynamic batch size by default
    
    print("="*60 + "\n")
    
    try:
        # Load the configuration
        hyper_param = load_config(config_path)
        
        # Set the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device.upper()}")
        
        # Prepare the export configuration
        export_config = {
            'export_mode': True,
            'fixed_batch_size': args.batch_size if is_static or not dynamic_batch else None,
            'use_fixed_size': is_static or not dynamic_shape
        }
        
        # Initialize the model
        model = initialize_model(hyper_param, device, export_config)
        
        # Load the weights
        model = load_checkpoint(model, checkpoint_path, device)
        
        # Set the model to evaluation mode
        model.eval()
        
        # Generate dummy inputs
        input_batch_size = args.batch_size
        dummy_content = torch.randn(input_batch_size, 3, *hyper_param.image_shape, device=device)
        dummy_style = torch.randn(input_batch_size, 3, *hyper_param.image_shape, device=device)
        
        # Dynamic axes configuration
        dynamic_axes = None
        if not is_static:
            dynamic_axes = {}
            if dynamic_batch:
                # Make the batch dimension dynamic
                dynamic_axes.update({
                    'content': {0: 'batch_size'},
                    'style': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                })
            if dynamic_shape:
                # Make the spatial dimensions dynamic
                dynamic_axes.update({
                    'content': {2: 'height', 3: 'width'},
                    'style': {2: 'height', 3: 'width'},
                    'output': {2: 'height', 3: 'width'}
                })
        
        # Perform the export
        export_onnx(
            model=model,
            dummy_input=(dummy_content, dummy_style),
            output_path=args.output,
            dynamic_axes=dynamic_axes,
            opset=args.opset
        )
        
        print(f"\n✅ Model exported successfully to: {args.output}")
        print(f"   - Batch size: {'Dynamic' if dynamic_batch else args.batch_size}")
        print(f"   - Dimension mode: {'Dynamic' if dynamic_shape else 'Fixed'}")
        
    except Exception as e:
        import traceback
        print(f"\n❌ Export failed: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()