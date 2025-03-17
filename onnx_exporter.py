import torch
import argparse
import onnx
import yaml
from model import StyleTransfer
from onnx import shape_inference
from typing import Tuple, Dict, Optional
from pathlib import Path
from hyperparam import Hyperparameter


def load_config(config_path: str) -> Hyperparameter:
    """Load and parse configuration from YAML file into Hyperparameter object"""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Convert to Hyperparameter object
    return Hyperparameter(**config)


def export_onnx(
    model: torch.nn.Module,
    dummy_input: Tuple[torch.Tensor, torch.Tensor],
    output_path: str,
    dynamic_axes: Optional[Dict] = None,
    opset: int = 16
):
    """Export model to ONNX format"""
    try:
        # Execute ONNX export
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
        
        # Shape inference
        model_onnx = onnx.load(output_path)
        inferred_model = shape_inference.infer_shapes(model_onnx)
        onnx.save(inferred_model, output_path)
        print(f"✅ ONNX export successful: {output_path}")
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        raise


def initialize_model(hyper_param: Hyperparameter, device: str) -> torch.nn.Module:
    """Initialize model with the correct parameters"""
    # Calculate groups if not explicitly set
    if hyper_param.groups is None and hyper_param.groups_list is None:
        base_channels = [512, 256, 128, 64]
        groups_list = [
            max(1, int(c * r)) 
            for c, r in zip(base_channels, hyper_param.groups_ratios)
        ]
        print(f"Auto-calculated groups: {groups_list}")
    elif hyper_param.groups is not None:
        # If groups is a single int, repeat it 4 times
        if isinstance(hyper_param.groups, int):
            groups_list = [hyper_param.groups] * 4
        else:
            groups_list = hyper_param.groups
    else:
        groups_list = hyper_param.groups_list
    
    # Validate groups
    if len(groups_list) != 4:
        raise ValueError("groups parameter must contain 4 elements for each decoder layer")

    return StyleTransfer(
        image_shape=hyper_param.image_shape,
        style_dim=hyper_param.style_dim,
        style_kernel=hyper_param.style_kernel,
        groups=groups_list,
        fixed_batch_size=hyper_param.fixed_batch_size if not hyper_param.use_fixed_size else None,
        use_fixed_size=hyper_param.use_fixed_size
    ).to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str):
    """Load weights from checkpoint with compatibility handling"""
    # 显式设置 weights_only=True
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Parameter name mapping for different versions
    name_mapping = {
        'kernel_predictor.depthwise_conv_kernel_predictor': 'kernel_predictor.0',
        'kernel_predictor.pointwise_conv_kernel_predictor': 'kernel_predictor.2'
    }
    
    # Parameter shape fixing
    new_state_dict = {}
    for name, param in state_dict.items():
        # Skip non-matching parameters
        if name not in model.state_dict():
            print(f"⚠️ Skipping non-matching parameter: {name}")
            continue
            
        # Auto-view FC layer weights if needed
        if 'fc' in name and param.dim() == 2:
            param = param.view(*model.state_dict()[name].shape)
            
        new_state_dict[name] = param
    
    # Load with strict=False to allow missing params
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    # Print debug info
    if missing:
        print(f"❌ Missing parameters: {missing}")
    if unexpected:
        print(f"❌ Unexpected parameters: {unexpected}")
    
    return model


def main(
    checkpoint_path: str,
    config_path: str,
    output_path: str,
    opset: int = 16
):
    # Load configuration
    hyper_param = load_config(config_path)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    # Initialize model
    model = initialize_model(hyper_param, device)
    
    # Load weights
    model = load_checkpoint(model, checkpoint_path, device)
    
    # Set to evaluation mode
    model.eval()
    
    # Generate dummy input
    batch_size = 1 if not hyper_param.use_fixed_size else hyper_param.fixed_batch_size
    dummy_content = torch.randn(batch_size, 3, *hyper_param.image_shape, device=device)
    dummy_style = torch.randn(batch_size, 3, *hyper_param.image_shape, device=device)
    
    # Dynamic axes configuration
    dynamic_axes = None
    if not hyper_param.use_fixed_size:
        dynamic_axes = {
            'content': {0: 'batch_size', 2: 'height', 3: 'width'},
            'style': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
        print("🔄 Dynamic export mode enabled")
    else:
        print("🔒 Static export mode enabled")
    
    # Execute export
    export_onnx(
        model=model,
        dummy_input=(dummy_content, dummy_style),
        output_path=output_path,
        dynamic_axes=dynamic_axes,
        opset=opset
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX exporter based on YAML configuration")
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint path (.pt)")
    parser.add_argument("--config", default="lambda100.yaml", help="Config file path (default: lambda100.yaml)")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version (default: 16)")
    
    args = parser.parse_args()
    
    try:
        main(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            output_path=args.output,
            opset=args.opset
        )
    except Exception as e:
        print(f"❌ Export process terminated with exception: {str(e)}")
        exit(1)