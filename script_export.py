import argparse
from utils.onnx_exporter import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX with preset paths")
    parser.add_argument("--output", required=True, help="Custom output ONNX model name (e.g. my_model.onnx)")
    
    args = parser.parse_args()
    
    # Preset paths (modify these if your paths change)
    PRESET_CHECKPOINT = "./logs/ckpts/last.pt"
    PRESET_CONFIG = "./configs/lambda100.yaml"
    
    main(
        checkpoint_path=PRESET_CHECKPOINT,
        config_path=PRESET_CONFIG,
        output_path=args.output,
        opset=16  # 使用默认opset版本
    )