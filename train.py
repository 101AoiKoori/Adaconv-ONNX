import argparse
import yaml
from pathlib import Path
from hyperparam.hyperparam import Hyperparameter
from trainers.trainer import Trainer

def parse_opt():
    parser = argparse.ArgumentParser(description="Style Transfer Training")

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to model config file",
    )
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default=None,
        help="Path to dataset (overrides config if provided)",
    )
    parser.add_argument(
        "--logdir",
        "-l",
        type=str,
        default=None,
        help="Log directory path (overrides config if provided)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config if provided)",
    )
    parser.add_argument(
        "--num_iteration",
        type=int,
        default=None,
        help="Number of training iterations (overrides config if provided)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config if provided)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Image size (overrides config if provided)",
    )
    parser.add_argument(
        "--fine_tune",
        "-f",
        action="store_true",
        help="Enable fine-tuning mode (resets optimizer but keeps model weights)",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume training from checkpoint if available",
    )

    opt = parser.parse_args()

    return opt

def main(opt):
    # Load configuration from file
    with open(opt.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Override config with command line arguments if provided
    if opt.data_path:
        config_data["data_path"] = opt.data_path
    if opt.logdir:
        config_data["logdir"] = opt.logdir
    if opt.batch_size:
        config_data["batch_size"] = opt.batch_size
    if opt.num_iteration:
        config_data["num_iteration"] = opt.num_iteration
    if opt.learning_rate:
        config_data["learning_rate"] = opt.learning_rate
    if opt.image_size:
        config_data["image_size"] = opt.image_size

    # 如果未指定logdir，检查是否能找到训练状态文件来决定是否继续训练
    if not opt.logdir and not opt.fine_tune and not opt.resume:
        config_logdir = Path(config_data.get("logdir", "logs"))
        state_path = config_logdir / "training_state.json"
        
        if state_path.exists():
            import json
            with open(state_path, 'r') as f:
                training_state = json.load(f)
                
            if training_state.get("completed", False):
                print(f"Found completed training in {config_logdir}. Use --fine_tune to fine-tune it or specify a different --logdir.")
                return
            
            if not opt.resume:
                print(f"Found incomplete training in {config_logdir}. Use --resume to continue or specify a different --logdir.")
                return

    # Create Hyperparameter object
    config = Hyperparameter(**config_data)

    # Initialize trainer
    trainer = Trainer(config)

    # Print configuration
    print("Training Configuration:")
    print(config.model_dump_json(indent=4))
    
    if opt.fine_tune:
        print("Starting in fine-tuning mode")
        trainer.train(fine_tuning=True)
    else:
        print("Starting in normal training mode")
        trainer.train(fine_tuning=False)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)