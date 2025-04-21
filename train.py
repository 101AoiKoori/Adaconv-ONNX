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
        "--style_weight",
        type=float,
        default=None,
        help="Style weight for loss calculation (overrides config if provided)",
    )
    # New finetune-related arguments
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Enable fine-tuning mode",
    )

    opt = parser.parse_args()

    # Validate finetune mode arguments
    if opt.finetune:
        # In finetune mode, pretrained model path is read from config
        pass

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
    if opt.style_weight:
        config_data["style_weight"] = opt.style_weight

    if opt.finetune:
        # Set logdir to logs/finetune in finetune mode
        config_data["logdir"] = str(Path(config_data.get("logdir", "logs")) / "finetune")
        config_data["learning_rate"] = config_data.get("finetune_learning_rate", config_data["learning_rate"] * 0.1)
        config_data["num_iteration"] = config_data.get("finetune_iterations", int(config_data["num_iteration"] * 0.25))
        pretrained_model_path = Path(config_data.get("pretrained_model", ""))
        if not pretrained_model_path.exists():
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_model_path}")

    hyper_param = Hyperparameter(**config_data)

    trainer = Trainer(
        hyper_param=hyper_param,
        finetune_mode=opt.finetune,
        pretrained_model_path=config_data.get("pretrained_model") if opt.finetune else None
    )

    trainer.train()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)