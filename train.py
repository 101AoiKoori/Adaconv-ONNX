import argparse
import yaml
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
    # 微调模式参数
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="启用微调模式：加载检查点但重置step和学习率",
    )
    # 检查点路径参数
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="加载指定的检查点文件进行训练或微调",
    )
    # 微调学习率参数
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=None,
        help="微调时使用的学习率（通常比初始训练小）",
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
    
    # 添加微调相关配置
    if opt.checkpoint:
        config_data["checkpoint_path"] = opt.checkpoint
    if opt.finetune_lr:
        config_data["finetune_lr"] = opt.finetune_lr
    elif opt.finetune and not opt.finetune_lr:
        # 如果是微调但未指定微调学习率，则默认为原学习率的十分之一
        config_data["finetune_lr"] = config_data.get("learning_rate", 0.0001) * 0.1

    # Create Hyperparameter object
    config = Hyperparameter(**config_data)

    # Initialize trainer
    trainer = Trainer(config)

    # Print configuration
    print("Training Configuration:")
    print(config.model_dump_json(indent=4))

    # Start training - 传递finetune参数
    trainer.train(fine_tuning=opt.finetune)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)