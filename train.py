import argparse

import yaml
from hyperparam import Hyperparameter
from trainer import Trainer


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to model config file",
    )
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        help="Path to dataset",
    )
    parser.add_argument(
        "--logdir",
        "-l",
        type=str,
        help="Log directory path",
    )

    opt = parser.parse_args()

    return opt


def main(config: str, data_path: str, logdir: str):
    with open(config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    config_data.update({"data_path": ""})
    config_data.update({"logdir": ""})

    config: Hyperparameter = Hyperparameter(**config_data)
    if data_path:
        config.data_path = data_path
    if logdir:
        config.logdir = logdir

    trainer = Trainer(config)

    print(config.model_dump_json(indent=4))

    trainer.train()


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
