"""Test adaconv model."""

import argparse
from pathlib import Path

import torch
import yaml
from dataloader import get_transform
from hyperparam import Hyperparameter
from model import StyleTransfer
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm


def get_images(input_path: Path) -> list[Path]:
    image_paths = []
    if input_path.is_dir():
        for ext in ["png", "jpg", "jpeg"]:
            image_paths += sorted(input_path.glob(f"*.{ext}"))
    else:
        image_paths = [input_path]
    return image_paths


def read_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to model config file",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        help="Path to model ckpt file",
    )
    parser.add_argument(
        "--content_path",
        type=str,
        help="Input Content Image or Input Content Images Dir",
    )
    parser.add_argument(
        "--style_path",
        type=str,
        help="Input Style Image or Input Style Images Dir",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output file path to save images",
    )

    opt = parser.parse_args()
    return opt


def main(config, model_ckpt, content_path, style_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    content_img_paths = get_images(Path(content_path))
    style_img_paths = get_images(Path(style_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    config_data.update({"data_path": ""})
    config_data.update({"logdir": ""})

    hyper_param = Hyperparameter(**config_data)

    model = StyleTransfer(
        image_shape=tuple(hyper_param.image_shape),
        style_dim=hyper_param.style_dim,
        style_kernel=hyper_param.style_kernel,
    ).to(device)

    checkpoint = torch.load(model_ckpt, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    transforms = get_transform(resize=hyper_param.image_shape)

    grid_image = [torch.zeros((3, *hyper_param.image_shape), dtype=torch.float32).cpu()]
    for content_img_path in content_img_paths:
        content_img = transforms(read_image(content_img_path)).cpu()
        grid_image.append(content_img)

    for style_img_path in style_img_paths:
        print(f"Style Image -> {style_img_path}")
        style_img = torch.unsqueeze(transforms(read_image(style_img_path)).cpu(), dim=0)
        grid_image.append(style_img[0])
        for content_img_path in tqdm(content_img_paths):
            content_img = torch.unsqueeze(
                transforms(read_image(content_img_path)).cpu(), dim=0
            )
            style_content_img = model(
                content=content_img.to(device),
                style=style_img.to(device),
            )
            grid_image.append(style_content_img.detach().cpu()[0])

    save_image(grid_image, output_path, nrow=len(content_img_paths) + 1)


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
