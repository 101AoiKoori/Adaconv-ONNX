"""Test adaconv model."""

import argparse
from pathlib import Path
import math

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

    with open(config, "r", encoding="utf8") as f:
        config_data = yaml.safe_load(f)
    config_data.update({"data_path": ""})
    config_data.update({"logdir": ""})

    hyper_param = Hyperparameter(**config_data)
    fixed_batch_size = hyper_param.fixed_batch_size

    model = StyleTransfer(
        image_shape=tuple(hyper_param.image_shape),
        style_dim=hyper_param.style_dim,
        style_kernel=hyper_param.style_kernel,
        groups=hyper_param.groups,
        fixed_batch_size=fixed_batch_size,
    ).to(device)

    checkpoint = torch.load(model_ckpt, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    transforms = get_transform(resize=hyper_param.image_shape)

    grid_image = [torch.zeros((3, *hyper_param.image_shape), dtype=torch.float32).cpu()]

    total_steps = (len(content_img_paths) // fixed_batch_size) * (len(style_img_paths) // fixed_batch_size)
    progress_bar = tqdm(total=total_steps)

    for i in range(0, len(content_img_paths), fixed_batch_size):
        content_batch_paths = content_img_paths[i:i + fixed_batch_size]
        if len(content_batch_paths) < fixed_batch_size:
            continue
        content_batch = []
        for content_img_path in content_batch_paths:
            content_img = transforms(read_image(content_img_path)).cpu()
            content_batch.append(content_img)
            grid_image.append(content_img)
        content_batch = torch.stack(content_batch, dim=0).to(device)

        for j in range(0, len(style_img_paths), fixed_batch_size):
            style_batch_paths = style_img_paths[j:j + fixed_batch_size]
            if len(style_batch_paths) < fixed_batch_size:
                continue
            style_batch = []
            for style_img_path in style_batch_paths:
                style_img = transforms(read_image(style_img_path)).cpu()
                # 确保风格图片维度为 [3, 256, 256]
                if style_img.dim() == 4:
                    style_img = style_img.squeeze(0)
                style_batch.append(style_img)
                grid_image.append(style_img)
            style_batch = torch.stack(style_batch, dim=0).to(device)

            assert content_batch.shape[0] == style_batch.shape[0], "Batch size mismatch"
            style_content_batch = model(
                content=content_batch,
                style=style_batch,
            )
            for style_content_img in style_content_batch.detach().cpu():
                # 确保推理结果维度为 [3, 256, 256]
                if style_content_img.dim() == 4:
                    style_content_img = style_content_img.squeeze(0)
                grid_image.append(style_content_img)
            progress_bar.update(1)

    progress_bar.close()

    # 计算合适的 nrow
    total_images = len(grid_image)
    nrow = int(math.ceil(math.sqrt(total_images)))

    save_image(grid_image, output_path, nrow=nrow)


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
    