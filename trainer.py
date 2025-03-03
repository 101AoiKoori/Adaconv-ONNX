import datetime
import os
from pathlib import Path

import torch
import yaml
from dataloader import ImageDataset, InfiniteDataLoader, get_transform
from hyperparam import Hyperparameter
from loss import MomentMatchingStyleLoss, MSEContentLoss
from model import StyleTransfer
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Trainer:
    """
    Model trainer class
    """

    def __init__(self, hyper_param: Hyperparameter):
        self.hyper_param = hyper_param
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup()
        print(f"Training Initialized -> device: {self.device}")

    def setup(self):
        self.model = StyleTransfer(
            image_shape=tuple(self.hyper_param.image_shape),
            style_dim=self.hyper_param.style_dim,
            style_kernel=self.hyper_param.style_kernel,
        ).to(self.device)

        self.content_loss_fn = MSEContentLoss()
        self.style_loss_fn = MomentMatchingStyleLoss()

        self.optimizer = Adam(
            self.model.parameters(), lr=self.hyper_param.learning_rate
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.hyper_param.learning_rate,
            total_steps=self.hyper_param.num_iteration,
        )

        self.step = 0

        self.content_train_dataloader = InfiniteDataLoader(
            ImageDataset(
                list(Path(self.hyper_param.data_path).glob("content/train*"))[0],
                transform=get_transform(
                    resize=self.hyper_param.resize_size,
                    crop_size=self.hyper_param.image_shape[-1],
                ),
            ),
            batch_size=self.hyper_param.batch_size,
            shuffle=True,
            num_workers=4,
        ).__iter__()

        self.style_train_dataloader = InfiniteDataLoader(
            ImageDataset(
                list(Path(self.hyper_param.data_path).glob("style/train*"))[0],
                transform=get_transform(
                    resize=self.hyper_param.resize_size,
                    crop_size=self.hyper_param.image_shape[-1],
                ),
            ),
            batch_size=self.hyper_param.batch_size,
            shuffle=True,
            num_workers=4,
        ).__iter__()

        self.content_test_dataloader = InfiniteDataLoader(
            ImageDataset(
                list(Path(self.hyper_param.data_path).glob("content/test*"))[0],
                transform=get_transform(
                    resize=self.hyper_param.resize_size,
                    crop_size=self.hyper_param.image_shape[-1],
                ),
            ),
            batch_size=self.hyper_param.batch_size,
            shuffle=True,
            num_workers=4,
        ).__iter__()

        self.style_test_dataloader = InfiniteDataLoader(
            ImageDataset(
                list(Path(self.hyper_param.data_path).glob("style/test*"))[0],
                transform=get_transform(
                    resize=self.hyper_param.resize_size,
                    crop_size=self.hyper_param.image_shape[-1],
                ),
            ),
            batch_size=self.hyper_param.batch_size,
            shuffle=True,
            num_workers=4,
        ).__iter__()

    def save_ckpts(self, ckpt_path):
        torch.save(
            {
                "steps": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            ckpt_path,
        )
        print(f"Saving ckpts to {ckpt_path} at {self.step}")

    def load_ckpts(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["steps"]
        print(f"Loaded ckpts from {ckpt_path}")

    def optimizer_step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

    def train(self):
        Path(self.hyper_param.logdir).mkdir(parents=True, exist_ok=True)
        with (Path(self.hyper_param.logdir) / "config.yaml").open("w") as outfile:
            yaml.dump(self.hyper_param.model_dump(), outfile, default_flow_style=False)

        tensorboard_dir = (
            Path(self.hyper_param.logdir)
            / "tensorboard"
            / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.writer = SummaryWriter(tensorboard_dir)
        # for makegrid nrows
        self.nrow = self.hyper_param.batch_size // 2
        self.writer.add_graph(
            self.model,
            (
                torch.randn(1, 3, *self.hyper_param.image_shape).to(self.device),
                torch.randn(1, 3, *self.hyper_param.image_shape).to(self.device),
            ),
        )

        ckpt_dir = Path(self.hyper_param.logdir) / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_files = sorted(list(ckpt_dir.glob("model_step_*.pt")))[
            -self.hyper_param.max_ckpts :
        ]

        last_ckpt = ckpt_dir / "last.pt"
        if last_ckpt.exists():
            self.load_ckpts(last_ckpt)
        _zfill = len(str(self.hyper_param.num_iteration))

        self.model.train(True)
        while self.step < self.hyper_param.num_iteration:
            self.optimizer.zero_grad()
            train_contents = next(self.content_train_dataloader).to(self.device)
            train_styles = next(self.style_train_dataloader).to(self.device)

            # Train Step
            start_time = datetime.datetime.now()
            (
                train_styled_content,
                loss,
                content_loss,
                style_loss,
            ) = self._step(contents=train_contents, styles=train_styles)

            loss.backward()

            duration = datetime.datetime.now() - start_time

            self.optimizer_step()

            self.step += 1

            if self.step % self.hyper_param.summary_step == 0:
                self.model.eval()
                with torch.no_grad():
                    test_contents = next(self.content_test_dataloader).to(self.device)
                    test_styles = next(self.style_test_dataloader).to(self.device)
                    (
                        test_styled_content,
                        test_loss,
                        test_content_loss,
                        test_style_loss,
                    ) = self._step(contents=test_contents, styles=test_styles)
                self.model.train(True)

                # Write Summary
                self.write_summary(
                    loss=loss,
                    style_loss=style_loss,
                    content_loss=content_loss,
                    contents=train_contents,
                    styles=train_styles,
                    styled_content=train_styled_content,
                    prefix="train",
                )

                self.write_summary(
                    loss=test_loss,
                    style_loss=test_style_loss,
                    content_loss=test_content_loss,
                    contents=test_contents,
                    styles=test_styles,
                    styled_content=test_styled_content,
                    prefix="test",
                )

            if self.step % self.hyper_param.save_step == 0:
                current_ckpt = (
                    ckpt_dir / f"model_step_{str(self.step).zfill(_zfill)}.pt"
                )
                self.save_ckpts(current_ckpt)
                ckpt_files.append(current_ckpt)
                self.save_ckpts(last_ckpt)
                if len(ckpt_files) > self.hyper_param.max_ckpts:
                    old_ckpt = ckpt_files.pop(0)
                    old_ckpt.unlink(missing_ok=True)

            if self.step % self.hyper_param.log_step == 0:
                self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], self.step)
                print(
                    f"{datetime.datetime.now()} "
                    f"step {self.step}, "
                    f"loss = {loss:.4f}, "
                    f"style_loss = {style_loss:.4f}, "
                    f"content_loss = {content_loss:.4f}, "
                    f"{(self.hyper_param.batch_size / duration.total_seconds()):.4f}  examples/sec, "
                    f"{duration.total_seconds():.4f} sec/batch "
                )

        self.writer.close()
        print("Training Done.")

    def write_summary(
        self,
        loss,
        style_loss,
        content_loss,
        contents,
        styles,
        styled_content,
        prefix="",
    ):
        self.writer.add_scalar(f"{prefix}_loss", loss, self.step)
        self.writer.add_scalar(f"{prefix}_style_loss", style_loss, self.step)
        self.writer.add_scalar(f"{prefix}_content_loss", content_loss, self.step)

        self.writer.add_image(
            f"{prefix}_content_images",
            make_grid(contents, nrow=self.nrow),
            self.step,
        )
        self.writer.add_image(
            f"{prefix}_style_images",
            make_grid(styles, nrow=self.nrow),
            self.step,
        )
        self.writer.add_image(
            f"{prefix}_styled_content_images",
            make_grid(styled_content, nrow=self.nrow),
            self.step,
        )

    def _step(
        self, contents: torch.Tensor, styles: torch.Tensor
    ) -> tuple[torch.Tensor]:
        x, content_feats, style_feats, x_feats = self.model(
            contents, styles, return_features=True
        )
        content_loss = self.content_loss_fn(content_feats[-1], x_feats[-1])
        style_loss = self.style_loss_fn(style_feats, x_feats)
        loss = content_loss + (style_loss * self.hyper_param.style_weight)
        return x, loss, content_loss, style_loss
