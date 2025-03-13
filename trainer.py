import datetime
import os
from pathlib import Path
import json

import torch
import yaml
import numpy as np
from dataloader import ImageDataset, InfiniteDataLoader, get_transform
from hyperparam import Hyperparameter
from loss import MomentMatchingStyleLoss, MSEContentLoss
from model import StyleTransfer
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class Trainer:
    def __init__(self, hyper_param: Hyperparameter):
        self.hyper_param = hyper_param
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup()
        print(f"Training Initialized -> device: {self.device}")

    def setup(self):
        # 处理图像尺寸参数
        if isinstance(self.hyper_param.image_size, int):
            image_shape = (self.hyper_param.image_size, self.hyper_param.image_size)
        else:
            image_shape = self.hyper_param.image_shape
        
        # 处理 groups 参数
        groups = self.hyper_param.groups
        if groups is None and self.hyper_param.groups_list is not None:
            groups = self.hyper_param.groups_list
        elif groups is None and self.hyper_param.groups_ratios is not None:
            # 基于比例计算分组
            channels = [512, 256, 128, 64]
            groups = [max(1, int(c * ratio)) for c, ratio in zip(channels, self.hyper_param.groups_ratios)]

        if self.hyper_param.fixed_batch_size is None:
            self.hyper_param.fixed_batch_size = self.hyper_param.batch_size  # 使用训练批大小
        
        # 保存这些计算出的参数，便于可视化
        self.image_shape = image_shape
        self.groups = groups
            
        self.model = StyleTransfer(
            image_shape=image_shape,
            style_dim=self.hyper_param.style_dim,
            style_kernel=self.hyper_param.style_kernel,
            groups=groups,
            fixed_batch_size=self.hyper_param.fixed_batch_size,
            use_fixed_size=self.hyper_param.use_fixed_size
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
        
        # 统计模型参数数量
        self.model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

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
        
    def visualize_model_structure(self):
        """创建模型结构的可视化图表"""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # 模型结构信息
        model_info = {
            "Image Shape": str(self.image_shape),
            "Style Dim": self.hyper_param.style_dim,
            "Style Kernel": self.hyper_param.style_kernel,
            "Groups": str(self.groups),
            "Batch Size": self.hyper_param.batch_size,
            "Fixed Batch Size": self.hyper_param.fixed_batch_size,
            "Use Fixed Size": self.hyper_param.use_fixed_size,
            "Total Parameters": f"{self.model_params:,}",
        }
        
        # 创建表格
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=[[k, v] for k, v in model_info.items()],
            colLabels=["Parameter", "Value"],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        fig.tight_layout()
        return fig
        
    def visualize_parameters_distribution(self):
        """可视化模型参数分布"""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # 收集参数直方图数据
        param_histograms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 截断名称，保留最后几个组件
                short_name = '/'.join(name.split('.')[-2:])
                param_histograms[short_name] = param.data.cpu().flatten().numpy()
        
        # 绘制前10个主要参数的直方图
        for i, (name, values) in enumerate(list(param_histograms.items())[:10]):
            ax.hist(values, alpha=0.5, bins=50, label=name)
        
        ax.set_title("Parameter Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        return fig
        
    def visualize_gradients(self):
        """可视化当前梯度分布"""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # 收集梯度数据
        grad_norms = []
        layer_names = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # 计算每层梯度的范数
                norm = param.grad.norm().item()
                grad_norms.append(norm)
                # 截断名称，保留最后几个组件
                short_name = '/'.join(name.split('.')[-2:])
                layer_names.append(short_name)
        
        # 只显示前15个层的梯度
        if len(grad_norms) > 15:
            indices = np.argsort(grad_norms)[-15:]
            grad_norms = [grad_norms[i] for i in indices]
            layer_names = [layer_names[i] for i in indices]
        
        # 绘制条形图
        y_pos = range(len(layer_names))
        ax.barh(y_pos, grad_norms, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layer_names)
        ax.invert_yaxis()  # 标签从顶部开始
        ax.set_xlabel('Gradient Norm')
        ax.set_title('Gradient Norms by Layer')
        
        fig.tight_layout()
        return fig
        
    def create_feature_maps_visualization(self, feature_maps):
        """创建特征图可视化"""
        # 特征图的可视化在风格迁移中很重要
        if not feature_maps:
            return None
            
        # 选择最后一个特征图进行可视化
        feature_map = feature_maps[-1]
        
        # 只取第一个批次的数据和少量通道
        batch_idx = 0
        num_channels = min(16, feature_map.shape[1])
        
        # 创建图像
        fig = Figure(figsize=(10, 10))
        
        for i in range(num_channels):
            ax = fig.add_subplot(4, 4, i+1)
            # 提取单个通道并转换为CPU
            channel_data = feature_map[batch_idx, i].detach().cpu().numpy()
            
            # 绘制特征图
            im = ax.imshow(channel_data, cmap='viridis')
            ax.set_title(f'Channel {i}')
            ax.axis('off')
        
        fig.tight_layout()
        return fig
        
    def plot_to_image(self, figure):
        """将matplotlib图像转换为图像张量以供TensorBoard使用"""
        import io
        import numpy as np
        from PIL import Image

        # 保存图像到内存缓冲区
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # 将PIL图像转换为张量
        image = Image.open(buf)
        image = np.array(image)
        
        # 转换为RGB
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)
        
        # 转换为CHW格式，适合PyTorch
        image = image.transpose((2, 0, 1))
        
        return torch.from_numpy(image)

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
        
        # 添加这个检查，如果fixed_batch_size为None，则使用batch_size
        batch_size_for_graph = self.hyper_param.fixed_batch_size if self.hyper_param.fixed_batch_size is not None else self.hyper_param.batch_size
        
        self.writer.add_graph(
            self.model,
            (
                torch.randn(batch_size_for_graph, 3, *self.hyper_param.image_shape).to(self.device),
                torch.randn(batch_size_for_graph, 3, *self.hyper_param.image_shape).to(self.device),
            ),
        )
        
        # 添加模型结构可视化
        model_structure_fig = self.visualize_model_structure()
        model_structure_img = self.plot_to_image(model_structure_fig)
        self.writer.add_image('Model/Structure', model_structure_img, 0)
        
        # 添加超参数配置到TensorBoard
        hyper_params = {k: str(v) for k, v in self.hyper_param.model_dump().items()}
        self.writer.add_text('Hyperparameters', json.dumps(hyper_params, indent=4), 0)
        
        # 添加参数分布可视化
        param_dist_fig = self.visualize_parameters_distribution()
        param_dist_img = self.plot_to_image(param_dist_fig)
        self.writer.add_image('Model/Parameter_Distribution', param_dist_img, 0)

        ckpt_dir = Path(self.hyper_param.logdir) / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_files = sorted(list(ckpt_dir.glob("model_step_*.pt")))[
            -self.hyper_param.max_ckpts :
        ]

        last_ckpt = ckpt_dir / "last.pt"
        if last_ckpt.exists():
            self.load_ckpts(last_ckpt)
        _zfill = len(str(self.hyper_param.num_iteration))

        # 记录训练开始时间
        training_start_time = datetime.datetime.now()
        
        # 初始化性能指标
        batch_times = []
        total_examples = 0
        
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
                content_feats,
                style_feats,
                styled_feats
            ) = self._step(contents=train_contents, styles=train_styles, return_features=True)

            loss.backward()

            duration = datetime.datetime.now() - start_time
            duration_seconds = duration.total_seconds()
            batch_times.append(duration_seconds)
            total_examples += self.hyper_param.batch_size

            # 添加梯度可视化
            if self.step % self.hyper_param.log_step == 0:
                grad_fig = self.visualize_gradients()
                grad_img = self.plot_to_image(grad_fig)
                self.writer.add_image('Training/Gradients', grad_img, self.step)
                
                # 添加GPU使用指标 (如果在CUDA上)
                if self.device == "cuda":
                    self.writer.add_scalar('System/GPU_Memory_Allocated_GB', 
                                          torch.cuda.memory_allocated() / (1024**3), 
                                          self.step)
                    if hasattr(torch.cuda, 'max_memory_allocated'):
                        self.writer.add_scalar('System/GPU_Memory_Peak_GB', 
                                              torch.cuda.max_memory_allocated() / (1024**3),
                                              self.step)

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
                        test_content_feats,
                        test_style_feats,
                        test_styled_feats
                    ) = self._step(contents=test_contents, styles=test_styles, return_features=True)
                    
                    # 添加特征图可视化
                    content_feats_fig = self.create_feature_maps_visualization(test_content_feats)
                    if content_feats_fig:
                        content_feats_img = self.plot_to_image(content_feats_fig)
                        self.writer.add_image('Features/Content', content_feats_img, self.step)
                    
                    style_feats_fig = self.create_feature_maps_visualization(test_style_feats)
                    if style_feats_fig:
                        style_feats_img = self.plot_to_image(style_feats_fig)
                        self.writer.add_image('Features/Style', style_feats_img, self.step)
                    
                    styled_feats_fig = self.create_feature_maps_visualization(test_styled_feats)
                    if styled_feats_fig:
                        styled_feats_img = self.plot_to_image(styled_feats_fig)
                        self.writer.add_image('Features/Styled', styled_feats_img, self.step)
                    
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
                
                # 计算和记录性能指标
                avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                examples_per_sec = self.hyper_param.batch_size / avg_batch_time if avg_batch_time > 0 else 0
                
                # 计算ETA
                elapsed_time = (datetime.datetime.now() - training_start_time).total_seconds()
                progress = self.step / self.hyper_param.num_iteration
                if progress > 0:
                    eta_seconds = (elapsed_time / progress) - elapsed_time
                    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                    
                    self.writer.add_scalar('Training/ETA_Minutes', eta_seconds / 60, self.step)
                    self.writer.add_text('Training/ETA', f"Estimated time remaining: {eta_str}", self.step)
                
                # 添加性能指标
                self.writer.add_scalar('Performance/Batch_Time_Seconds', avg_batch_time, self.step)
                self.writer.add_scalar('Performance/Examples_Per_Second', examples_per_sec, self.step)
                self.writer.add_scalar('Performance/Progress_Percent', progress * 100, self.step)
                
                # 清空批次时间列表，避免内存增长
                batch_times = batch_times[-100:]

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
                # 更多的tensorboard记录
                self.writer.add_scalar("Training/Learning_Rate", self.scheduler.get_last_lr()[0], self.step)
                
                # 训练速度指标
                examples_per_sec = self.hyper_param.batch_size / duration_seconds
                self.writer.add_scalar("Performance/Current_Batch_Time", duration_seconds, self.step)
                self.writer.add_scalar("Performance/Current_Examples_Per_Second", examples_per_sec, self.step)
                
                print(
                    f"{datetime.datetime.now()} "
                    f"step {self.step}/{self.hyper_param.num_iteration} ({(self.step/self.hyper_param.num_iteration*100):.1f}%), "
                    f"loss = {loss:.4f}, "
                    f"style_loss = {style_loss:.4f}, "
                    f"content_loss = {content_loss:.4f}, "
                    f"{examples_per_sec:.2f} examples/sec, "
                    f"{duration_seconds:.4f} sec/batch "
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
        self, contents: torch.Tensor, styles: torch.Tensor, return_features: bool = False
    ) -> tuple:
        x, content_feats, style_feats, x_feats = self.model.forward_with_features(contents, styles)
        content_loss = self.content_loss_fn(content_feats[-1], x_feats[-1])
        style_loss = self.style_loss_fn(style_feats, x_feats)
        loss = content_loss + (style_loss * self.hyper_param.style_weight)
        
        if return_features:
            return x, loss, content_loss, style_loss, content_feats, style_feats, x_feats
        else:
            return x, loss, content_loss, style_loss