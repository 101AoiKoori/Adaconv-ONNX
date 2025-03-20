import datetime
import os
from pathlib import Path
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from trainers.data_manager import DataManager
from trainers.model_manager import ModelManager
from trainers.visualization_manager import VisualizationManager
from hyperparam.hyperparam import Hyperparameter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class Trainer:
    """Main trainer class for style transfer model"""
    
    def __init__(self, hyper_param: Hyperparameter):
        self.hyper_param = hyper_param
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_manager = ModelManager(hyper_param)
        self.data_manager = DataManager(hyper_param)
        print(f"Training Initialized -> device: {self.device}")

    def train(self):
        # 初始化日志和检查点目录
        Path(self.hyper_param.logdir).mkdir(parents=True, exist_ok=True)
        with (Path(self.hyper_param.logdir) / "config.yaml").open("w") as outfile:
            yaml.dump(self.hyper_param.model_dump(), outfile, default_flow_style=False)

        # 初始化TensorBoard
        tensorboard_dir = Path(self.hyper_param.logdir) / "tensorboard" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(tensorboard_dir)
        self.vis_manager = VisualizationManager(writer, self.hyper_param.batch_size)

        # 添加模型结构和参数分布图
        batch_size_for_graph = self.hyper_param.fixed_batch_size or self.hyper_param.batch_size
        writer.add_graph(
            self.model_manager.model,
            (
                torch.randn(batch_size_for_graph, 3, *self.hyper_param.image_shape).to(self.device),
                torch.randn(batch_size_for_graph, 3, *self.hyper_param.image_shape).to(self.device),
            ),
        )
        model_structure_fig = self.vis_manager.visualize_model_structure(self.model_manager.get_model_info())
        self.vis_manager.add_figure_to_tensorboard('Model/Structure', model_structure_fig, 0)
        self.vis_manager.write_hyperparameters(self.hyper_param.model_dump())
        param_dist_fig = self.vis_manager.visualize_parameters_distribution(self.model_manager.named_parameters())
        self.vis_manager.add_figure_to_tensorboard('Model/Parameter_Distribution', param_dist_fig, 0)

        # 检查点设置
        ckpt_dir = Path(self.hyper_param.logdir) / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        last_ckpt = ckpt_dir / "last.pt"
        if last_ckpt.exists():
            self.model_manager.load_checkpoint(last_ckpt)
        _zfill = len(str(self.hyper_param.num_iteration))

        # 主训练循环
        training_start_time = datetime.datetime.now()
        batch_times = []
        total_examples = 0
        self.model_manager.set_train(True)
        
        while self.model_manager.step < self.hyper_param.num_iteration:
            train_contents, train_styles = self.data_manager.get_batch(is_training=True)
            
            # 训练步骤
            start_time = datetime.datetime.now()
            self.model_manager.optimizer.zero_grad()
            (train_styled_content, loss, content_loss, style_loss, *_) = self.model_manager.forward(
                contents=train_contents, styles=train_styles, return_features=True
            )
            loss.backward()
            
            # 记录梯度可视化
            if self.model_manager.step % self.hyper_param.log_step == 0:
                grad_fig = self.vis_manager.visualize_gradients(self.model_manager.named_parameters())
                self.vis_manager.add_figure_to_tensorboard('Training/Gradients', grad_fig, self.model_manager.step)
                self.vis_manager.log_system_metrics(self.model_manager.step, self.device)

            self.model_manager.optimizer_step()
            
            # 记录性能指标和图像
            if self.model_manager.step % self.hyper_param.summary_step == 0:
                self._run_evaluation()  # 评估测试集
                
                # 记录训练指标和图像（使用当前训练batch的数据）
                self.vis_manager.write_training_metrics_and_images(
                    train_loss=loss,
                    train_style_loss=style_loss,
                    train_content_loss=content_loss,
                    train_contents=train_contents,
                    train_styles=train_styles,
                    train_styled_content=train_styled_content,
                    step=self.model_manager.step
                )
                
                # 计算性能指标
                avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                examples_per_sec = self.hyper_param.batch_size / avg_batch_time if avg_batch_time > 0 else 0
                self.vis_manager.log_performance_metrics(avg_batch_time, examples_per_sec, 
                                                       self.model_manager.step / self.hyper_param.num_iteration,
                                                       self.model_manager.step)
                batch_times = batch_times[-100:]

            # 保存检查点
            if self.model_manager.step % self.hyper_param.save_step == 0:
                current_ckpt = ckpt_dir / f"model_step_{str(self.model_manager.step).zfill(_zfill)}.pt"
                self.model_manager.save_checkpoint(current_ckpt)
                self.model_manager.save_checkpoint(last_ckpt)

            # 打印日志
            if self.model_manager.step % self.hyper_param.log_step == 0:
                current_lr = self.model_manager.get_lr()
                self.vis_manager.writer.add_scalar("Training/Learning_Rate", current_lr, self.model_manager.step)
                examples_per_sec = self.hyper_param.batch_size / (datetime.datetime.now() - start_time).total_seconds()
                print(
                    f"{datetime.datetime.now()} step {self.model_manager.step}/{self.hyper_param.num_iteration} "
                    f"({(self.model_manager.step/self.hyper_param.num_iteration*100):.1f}%), "
                    f"loss={loss:.4f}, style_loss={style_loss:.4f}, content_loss={content_loss:.4f}, "
                    f"{examples_per_sec:.2f} ex/s"
                )

        self.vis_manager.writer.close()
        print("Training Done.")

    def _run_evaluation(self):
        """评估测试集并记录结果"""
        self.model_manager.set_train(False)
        with torch.no_grad():
            test_contents, test_styles = self.data_manager.get_batch(is_training=False)
            (test_styled_content, test_loss, test_content_loss, test_style_loss, *feats) = \
                self.model_manager.forward(contents=test_contents, styles=test_styles, return_features=True)
            
            # 记录测试集指标和图像
            self.vis_manager.write_metrics(
                {"loss": test_loss, "style_loss": test_style_loss, "content_loss": test_content_loss},
                self.model_manager.step,
                prefix="test"
            )
            self.vis_manager.write_images(
                {
                    "content_images": test_contents,
                    "style_images": test_styles,
                    "styled_content_images": test_styled_content
                },
                self.model_manager.step,
                prefix="test"
            )
            
            # 计算SSIM和PSNR
            self.vis_manager.calculate_ssim_psnr(test_contents, test_styled_content, self.model_manager.step)
            
        self.model_manager.set_train(True)