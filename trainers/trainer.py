import datetime
import os
from pathlib import Path
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
import json

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
        self.training_state = {
            "completed": False,
            "current_step": 0,
            "target_steps": self.hyper_param.num_iteration,
            "fine_tuning": False,
            "history": [],
            "max_history": self.hyper_param.max_ckpts if hasattr(self.hyper_param, 'max_ckpts') else 3
        }
        print(f"Training Initialized -> device: {self.device}")

    def _load_training_state(self, state_path):
        """加载训练状态"""
        if Path(state_path).exists():
            with open(state_path, 'r') as f:
                self.training_state = json.load(f)
            return True
        return False

    def _save_training_state(self, state_path):
        """保存训练状态"""
        with open(state_path, 'w') as f:
            json.dump(self.training_state, f, indent=4)

    def train(self, fine_tuning=False):
        """训练模型

        Args:
            fine_tuning (bool): 是否为微调模式
        """
        # 初始化日志和检查点目录
        Path(self.hyper_param.logdir).mkdir(parents=True, exist_ok=True)
        state_path = Path(self.hyper_param.logdir) / "training_state.json"
        
        # 保存训练配置
        with (Path(self.hyper_param.logdir) / "config.yaml").open("w") as outfile:
            yaml.dump(self.hyper_param.model_dump(), outfile, default_flow_style=False)
            
        # 尝试加载已有训练状态
        state_loaded = self._load_training_state(state_path)
        
        # 处理微调模式
        if fine_tuning:
            if state_loaded and self.training_state["completed"]:
                print(f"Starting fine-tuning from completed model")
                self.training_state["fine_tuning"] = True
                self.training_state["current_step"] = 0
                self.training_state["target_steps"] = self.hyper_param.num_iteration
                self.training_state["completed"] = False
            elif state_loaded and not self.training_state["completed"]:
                print(f"Cannot start fine-tuning: previous training not completed")
                return
            else:
                print(f"No previous training state found, starting new training instead of fine-tuning")
                self.training_state["fine_tuning"] = True
                self.training_state["current_step"] = 0
                self.training_state["target_steps"] = self.hyper_param.num_iteration
                self.training_state["completed"] = False
        else:
            if state_loaded:
                if self.training_state["completed"]:
                    print(f"Training already completed. Use fine_tuning=True for fine-tuning.")
                    return
                print(f"Resuming training from step {self.training_state['current_step']}")
            else:
                self.training_state["current_step"] = 0
                self.training_state["target_steps"] = self.hyper_param.num_iteration
                self.training_state["fine_tuning"] = False
        
        # 初始化TensorBoard
        tensorboard_suffix = "fine-tuning" if self.training_state["fine_tuning"] else "training"
        tensorboard_dir = Path(self.hyper_param.logdir) / "tensorboard" / f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{tensorboard_suffix}"
        writer = SummaryWriter(tensorboard_dir)
        self.vis_manager = VisualizationManager(writer, self.hyper_param.batch_size)

        # 检查点设置
        ckpt_dir = Path(self.hyper_param.logdir) / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        last_ckpt = ckpt_dir / "last.pt"
        
        # 处理检查点加载逻辑
        if self.hyper_param.checkpoint_path and Path(self.hyper_param.checkpoint_path).exists():
            # 如果指定了checkpoint_path，优先使用它
            checkpoint_path = Path(self.hyper_param.checkpoint_path)
            if self.training_state["fine_tuning"]:
                print(f"Loading checkpoint {checkpoint_path} for fine-tuning")
                self._load_for_fine_tuning(checkpoint_path)
            else:
                print(f"Loading checkpoint {checkpoint_path} for resuming training")
                self.model_manager.load_checkpoint(checkpoint_path)
                # 确保step一致
                self.training_state["current_step"] = self.model_manager.step
        # 微调模式下，如果存在last.pt，加载它但重置优化器和scheduler
        elif self.training_state["fine_tuning"] and last_ckpt.exists():
            print("Loading last checkpoint for fine-tuning but resetting optimizer and scheduler")
            self._load_for_fine_tuning(last_ckpt)
        # 正常训练模式下，如果存在last.pt，恢复训练状态
        elif last_ckpt.exists() and not self.training_state["fine_tuning"]:
            print(f"Loading last checkpoint for resuming normal training")
            self.model_manager.load_checkpoint(last_ckpt)
            # 确保step一致
            self.training_state["current_step"] = self.model_manager.step
        
        # 如果是微调模式且设置了专用微调学习率，则使用它
        if self.training_state["fine_tuning"] and hasattr(self.hyper_param, 'finetune_lr') and self.hyper_param.finetune_lr is not None:
            self.model_manager.update_learning_rate(self.hyper_param.finetune_lr)
            print(f"Updated learning rate for fine-tuning: {self.hyper_param.finetune_lr}")
        
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

        # 主训练循环
        _zfill = len(str(self.hyper_param.num_iteration))
        training_start_time = datetime.datetime.now()
        batch_times = []
        self.model_manager.set_train(True)
        
        while self.training_state["current_step"] < self.training_state["target_steps"]:
            train_contents, train_styles = self.data_manager.get_batch(is_training=True)
            
            # 训练步骤
            start_time = datetime.datetime.now()
            self.model_manager.optimizer.zero_grad()
            (train_styled_content, loss, content_loss, style_loss, *_) = self.model_manager.forward(
                contents=train_contents, styles=train_styles, return_features=True
            )
            loss.backward()

            duration_seconds = (datetime.datetime.now() - start_time).total_seconds()
            batch_times.append(duration_seconds)

            # 记录梯度可视化
            if self.training_state["current_step"] % self.hyper_param.log_step == 0:
                grad_fig = self.vis_manager.visualize_gradients(self.model_manager.named_parameters())
                self.vis_manager.add_figure_to_tensorboard('Training/Gradients', grad_fig, self.training_state["current_step"])
                self.vis_manager.log_system_metrics(self.training_state["current_step"], self.device)

            # 更新模型参数
            self.model_manager.optimizer_step()
            self.training_state["current_step"] = self.model_manager.step
            
            # 记录性能指标和图像
            if self.training_state["current_step"] % self.hyper_param.summary_step == 0:
                self._run_evaluation()  # 评估测试集
                
                # 记录训练指标和图像（使用当前训练batch的数据）
                self.vis_manager.write_training_metrics_and_images(
                    train_loss=loss,
                    train_style_loss=style_loss,
                    train_content_loss=content_loss,
                    train_contents=train_contents,
                    train_styles=train_styles,
                    train_styled_content=train_styled_content,
                    step=self.training_state["current_step"]
                )
                
                # 计算性能指标
                avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                examples_per_sec = self.hyper_param.batch_size / avg_batch_time if avg_batch_time > 0 else 0
                current_examples_per_sec = self.hyper_param.batch_size / duration_seconds
                self.vis_manager.log_performance_metrics(
                    current_batch_time=duration_seconds,
                    current_examples_per_sec=current_examples_per_sec,
                    batch_time=avg_batch_time,
                    examples_per_sec=examples_per_sec,
                    progress=self.training_state["current_step"] / self.training_state["target_steps"],
                    step=self.training_state["current_step"]
                )
                batch_times = batch_times[-100:]

            # 保存检查点
            if self.training_state["current_step"] % self.hyper_param.save_step == 0:
                # 生成当前检查点路径
                checkpoint_prefix = "ft" if self.training_state["fine_tuning"] else "model"
                current_ckpt = ckpt_dir / f"{checkpoint_prefix}_step_{str(self.training_state['current_step']).zfill(_zfill)}.pt"
                
                # 保存当前检查点
                self.model_manager.save_checkpoint(current_ckpt)
                self.model_manager.save_checkpoint(last_ckpt)
                
                # 维护历史检查点列表
                self.training_state["history"].append(str(current_ckpt))
                
                # 删除超出数量的最旧检查点
                if len(self.training_state["history"]) > self.training_state["max_history"]:
                    old_ckpt = Path(self.training_state["history"].pop(0))
                    if old_ckpt.exists():
                        old_ckpt.unlink()
                
                # 保存当前训练状态
                self._save_training_state(state_path)

            # 打印日志
            if self.training_state["current_step"] % self.hyper_param.log_step == 0:
                current_lr = self.model_manager.get_lr()
                self.vis_manager.writer.add_scalar("Training/Learning_Rate", current_lr, self.training_state["current_step"])
                examples_per_sec = self.hyper_param.batch_size / (datetime.datetime.now() - start_time).total_seconds()
                print(
                    f"{datetime.datetime.now()} step {self.training_state['current_step']}/{self.training_state['target_steps']} "
                    f"({(self.training_state['current_step']/self.training_state['target_steps']*100):.1f}%), "
                    f"loss={loss:.4f}, style_loss={style_loss:.4f}, content_loss={content_loss:.4f}, "
                    f"{examples_per_sec:.2f} ex/s"
                )

        # 训练完成
        self.training_state["completed"] = True
        self._save_training_state(state_path)
        self.vis_manager.writer.close()
        
        mode = "Fine-tuning" if self.training_state["fine_tuning"] else "Training"
        print(f"{mode} Done.")

    def _load_for_fine_tuning(self, ckpt_path):
        """为微调加载模型权重但重置优化器状态
        
        Args:
            ckpt_path: 检查点路径
        """
        # 设置权重加载选项以处理潜在的版本差异
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # 仅加载模型权重，不加载优化器和调度器状态
        if "model_state_dict" in checkpoint:
            self.model_manager.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"Warning: 检查点格式不正确，尝试直接加载到模型")
            self.model_manager.model.load_state_dict(checkpoint)
            
        # 重新初始化优化器和调度器
        self.model_manager.setup_optimizer()
        
        # 如果有微调专用学习率，使用它
        if hasattr(self.hyper_param, 'finetune_lr') and self.hyper_param.finetune_lr is not None:
            self.model_manager.update_learning_rate(self.hyper_param.finetune_lr)
            
        # 重置步数
        self.model_manager.step = 0
        print(f"Loaded model weights from {ckpt_path} for fine-tuning (optimizer and scheduler reset)")

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
                self.training_state["current_step"],
                prefix="test"
            )
            self.vis_manager.write_images(
                {
                    "content_images": test_contents,
                    "style_images": test_styles,
                    "styled_content_images": test_styled_content
                },
                self.training_state["current_step"],
                prefix="test"
            )
            
            # 计算SSIM和PSNR
            self.vis_manager.calculate_ssim_psnr(test_contents, test_styled_content, self.training_state["current_step"])
            
        self.model_manager.set_train(True)