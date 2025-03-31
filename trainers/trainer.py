# 文件 2
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
    
    def __init__(self, hyper_param: Hyperparameter, finetune_mode=False, pretrained_model_path=None):
        """
        初始化训练器
        
        Args:
            hyper_param: 超参数配置
            finetune_mode: 是否为微调模式
            pretrained_model_path: 预训练模型路径，用于微调
        """
        self.hyper_param = hyper_param
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finetune_mode = finetune_mode
        self.pretrained_model_path = pretrained_model_path
        
        # 初始化模型管理器，传入微调模式标志
        self.model_manager = ModelManager(hyper_param, finetune_mode=finetune_mode)
        self.data_manager = DataManager(hyper_param)
        
        # 如果是微调模式，保存ckpt历史信息到实例变量（而不是全局变量）
        self.ckpt_history = []
        self.max_ckpt_history = hyper_param.max_ckpts if hasattr(hyper_param, 'max_ckpts') else 3
        
        print(f"Training Initialized -> device: {self.device}, mode: {'Fine-tuning' if finetune_mode else 'Normal training'}")

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
        
        # 保存配置
        config_data = self.hyper_param.model_dump()
        config_data['finetune_mode'] = self.finetune_mode
        if self.pretrained_model_path:
            config_data['pretrained_model_path'] = str(self.pretrained_model_path)
            
        with (Path(self.hyper_param.logdir) / "config.yaml").open("w") as outfile:
            yaml.dump(config_data, outfile, default_flow_style=False)

        # 初始化TensorBoard
        tensorboard_dir = Path(self.hyper_param.logdir) / "tensorboard" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
        self.vis_manager = VisualizationManager(writer, self.hyper_param.batch_size)

        # 检查点设置
        ckpt_dir = Path(self.hyper_param.logdir) / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        last_ckpt = ckpt_dir / "last.pt"
        
        # 如果是微调模式，优先加载last.pt文件
        if self.finetune_mode and last_ckpt.exists():
            # 加载last.pt文件
            ckpt_info = self.model_manager.load_checkpoint(last_ckpt)
            print(f"Resumed fine-tuning from last checkpoint at step {self.model_manager.step}")
        elif self.finetune_mode and self.pretrained_model_path:
            # 加载预训练模型，但重置步数和优化器状态
            ckpt_info = self.model_manager.load_checkpoint(
                self.pretrained_model_path, 
                reset_step=True,
                reset_optimizer=True
            )
            print(f"Loaded pretrained model for fine-tuning from {self.pretrained_model_path}")
            
            # 检查模型是否已经完成了训练
            if ckpt_info["loaded_step"] >= ckpt_info["original_num_iteration"]:
                print("Loaded model was fully trained. Starting fine-tuning from step 0.")
            else:
                print(f"Warning: Loaded model was not fully trained ({ckpt_info['loaded_step']}/{ckpt_info['original_num_iteration']} steps).")
        elif last_ckpt.exists():
            # 如果存在上次的检查点，尝试恢复训练
            ckpt_info = self.model_manager.load_checkpoint(last_ckpt)
            
            # 检查是否微调检查点与当前模式不匹配
            if ckpt_info["is_finetune_ckpt"] != self.finetune_mode:
                print(f"Warning: Checkpoint finetune mode ({ckpt_info['is_finetune_ckpt']}) doesn't match current mode ({self.finetune_mode}).")
                
                if self.finetune_mode:
                    # 如果当前是微调模式但加载的不是微调检查点，重置步数和优化器
                    self.model_manager.reset_for_finetuning()
                    print("Reset model for fine-tuning.")
            else:
                print(f"Resumed {'fine-tuning' if self.finetune_mode else 'training'} from step {self.model_manager.step}")
                
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
        
        # 添加超参数信息
        hyperparams = self.hyper_param.model_dump()
        hyperparams['finetune_mode'] = self.finetune_mode
        if self.pretrained_model_path:
            hyperparams['pretrained_model_path'] = str(self.pretrained_model_path)
        self.vis_manager.write_hyperparameters(hyperparams)
        
        param_dist_fig = self.vis_manager.visualize_parameters_distribution(self.model_manager.named_parameters())
        self.vis_manager.add_figure_to_tensorboard('Model/Parameter_Distribution', param_dist_fig, 0)

        _zfill = len(str(self.hyper_param.num_iteration))

        # 主训练循环
        _zfill = len(str(self.hyper_param.num_iteration))
        training_start_time = datetime.datetime.now()
        batch_times = []
        self.model_manager.set_train(True)
        
        # 设置训练前缀（用于日志和检查点命名）
        mode_prefix = "finetune_" if self.finetune_mode else ""
        
        while self.model_manager.step < self.hyper_param.num_iteration:
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
                self.vis_manager.add_figure_to_tensorboard(f'{mode_prefix}Training/Gradients', grad_fig, self.model_manager.step)
                self.vis_manager.log_system_metrics(self.model_manager.step, self.device)

            # 更新模型参数
            self.model_manager.optimizer_step()
            self.training_state["current_step"] = self.model_manager.step
            
            # 记录性能指标和图像
            if self.model_manager.step % self.hyper_param.summary_step == 0:
                self._run_evaluation(mode_prefix=mode_prefix)  # 评估测试集
                
                # 记录训练指标和图像（使用当前训练batch的数据）
                self.vis_manager.write_training_metrics_and_images(
                    train_loss=loss,
                    train_style_loss=style_loss,
                    train_content_loss=content_loss,
                    train_contents=train_contents,
                    train_styles=train_styles,
                    train_styled_content=train_styled_content,
                    step=self.model_manager.step,
                    prefix=mode_prefix
                )
                
                # 计算性能指标
                avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                examples_per_sec = self.hyper_param.batch_size / avg_batch_time if avg_batch_time > 0 else 0
                current_examples_per_sec = self.hyper_param.batch_size / duration_seconds
                self.vis_manager.log_performance_metrics(
                    current_batch_time=duration_seconds,
                    current_examples_per_sec=current_examples_per_sec,
                    batch_time=avg_batch_time if 'avg_batch_time' in locals() else 0,
                    examples_per_sec=examples_per_sec if 'examples_per_sec' in locals() else 0,
                    progress=self.model_manager.step / self.hyper_param.num_iteration,
                    step=self.model_manager.step,
                    prefix=mode_prefix
                )
                batch_times = batch_times[-100:]

            # 保存检查点
            if self.training_state["current_step"] % self.hyper_param.save_step == 0:
                # 生成当前检查点路径
                current_ckpt = ckpt_dir / f"{mode_prefix}model_step_{str(self.model_manager.step).zfill(_zfill)}.pt"
                
                # 保存当前检查点
                self.model_manager.save_checkpoint(current_ckpt, is_finetune_ckpt=self.finetune_mode)
                self.model_manager.save_checkpoint(last_ckpt, is_finetune_ckpt=self.finetune_mode)
                
                # 维护历史检查点列表
                self.ckpt_history.append(current_ckpt)
                
                # 删除超出数量的最旧检查点
                if len(self.ckpt_history) > self.max_ckpt_history:
                    old_ckpt = self.ckpt_history.pop(0)
                    if old_ckpt.exists():
                        old_ckpt.unlink()

            # 打印日志
            if self.training_state["current_step"] % self.hyper_param.log_step == 0:
                current_lr = self.model_manager.get_lr()
                self.vis_manager.writer.add_scalar(f"{mode_prefix}Training/Learning_Rate", current_lr, self.model_manager.step)
                examples_per_sec = self.hyper_param.batch_size / (datetime.datetime.now() - start_time).total_seconds()
                print(
                    f"{datetime.datetime.now()} {'[FINETUNE] ' if self.finetune_mode else ''}step {self.model_manager.step}/{self.hyper_param.num_iteration} "
                    f"({(self.model_manager.step/self.hyper_param.num_iteration*100):.1f}%), "
                    f"loss={loss:.4f}, style_loss={style_loss:.4f}, content_loss={content_loss:.4f}, "
                    f"{examples_per_sec:.2f} ex/s"
                )

        # 训练完成
        self.training_state["completed"] = True
        self._save_training_state(state_path)
        self.vis_manager.writer.close()
        print(f"{'Fine-tuning' if self.finetune_mode else 'Training'} Done.")

    def _run_evaluation(self, mode_prefix=""):
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
                prefix=f"{mode_prefix}test"
            )
            self.vis_manager.write_images(
                {
                    "content_images": test_contents,
                    "style_images": test_styles,
                    "styled_content_images": test_styled_content
                },
                self.model_manager.step,
                prefix=f"{mode_prefix}test"
            )
            
            # 计算SSIM和PSNR
            self.vis_manager.calculate_ssim_psnr(test_contents, test_styled_content, self.model_manager.step, prefix=mode_prefix)
            
        self.model_manager.set_train(True)