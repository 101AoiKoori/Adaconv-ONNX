import torch
from pathlib import Path
from models.model import StyleTransfer
from losses.loss import MomentMatchingStyleLoss, MSEContentLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from hyperparam.hyperparam import Hyperparameter

class ModelManager:
    """
    Manages the model, optimizers, losses, and checkpoints
    
    This class is responsible for:
    - Creating and configuring the model
    - Setting up loss functions
    - Creating and managing optimizers
    - Handling checkpoint saving and loading
    - Running inference
    """
    
    def __init__(self, hyper_param: Hyperparameter, finetune_mode=False):
        """
        Initialize the ModelManager
        
        Args:
            hyper_param: Hyperparameter configuration
            finetune_mode: 是否为微调模式
        """
        self.hyper_param = hyper_param
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.step = 0
        self.finetune_mode = finetune_mode
        self.setup_model()
        self.setup_losses()
        self.setup_optimizer()
        
    def setup_model(self):
        """Initialize and configure the style transfer model"""
        # Process image size parameters
        if isinstance(self.hyper_param.image_size, int):
            self.image_shape = (self.hyper_param.image_size, self.hyper_param.image_size)
        else:
            self.image_shape = self.hyper_param.image_shape

        # Process groups parameter
        groups = self.hyper_param.groups
        if groups is None and self.hyper_param.groups_list is not None:
            groups = self.hyper_param.groups_list
        elif groups is None and self.hyper_param.groups_ratios is not None:
            # Calculate groups based on ratios
            channels = [512, 256, 128, 64]
            groups = [max(1, int(c * ratio)) for c, ratio in zip(channels, self.hyper_param.groups_ratios)]

        if self.hyper_param.fixed_batch_size is None:
            self.hyper_param.fixed_batch_size = self.hyper_param.batch_size  # Use training batch size

        # Save these computed parameters for visualization
        self.groups = groups

        # Create the model
        self.model = StyleTransfer(
            image_shape=self.image_shape,
            style_dim=self.hyper_param.style_dim,
            style_kernel=self.hyper_param.style_kernel,
            groups=groups,
            fixed_batch_size=self.hyper_param.fixed_batch_size,
            use_fixed_size=self.hyper_param.use_fixed_size
        ).to(self.device)
        
        # Count model parameters
        self.model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
    def setup_losses(self):
        """Initialize loss functions"""
        self.content_loss_fn = MSEContentLoss()
        self.style_loss_fn = MomentMatchingStyleLoss()
        
    def setup_optimizer(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = Adam(
            self.model.parameters(), lr=self.hyper_param.learning_rate
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.hyper_param.learning_rate,
            total_steps=self.hyper_param.num_iteration,
        )
        
    def get_model_info(self):
        """
        Get model structure information
        
        Returns:
            dict: Dictionary with model structure information
        """
        return {
            "Image Shape": str(self.image_shape),
            "Style Dim": self.hyper_param.style_dim,
            "Style Kernel": self.hyper_param.style_kernel,
            "Groups": str(self.groups),
            "Batch Size": self.hyper_param.batch_size,
            "Fixed Batch Size": self.hyper_param.fixed_batch_size,
            "Use Fixed Size": self.hyper_param.use_fixed_size,
            "Total Parameters": f"{self.model_params:,}",
            "Finetune Mode": self.finetune_mode,
        }
    
    def save_checkpoint(self, ckpt_path, is_finetune_ckpt=False):
        """
        Save model checkpoint
        
        Args:
            ckpt_path: Path to save checkpoint
            is_finetune_ckpt: 是否为微调模式的检查点
        """
        torch.save(
            {
                "steps": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "is_finetune_ckpt": is_finetune_ckpt or self.finetune_mode,
                "original_num_iteration": self.hyper_param.num_iteration,
            },
            ckpt_path,
        )
        print(f"Saving {'finetune ' if is_finetune_ckpt or self.finetune_mode else ''}checkpoint to {ckpt_path} at step {self.step}")

    def load_checkpoint(self, ckpt_path, reset_step=False, reset_optimizer=False):
        """
        Load model checkpoint
        
        Args:
            ckpt_path: Path to checkpoint
            reset_step: 是否重置步数（用于微调）
            reset_optimizer: 是否重置优化器状态（用于微调）
        """
        checkpoint = torch.load(ckpt_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if reset_optimizer:
            # 重新初始化优化器，以便使用新的学习率
            self.setup_optimizer()
            print(f"Reset optimizer with new learning rate: {self.hyper_param.learning_rate}")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if reset_step:
            # 微调模式下，重置步数为0
            self.step = 0
            print("Reset step counter to 0 for fine-tuning")
        else:
            self.step = checkpoint["steps"]
        
        # 检查是否为微调检查点
        is_finetune = checkpoint.get("is_finetune_ckpt", False)
        original_num_iteration = checkpoint.get("original_num_iteration", self.hyper_param.num_iteration)
        
        print(f"Loaded checkpoint from {ckpt_path}")
        print(f"Checkpoint info: step={self.step}, is_finetune={is_finetune}, original_num_iteration={original_num_iteration}")
        
        return {
            "is_finetune_ckpt": is_finetune,
            "original_num_iteration": original_num_iteration,
            "loaded_step": checkpoint["steps"]
        }
        
    def forward(self, contents, styles, return_features=False):
        """
        Forward pass through the model
        
        Args:
            contents: Content images tensor
            styles: Style images tensor
            return_features: Whether to return intermediate features
        
        Returns:
            tuple: Results from forward pass depending on return_features flag
        """
        if return_features:
            x, content_feats, style_feats, x_feats = self.model.forward_with_features(contents, styles)
            content_loss = self.content_loss_fn(content_feats[-1], x_feats[-1])
            style_loss = self.style_loss_fn(style_feats, x_feats)
            loss = content_loss + (style_loss * self.hyper_param.style_weight)
            return x, loss, content_loss, style_loss, content_feats, style_feats, x_feats
        else:
            x, content_feats, style_feats, x_feats = self.model.forward_with_features(contents, styles)
            content_loss = self.content_loss_fn(content_feats[-1], x_feats[-1])
            style_loss = self.style_loss_fn(style_feats, x_feats)
            loss = content_loss + (style_loss * self.hyper_param.style_weight)
            return x, loss, content_loss, style_loss
    
    def reset_for_finetuning(self, new_lr=None):
        """
        重置优化器和学习率调度器，准备进行微调
        
        Args:
            new_lr: 微调时使用的新学习率，如果为None则使用配置中的值
        """
        if new_lr is not None:
            self.hyper_param.learning_rate = new_lr
            
        # 重置步数
        self.step = 0
        
        # 重新初始化优化器和调度器
        self.setup_optimizer()
        
        # 设置微调模式
        self.finetune_mode = True
        
        print(f"Model reset for fine-tuning with learning rate {self.hyper_param.learning_rate}")
    
    def optimizer_step(self):
        """Perform optimization step with gradient clipping"""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        
    def get_lr(self):
        """
        Get current learning rate
        
        Returns:
            float: Current learning rate
        """
        return self.scheduler.get_last_lr()[0]
        
    def named_parameters(self):
        """
        Get named parameters from the model
        
        Returns:
            iterator: Named parameters
        """
        return [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        
    def set_train(self, mode=True):
        """
        Set model training mode
        
        Args:
            mode: True for train mode, False for eval mode
        """
        self.model.train(mode)