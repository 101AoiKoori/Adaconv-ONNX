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
    
    def __init__(self, hyper_param: Hyperparameter, finetune_mode=False, pretrained_model_path=None):
        """
        Initialize the trainer
        
        Args:
            hyper_param: Hyperparameter configuration
            finetune_mode: Whether in fine-tuning mode
            pretrained_model_path: Path to pretrained model for fine-tuning
        """
        self.hyper_param = hyper_param
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.finetune_mode = finetune_mode
        self.pretrained_model_path = pretrained_model_path
        
        # If in fine-tuning mode, apply specific fine-tuning parameters
        if self.finetune_mode:
            self._setup_finetune_params()
        
        # Initialize model manager, passing fine-tuning mode flag
        self.model_manager = ModelManager(hyper_param, finetune_mode=finetune_mode)
        self.data_manager = DataManager(hyper_param)
        
        # Save checkpoint history to instance variable (not global variable)
        self.ckpt_history = []
        self.max_ckpt_history = hyper_param.max_ckpts if hasattr(hyper_param, 'max_ckpts') else 3
        
        print(f"Training Initialized -> device: {self.device}, mode: {'Fine-tuning' if finetune_mode else 'Normal training'}")
        
    def _setup_finetune_params(self):
        """Set up specific parameters for fine-tuning mode"""
        # If there's a predefined fine-tuning learning rate, use it
        if self.hyper_param.finetune_learning_rate is not None:
            self.hyper_param.learning_rate = self.hyper_param.finetune_learning_rate
        elif hasattr(self.hyper_param, 'learning_rate'):
            # Otherwise, use a fraction of the base learning rate
            self.hyper_param.learning_rate *= 0.1
            
        # Modify iteration count (if there's a fine-tuning specific iteration count)
        if self.hyper_param.finetune_iterations is not None:
            self.hyper_param.num_iteration = self.hyper_param.finetune_iterations
        else:
            # Default to using 1/4 of the original iteration count
            self.hyper_param.num_iteration = int(self.hyper_param.num_iteration * 0.25)
            
        print(f"Fine-tuning configuration: LR={self.hyper_param.learning_rate}, "
              f"total iterations={self.hyper_param.num_iteration}")

    def train(self):
        """Execute the training process"""
        # Initialize logging and checkpoint directories
        Path(self.hyper_param.logdir).mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_data = self.hyper_param.model_dump()
        config_data['finetune_mode'] = self.finetune_mode
        if self.pretrained_model_path:
            config_data['pretrained_model_path'] = str(self.pretrained_model_path)
            
        with (Path(self.hyper_param.logdir) / "config.yaml").open("w") as outfile:
            yaml.dump(config_data, outfile, default_flow_style=False)

        # Initialize TensorBoard
        tensorboard_dir = Path(self.hyper_param.logdir) / "tensorboard" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
        self.vis_manager = VisualizationManager(writer, self.hyper_param.batch_size, hyper_param=self.hyper_param)

        # Checkpoint setup
        ckpt_dir = Path(self.hyper_param.logdir) / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        last_ckpt = ckpt_dir / "last.pt"
        
        # Handle checkpoint loading logic
        self._handle_checkpoint_loading(last_ckpt)
            
        # Add model structure and parameter distribution figures
        self._log_model_info(writer)

        _zfill = len(str(self.hyper_param.num_iteration))

        # Main training loop
        training_start_time = datetime.datetime.now()
        self.batch_times = []
        total_examples = 0
        self.model_manager.set_train(True)
        
        # Set training prefix (for logs and checkpoint naming)
        mode_prefix = "finetune_" if self.finetune_mode else ""
        
        while self.model_manager.step < self.hyper_param.num_iteration:
            train_contents, train_styles = self.data_manager.get_batch(is_training=True)
            
            # Training step
            start_time = datetime.datetime.now()
            self.model_manager.optimizer.zero_grad()
            (train_styled_content, loss, content_loss, style_loss, *_) = self.model_manager.forward(
                contents=train_contents, styles=train_styles, return_features=True
            )
            loss.backward()

            duration_seconds = (datetime.datetime.now() - start_time).total_seconds()
            self.batch_times.append(duration_seconds)

            # Record gradient visualization
            if self.model_manager.step % self.hyper_param.log_step == 0:
                grad_fig = self.vis_manager.visualize_gradients(self.model_manager.named_parameters())
                self.vis_manager.add_figure_to_tensorboard(f'{mode_prefix}Training/Gradients', grad_fig, self.model_manager.step)
                self.vis_manager.log_system_metrics(self.model_manager.step, self.device)

            self.model_manager.optimizer_step()
            
            # Calculate ETA and log it
            if self.model_manager.step % self.hyper_param.log_step == 0:
                self._log_eta(self.model_manager.step, self.hyper_param.num_iteration, duration_seconds, mode_prefix)
            
            # Periodic memory cleanup
            if self.model_manager.step % (self.hyper_param.summary_step * 10) == 0:
                self.vis_manager.clean_memory()
            
            # Record performance metrics and images
            if self.model_manager.step % self.hyper_param.summary_step == 0:
                eval_results = self._run_evaluation(mode_prefix=mode_prefix)  # Evaluate test set
                
                # Record training metrics and images (using current training batch)
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
                
                # Calculate performance metrics
                self._log_performance_metrics(duration_seconds, mode_prefix)

            # Save checkpoints
            if self.model_manager.step % self.hyper_param.save_step == 0:
                self._save_checkpoints(ckpt_dir, mode_prefix, _zfill)

            # Print log
            if self.model_manager.step % self.hyper_param.log_step == 0:
                current_lr = self.model_manager.get_lr()
                self.vis_manager.writer.add_scalar(f"{mode_prefix}Training/Learning_Rate", current_lr, self.model_manager.step)
                examples_per_sec = self.hyper_param.batch_size / duration_seconds
                progress_percent = self.model_manager.step / self.hyper_param.num_iteration * 100
                
                print(
                    f"{datetime.datetime.now()} {'[FINETUNE] ' if self.finetune_mode else ''}step {self.model_manager.step}/{self.hyper_param.num_iteration} "
                    f"({progress_percent:.1f}%), "
                    f"loss={loss:.4f}, style_loss={style_loss:.4f}, content_loss={content_loss:.4f}, "
                    f"lr={current_lr:.6f}, {examples_per_sec:.2f} examples/sec"
                )

        # Training complete, save final model
        final_model_path = ckpt_dir / f"{mode_prefix}final_model.pt"
        self.model_manager.save_checkpoint(final_model_path, is_finetune_ckpt=self.finetune_mode)
        self.model_manager.save_checkpoint(last_ckpt, is_finetune_ckpt=self.finetune_mode)

        self.vis_manager.writer.close()
        print(f"{'Fine-tuning' if self.finetune_mode else 'Training'} completed.")

    def _handle_checkpoint_loading(self, last_ckpt):
        """Handle checkpoint loading logic"""
        # If in fine-tuning mode, preferentially load last.pt file
        if self.finetune_mode and last_ckpt.exists():
            # Load last.pt file
            ckpt_info = self.model_manager.load_checkpoint(last_ckpt)
            print(f"Resumed fine-tuning from last checkpoint at step {self.model_manager.step}")
        elif self.finetune_mode and self.pretrained_model_path:
            # Load pretrained model, but reset step count and optimizer state
            ckpt_info = self.model_manager.load_checkpoint(
                self.pretrained_model_path, 
                reset_step=True,
                reset_optimizer=True
            )
            print(f"Loaded pretrained model for fine-tuning from {self.pretrained_model_path}")
            
            # Check if model was fully trained
            if ckpt_info["loaded_step"] >= ckpt_info["original_num_iteration"]:
                print("Loaded model was fully trained. Starting fine-tuning from step 0.")
            else:
                print(f"Warning: Loaded model was not fully trained ({ckpt_info['loaded_step']}/{ckpt_info['original_num_iteration']} steps).")
        elif last_ckpt.exists():
            # If last checkpoint exists, try to resume training
            ckpt_info = self.model_manager.load_checkpoint(last_ckpt)
            
            # Check if fine-tuning checkpoint doesn't match current mode
            if ckpt_info["is_finetune_ckpt"] != self.finetune_mode:
                print(f"Warning: Checkpoint finetune mode ({ckpt_info['is_finetune_ckpt']}) doesn't match current mode ({self.finetune_mode}).")
                
                if self.finetune_mode:
                    # If current mode is fine-tuning but loaded checkpoint isn't, reset step count and optimizer
                    self.model_manager.reset_for_finetuning()
                    print("Reset model for fine-tuning.")
            else:
                print(f"Resumed {'fine-tuning' if self.finetune_mode else 'training'} from step {self.model_manager.step}")
    
    def _log_model_info(self, writer):
        """Log model structure information"""
        # Add model structure and parameter distribution plots
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
        
        # Add hyperparameter information
        hyperparams = self.hyper_param.model_dump()
        hyperparams['finetune_mode'] = self.finetune_mode
        if self.pretrained_model_path:
            hyperparams['pretrained_model_path'] = str(self.pretrained_model_path)
        self.vis_manager.write_hyperparameters(hyperparams)
        
        param_dist_fig = self.vis_manager.visualize_parameters_distribution(self.model_manager.named_parameters())
        self.vis_manager.add_figure_to_tensorboard('Model/Parameter_Distribution', param_dist_fig, 0)
    
    def _log_eta(self, current_step, total_steps, recent_step_time, prefix=""):
        """Calculate and log estimated time to completion"""
        remaining_steps = total_steps - current_step
        # Use recent steps to calculate average step time
        recent_times = self.batch_times[-min(100, len(self.batch_times)):]
        avg_step_time = sum(recent_times) / len(recent_times) if recent_times else recent_step_time
        
        # Calculate estimated remaining time
        eta_seconds = remaining_steps * avg_step_time
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        # Log to TensorBoard
        self.vis_manager.writer.add_scalar(f'{prefix}Training/ETA_Minutes', eta_seconds / 60, current_step)
        self.vis_manager.writer.add_text(f'{prefix}Training/ETA', f"Estimated time remaining: {eta_str}", current_step)
        
        # Print current progress and ETA
        progress_percent = (current_step / total_steps) * 100
        print(f"Progress: {progress_percent:.1f}% | ETA: {eta_str}")
    
    def _log_performance_metrics(self, duration_seconds, prefix=""):
        """Log performance metrics"""
        # Calculate current and average performance metrics
        current_examples_per_sec = self.hyper_param.batch_size / duration_seconds
        avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        examples_per_sec = self.hyper_param.batch_size / avg_batch_time if avg_batch_time > 0 else 0
        
        # Calculate progress percentage
        progress = self.model_manager.step / self.hyper_param.num_iteration
        
        # Log to TensorBoard
        self.vis_manager.log_performance_metrics(
            current_batch_time=duration_seconds,
            current_examples_per_sec=current_examples_per_sec,
            batch_time=avg_batch_time,
            examples_per_sec=examples_per_sec,
            progress=progress,
            step=self.model_manager.step,
            prefix=prefix
        )
        
        # Keep batch_times list at reasonable size
        self.batch_times = self.batch_times[-100:]
    
    def _save_checkpoints(self, ckpt_dir, mode_prefix, _zfill):
        """Save checkpoint files"""
        # Generate current checkpoint path
        current_ckpt = ckpt_dir / f"{mode_prefix}model_step_{str(self.model_manager.step).zfill(_zfill)}.pt"
        
        # Save current checkpoint
        self.model_manager.save_checkpoint(current_ckpt, is_finetune_ckpt=self.finetune_mode)
        self.model_manager.save_checkpoint(ckpt_dir / "last.pt", is_finetune_ckpt=self.finetune_mode)
        
        # Maintain checkpoint history list
        self.ckpt_history.append(current_ckpt)
        
        # Delete oldest checkpoints beyond max count
        if len(self.ckpt_history) > self.max_ckpt_history:
            old_ckpt = self.ckpt_history.pop(0)
            if old_ckpt.exists():
                old_ckpt.unlink()

    def _run_evaluation(self, mode_prefix=""):
        """Evaluate test set and record results"""
        self.model_manager.set_train(False)
        results = {}
        
        with torch.no_grad():
            test_contents, test_styles = self.data_manager.get_batch(is_training=False)
            (test_styled_content, test_loss, test_content_loss, test_style_loss, *feats) = \
                self.model_manager.forward(contents=test_contents, styles=test_styles, return_features=True)
            
            # Save evaluation results
            results['loss'] = test_loss.item()
            results['content_loss'] = test_content_loss.item()
            results['style_loss'] = test_style_loss.item()
            
            # Log test set metrics and images
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
            
            # Calculate SSIM and PSNR with reduced frequency
            metrics_interval = getattr(self.hyper_param, 'metrics_calc_interval')
            should_calculate = self.model_manager.step % (self.hyper_param.summary_step * metrics_interval) == 0
            
            # Call the visualization manager's method with force=should_calculate
            # This way we leverage the existing visualization logic while controlling frequency
            metrics_results = self.vis_manager.calculate_ssim_psnr(
                test_contents, 
                test_styled_content, 
                self.model_manager.step, 
                prefix=mode_prefix,
                force=should_calculate  # Only calculate when needed
            )
            
            # If metrics were calculated, add them to results
            if metrics_results:
                results.update(metrics_results)
        
        self.model_manager.set_train(True)
        return results