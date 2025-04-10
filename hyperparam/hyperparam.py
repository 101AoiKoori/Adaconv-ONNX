from pydantic import BaseModel, validator, root_validator
from typing import Optional, List, Tuple, Union, Any, Dict
from pydantic.fields import Field

class Hyperparameter(BaseModel):
    # Dataset params
    data_path: str = "data"
    logdir: str = "runs"

    # Model params
    image_size: int = 256  
    image_shape: Tuple[int, int] = (256, 256)  
    style_dim: int = 512
    style_kernel: int = 3
    style_weight: float = 100.0

    # Group configuration
    groups_ratios: List[float] = [1.0, 0.5, 0.25, 0.125]
    groups: Optional[Union[int, List[int]]] = None
    groups_list: Optional[List[int]] = None

    # Training params
    learning_rate: float = 0.0001
    batch_size: int = 8
    fixed_batch_size: Optional[int] = None
    resize_size: int = 512 
    
    # Export params
    use_fixed_size: bool = False  # For static computational graph
    export_mode: bool = False     # Enable export mode
    
    # Training iteration params
    num_iteration: int = 160000
    log_step: int = 10
    save_step: int = 1000
    summary_step: int = 100
    max_ckpts: int = 3
    
    # Metrics calculation frequency
    metrics_calc_interval: int = 10  # Multiplier for SSIM/PSNR calculation interval
    
    # Fine-tuning specific params
    finetune_learning_rate: Optional[float] = None
    finetune_iterations: Optional[int] = None
    pretrained_model: Optional[str] = None  # Path to pretrained model

    @validator('image_shape')
    def validate_image_shape(cls, v, values):
        """Ensure image_shape is a tuple of two integers"""
        if not isinstance(v, tuple) or len(v) != 2:
            # If not a proper tuple, try to convert from image_size
            image_size = values.get('image_size', 256)
            return (image_size, image_size)
        return v
    
    @validator('groups')
    def validate_groups(cls, v):
        """If groups is an integer, convert it to a list"""
        if isinstance(v, int):
            return [v] * 4  # Create same group value for all 4 decoder layers
        return v
    
    def get_groups(self) -> List[int]:
        """Get the final groups configuration"""
        # Priority: groups_list > groups > calculated from ratios
        if self.groups_list:
            return self.groups_list
        elif self.groups:
            # Already validated to be a list
            return self.groups
        else:
            # Calculate based on ratios
            base_channels = [512, 256, 128, 64]
            return [max(1, int(c * r)) for c, r in zip(base_channels, self.groups_ratios)]
            
    def create_export_config(self) -> Dict[str, Any]:
        """Create export configuration dictionary for model export"""
        return {
            'export_mode': self.export_mode,
            'fixed_batch_size': self.fixed_batch_size,
            'use_fixed_size': self.use_fixed_size
        }
        
    def get_finetune_config(self) -> Dict[str, Any]:
        """Get fine-tuning mode configuration"""
        return {
            'learning_rate': self.finetune_learning_rate or self.learning_rate * 0.1,
            'num_iteration': self.finetune_iterations or int(self.num_iteration * 0.25),
        }
        
    class Config:
        """Pydantic configuration"""
        validate_assignment = True  # Validate on attribute assignment
        arbitrary_types_allowed = True  # Allow arbitrary types