from pydantic import BaseModel, validator
from typing import Optional, List, Tuple, Union, Any
from pydantic.fields import Field
import math

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
    fixed_batch_size: Optional[int] = 8
    resize_size: int = 512 
    
    # Export params
    use_fixed_size: bool = False  # For static computational graph
    
    # Training iteration params
    num_iteration: int = 160000
    log_step: int = 10
    save_step: int = 1000
    summary_step: int = 100
    max_ckpts: int = 3
    
    # Fine-tuning specific params
    finetune_learning_rate: Optional[float] = None
    finetune_iterations: Optional[int] = None
    pretrained_model: Optional[str] = None  # 添加预训练模型路径

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
        """Convert groups to list if it's an integer"""
        if isinstance(v, int):
            return [v] * 4
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
            # Calculate from ratios
            base_channels = [512, 256, 128, 64]
            return [max(1, int(c * r)) for c, r in zip(base_channels, self.groups_ratios)]
        
    @validator('num_iteration')
    def validate_iterations(cls, v, values):
        if values.get('finetune_lr') and not values.get('checkpoint_path'):
            raise ValueError("微调模式需要提供checkpoint_path")
        return v