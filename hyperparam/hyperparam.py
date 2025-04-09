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
    
    # Fine-tuning specific params
    finetune_learning_rate: Optional[float] = None
    finetune_iterations: Optional[int] = None
    pretrained_model: Optional[str] = None  # 预训练模型路径

    @validator('image_shape')
    def validate_image_shape(cls, v, values):
        """确保image_shape是一个由两个整数组成的元组"""
        if not isinstance(v, tuple) or len(v) != 2:
            # 如果不是正确的元组格式，尝试从image_size转换
            image_size = values.get('image_size', 256)
            return (image_size, image_size)
        return v
    
    @validator('groups')
    def validate_groups(cls, v):
        """如果groups是一个整数，则将其转换为列表"""
        if isinstance(v, int):
            return [v] * 4  # 为4个解码器层创建相同的分组值
        return v
    
    def get_groups(self) -> List[int]:
        """获取最终的分组配置"""
        # 优先级: groups_list > groups > 基于ratios计算
        if self.groups_list:
            return self.groups_list
        elif self.groups:
            # 已经通过验证器转换为列表
            return self.groups
        else:
            # 基于比例计算
            base_channels = [512, 256, 128, 64]
            return [max(1, int(c * r)) for c, r in zip(base_channels, self.groups_ratios)]
            
    def create_export_config(self) -> Dict[str, Any]:
        """创建用于模型导出的配置字典"""
        return {
            'export_mode': self.export_mode,
            'fixed_batch_size': self.fixed_batch_size,
            'use_fixed_size': self.use_fixed_size
        }
        
    class Config:
        """Pydantic配置"""
        validate_assignment = True  # 在属性赋值时验证
        arbitrary_types_allowed = True  # 允许任意类型