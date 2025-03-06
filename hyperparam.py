from pydantic import BaseModel

class Hyperparameter(BaseModel):
    # dataset params
    data_path: str = "data"
    logdir: str = "runs"
    # model params
    image_shape: list[int] = [256, 256]
    style_dim: int = 512
    style_kernel: int = 3
    resize_size: int = 512
    style_weight: float = 100.0
    learning_rate: float = 0.0001
    batch_size: int = 8
    num_iteration: int = 160000
    log_step: int = 10
    save_step: int = 1000
    summary_step: int = 100
    max_ckpts: int = 3
    groups: int = 32  # 新增 groups 参数
    # training params
    fixed_batch_size: int = 8
    # inference params