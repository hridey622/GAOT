from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

# model default config
from src.model.layers.attn import TransformerConfig
from src.model.layers.magno2d_vx import MAGNOConfig
from ..optimizers import OptimizerargsConfig

from omegaconf import OmegaConf

def merge_config(default_config_class, user_config):
    default_config_struct = OmegaConf.structured(default_config_class)
    merged_config = OmegaConf.merge(default_config_struct, user_config)
    return OmegaConf.to_object(merged_config)

@dataclass
class SetUpConfig:
    seed: int = 42                                          
    device: str = "cuda:0"
    dtype: str = "torch.float32"
    trainer_name: str = "sequential"                                        # [static, static_unstruc, sequential]
    train: bool = True
    test: bool = False
    ckpt: bool = False
    use_variance_test: bool = False                                         # TODO needs to develop.
    measure_inf_time: bool = False                                          # TODO needs to be examined
    visualize_encoder_output: bool = False
    vis_component: str = "encoder"
    # Parameters for distributed mode
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"

@dataclass
class ModelArgsConfig:
    magno: MAGNOConfig = field(default_factory=MAGNOConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

@dataclass
class ModelConfig:
    name: str = "goat2d_vx"
    use_conditional_norm: bool = False
    latent_tokens_size: Tuple[int, int] = (64, 64)          # H, W
    args: ModelArgsConfig = field(default_factory=ModelArgsConfig)

@dataclass
class DatasetConfig:
    name: str = "CE-Gauss"
    metaname: str = "rigno-unstructured/CE-Gauss"
    base_path: str = "/cluster/work/math/camlab-data/rigno-unstructured/"
    train_size: int = 1024
    val_size: int = 128
    test_size: int = 256
    coord_scaling: str = "per_dim_scaling"                                  #  Support ['global_scaling', 'per_dim_scaling'].
    batch_size: int = 64                                                    
    num_workers: int = 4
    shuffle: bool = True
    use_metadata_stats: bool = False
    sample_rate: float = 0.1
    use_sparse: bool = False                                                # Use full resolution for Poseidon Dataset
    rand_dataset: bool = False                                              # Whether to randomize the sequence of loaded dataset
    # Designed for Time-dependent dataset
    max_time_diff: int = 14                                                 # Max time difference for creating data pairs
    use_time_norm: bool = True                                              # whether to use normalization for lead time and time_difference
    metric: str = "final_step"
    predict_mode: str = "all"
    stepper_mode: str = "output"                                            # [output, residual, time_der]


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    args: OptimizerargsConfig = field(default_factory=OptimizerargsConfig)

@dataclass
class PathConfig:
    ckpt_path: str = ".ckpt/test/test.pt"
    loss_path: str = ".loss/test/test.png"
    result_path: str = ".result/test/test.png"
    database_path: str = ".database/test/test.csv"


