from typing import Tuple, Optional, Union
from dataclasses import dataclass
from omegaconf import OmegaConf
import math

from .goat2d_vx import GOAT2D_VX
from .goat2d_fx import GOAT2D_FX

def init_model(
        input_size: int,
        output_size: int,
        model: str = "goat2d_vx",
        config: Optional[dataclass] = None
):
    """
    Initialize the model based on the provided model name and configuration.
    
    Args:
        input_size (int): The size of the input data.
        output_size (int): The size of the output data.
        model (str): The name of the model to initialize. Default is "goat2d_vx".
        config (Optional[dataclass]): Configuration object for the model. Default is None.
    
    Returns:
        An instance of the specified model.
    """
    supported_models = [
        "goat2d_vx",
        "goat2d_fx",
    ]
    assert model.lower() in supported_models, (
        f"model {model} not supported, only support {supported_models} "
    )

    if model.lower() == "goat2d_vx":
        return GOAT2D_VX(
            input_size = input_size,
            output_size = output_size,
            config = config)
    elif model.lower() == "goat2d_fx":
        return GOAT2D_FX(
            input_size = input_size,
            output_size = output_size,
            config = config)
    else:
        raise NotImplementedError(f"Model {model} is not implemented.")