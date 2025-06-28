import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.envs_tools import get_shape_from_obs_space


class ExpertPolicy(nn.Module):
    """Expert policy model. Outputs actions given observations."""
    pass