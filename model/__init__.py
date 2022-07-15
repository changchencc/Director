import torch
from .dreamer import Dreamer
from .dreamer_plan import DreamerPlan


def get_model(cfg, device, seed=0):

    torch.manual_seed(seed=seed)
    if cfg.model == "dreamer":
        model = Dreamer(cfg)
    if cfg.model == "dreamer_plan":
        model = DreamerPlan(cfg)

    return model

