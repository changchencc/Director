import pdb

from torch import optim
from torch.cuda.amp import GradScaler, autocast


def get_optimizer(cfg, model, params=None):

    if cfg.optimize.optimizer == "adam":
        opt_fn = optim.Adam
    if cfg.optimize.optimizer == "adamW":
        opt_fn = optim.AdamW

    kwargs = {"weight_decay": cfg.optimize.weight_decay, "eps": cfg.optimize.eps}

    if cfg.model == "dreamer":
        model_optimizer = opt_fn(
            model.world_model.parameters(), lr=cfg.optimize.model_lr, **kwargs
        )
        value_optimizer = opt_fn(
            model.value.parameters(), lr=cfg.optimize.value_lr, **kwargs
        )
        actor_optimizer = opt_fn(
            model.actor.parameters(), lr=cfg.optimize.actor_lr, **kwargs
        )

        return {
            "model_optimizer": [model_optimizer,],
            "value_optimizer": [value_optimizer,],
            "actor_optimizer": [actor_optimizer,],
        }
    if cfg.model == "dreamer_plan":
        model_optimizer = opt_fn(
            model.world_model.parameters(), lr=cfg.optimize.model_lr, **kwargs
        )
        value_optimizer = opt_fn(
            model.value.parameters(), lr=cfg.optimize.value_lr, **kwargs
        )
        actor_optimizer = opt_fn(
            model.actor.parameters(), lr=cfg.optimize.actor_lr, **kwargs
        )
        goal_vae_optimizer = opt_fn(
            model.goal_vae.parameters(), lr=cfg.optimize.goal_vae_lr, **kwargs
        )

        mgr_actor_optimizer = opt_fn(
            model.mgr_actor.parameters(), lr=cfg.optimize.mgr_actor_lr, **kwargs
        )

        mgr_value_optimizer = opt_fn(
            model.mgr_value.parameters(), lr=cfg.optimize.mgr_value_lr, **kwargs
        )

        return {
            "model_optimizer": [model_optimizer, GradScaler()],
            "value_optimizer": [value_optimizer, GradScaler()],
            "actor_optimizer": [actor_optimizer, GradScaler()],
            "goal_vae_optimizer": [goal_vae_optimizer, GradScaler()],
            "mgr_actor_optimizer": [mgr_actor_optimizer, GradScaler()],
            "mgr_value_optimizer": [mgr_value_optimizer, GradScaler()],
        }

