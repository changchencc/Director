import pdb

import torch
from torch import optim
from torch.cuda.amp import GradScaler


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: torch.optim.NAdam(parameters, lr=lr, eps=eps),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=False):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def get_optimizer(cfg, model, params=None):

    if cfg.optimize.optimizer == "adam":
        opt_fn = optim.Adam
    if cfg.optimize.optimizer == "adamW":
        opt_fn = optim.AdamW

    if cfg.model == "dreamer":

        kwargs = {"weight_decay": 0.0, "eps": cfg.optimize.eps}

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
            "model_optimizer": model_optimizer,
            "value_optimizer": value_optimizer,
            "actor_optimizer": actor_optimizer,
        }
    if cfg.model == "dreamer_plan":

        model_kwargs = {
            "weight_decay": cfg.optimize.model.weight_decay,
            "eps": cfg.optimize.model.eps,
            "lr": cfg.optimize.model.lr,
        }
        model_optimizer = opt_fn(model.world_model.parameters(), **model_kwargs)

        value_kwargs = {
            "weight_decay": cfg.optimize.value.weight_decay,
            "eps": cfg.optimize.value.eps,
            "lr": cfg.optimize.value.lr,
        }
        value_optimizer = opt_fn(model.value.parameters(), **value_kwargs)

        actor_kwargs = {
            "weight_decay": cfg.optimize.actor.weight_decay,
            "eps": cfg.optimize.actor.eps,
            "lr": cfg.optimize.actor.lr,
        }
        actor_optimizer = opt_fn(model.actor.parameters(), **actor_kwargs)

        goal_vae_kwargs = {
            "weight_decay": cfg.optimize.goal_vae.weight_decay,
            "eps": cfg.optimize.goal_vae.eps,
            "lr": cfg.optimize.goal_vae.lr,
        }
        goal_vae_optimizer = opt_fn(model.goal_vae.parameters(), **goal_vae_kwargs)

        mgr_actor_kwargs = {
            "weight_decay": cfg.optimize.mgr_actor.weight_decay,
            "eps": cfg.optimize.mgr_actor.eps,
            "lr": cfg.optimize.mgr_actor.lr,
        }
        mgr_actor_optimizer = opt_fn(model.mgr_actor.parameters(), **mgr_actor_kwargs)

        mgr_value_kwargs = {
            "weight_decay": cfg.optimize.mgr_value.weight_decay,
            "eps": cfg.optimize.mgr_value.eps,
            "lr": cfg.optimize.mgr_value.lr,
        }
        mgr_value_optimizer = opt_fn(model.mgr_value.parameters(), **mgr_value_kwargs)

        if cfg.optimize.seperate_scaler:
            return {
                "model_optimizer": [model_optimizer, GradScaler()],
                "value_optimizer": [value_optimizer, GradScaler()],
                "actor_optimizer": [actor_optimizer, GradScaler()],
                "goal_vae_optimizer": [goal_vae_optimizer, GradScaler()],
                "mgr_actor_optimizer": [mgr_actor_optimizer, GradScaler()],
                "mgr_value_optimizer": [mgr_value_optimizer, GradScaler()],
            }
        else:
            return {
                "model_optimizer": [model_optimizer,],
                "value_optimizer": [value_optimizer,],
                "actor_optimizer": [actor_optimizer,],
                "goal_vae_optimizer": [goal_vae_optimizer,],
                "mgr_actor_optimizer": [mgr_actor_optimizer,],
                "mgr_value_optimizer": [mgr_value_optimizer,],
            }

