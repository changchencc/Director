import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils
from utils import Checkpointer
from solver import get_optimizer
from envs import make_env, count_steps
from data import EnvIterDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from vis.vis_logger import log_vis
import os
import numpy as np
from pprint import pprint
import pdb
import torch.autograd.profiler as profiler
from time import time
from collections import defaultdict


def anneal_learning_rate(global_step, cfg):

    if global_step < cfg.optim.warmup_iter:
        # warmup
        lr = cfg.optim.base_lr / cfg.optim.warmup_iter * global_step

    else:
        lr = cfg.optim.base_lr

    # decay
    lr = lr * cfg.optim.exp_rate ** (global_step / cfg.optim.decay_step)

    if global_step > cfg.optim.decay_step:
        lr = max(lr, cfg.optim.end_lr)

    return lr


def simulate_test(model, test_env, cfg, global_step, device):

    model.eval()

    obs = test_env.reset()
    action = torch.zeros(1, cfg.env.action_size).float()
    state = None
    done = False
    goal = None
    K = 0

    with torch.no_grad():
        while not done:
            image = torch.tensor(obs["image"].copy())
            action, state, goal = model.policy(
                image.to(device),
                action.to(device),
                goal,
                state,
                training=False,
                sample_goal=(K % cfg.arch.manager.K == 0),
            )
            next_obs, reward, done, info = test_env.step(action[0].cpu().numpy())
            obs = next_obs
            K += 1


def train_16(model, cfg, device):

    print("======== Settings ========")
    pprint(cfg)
    input()

    model = model.to(device)

    print("======== Model ========")
    pprint(model)
    input()

    optimizers = get_optimizer(cfg, model)
    checkpointer_path = os.path.join(
        cfg.checkpoint.checkpoint_dir, cfg.exp_name, cfg.env.name, cfg.run_id
    )
    checkpointer = Checkpointer(checkpointer_path, max_num=cfg.checkpoint.max_num,)
    with open(checkpointer_path + "/config.yaml", "w") as f:
        cfg.dump(stream=f, default_flow_style=False)
        print(f"config file saved to {checkpointer_path + '/config.yaml'}")

    if cfg.resume:
        checkpoint = checkpointer.load(cfg.resume_ckpt)

        if checkpoint:
            model.load_state_dict(checkpoint["model"])
            for k, v in optimizers.items():
                for i in range(len(v)):
                    v[i].load_state_dict(checkpoint[k][i])
            env_step = checkpoint["env_step"]
            global_step = checkpoint["global_step"]

        else:
            env_step = 0
            global_step = 0

    else:
        env_step = 0
        global_step = 0

    writer = SummaryWriter(
        log_dir=os.path.join(cfg.logdir, cfg.exp_name, cfg.env.name, cfg.run_id),
        flush_secs=30,
    )

    datadir = os.path.join(
        cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, "train_episodes"
    )
    test_datadir = os.path.join(
        cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, "test_episodes"
    )
    train_env = make_env(cfg, writer, "train", datadir, store=True)
    test_env = make_env(cfg, writer, "test", test_datadir, store=True)
    if "dmc" in cfg.env.name:
        acts = train_env.action_space
        cfg.env.action_size = acts.n if hasattr(acts, "n") else acts.shape[0]

    # fill in length of 5000 frames
    train_env.reset()
    steps = count_steps(datadir, cfg)
    length = 0
    while steps < cfg.arch.prefill:
        action = train_env.sample_random_action()
        next_obs, reward, done, info = train_env.step(action)
        length += 1
        steps += done * length
        length = length * (1.0 - done)
        if done:
            train_env.reset()

    steps = count_steps(datadir, cfg)
    print(f"collected {steps} steps. Start training...")
    train_ds = EnvIterDataset(datadir, cfg.train.train_steps, cfg.train.batch_length)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=8)
    train_iter = iter(train_dl)
    global_step = max(global_step, steps)

    obs = train_env.reset()
    state = None
    goal = None
    K = 0
    action = torch.zeros(1, cfg.env.action_size).float()
    action[0, 0] = 1.0

    if not cfg.optimize.seperate_scaler:
        scaler = GradScaler()

    while global_step < cfg.total_steps:

        with autocast():
            # print(f'collecting data, global step: {global_step}')
            with torch.no_grad():
                model.eval()
                image = torch.tensor(obs["image"].copy())
                action, state, goal = model.policy(
                    image.to(device),
                    action.to(device),
                    goal,
                    state,
                    sample_goal=(K % cfg.arch.manager.K == 0),
                )
                next_obs, reward, done, info = train_env.step(
                    action[0].detach().cpu().numpy()
                )
                obs = next_obs
                K += 1
                if done:
                    obs = train_env.reset()
                    state = None
                    goal = None
                    K = 0
                    action = torch.zeros(1, cfg.env.action_size).float()
                    action[0, 0] = 1.0

        if global_step % cfg.train.train_every == 0:
            model.train()
            model.requires_grad_(True)

            traj = next(train_iter)
            for k, v in traj.items():
                traj[k] = v.to(device)

            # very complex way to do this experiment!!!
            if cfg.optimize.seperate_scaler:
                model_optimizer, model_scaler = optimizers["model_optimizer"]
            else:
                model_optimizer = optimizers["model_optimizer"][0]

            model_optimizer.zero_grad()

            with autocast():
                (
                    model_loss,
                    model_logs,
                    prior_state,
                    post_state,
                ) = model.world_model_loss(global_step, traj)

            if cfg.optimize.seperate_scaler:
                grad_norm_model = model.world_model.optimize_world_model16(
                    model_loss,
                    model_optimizer,
                    model_scaler,
                    global_step,
                    writer,
                    cfg.optimize.model.weight_decay,
                )
                model_scaler.update()
            else:
                grad_norm_model = model.world_model.optimize_world_model16(
                    model_loss,
                    model_optimizer,
                    scaler,
                    global_step,
                    writer,
                    cfg.optimize.model.weight_decay,
                )

            if cfg.optimize.seperate_scaler:
                goal_vae_optimizer, goal_vae_scaler = optimizers["goal_vae_optimizer"]
            else:
                goal_vae_optimizer = optimizers["goal_vae_optimizer"][0]
            goal_vae_optimizer.zero_grad()
            with autocast():
                goal_loss, goal_logs = model.goal_vae_loss(post_state)

            if cfg.optimize.seperate_scaler:
                grad_norm_goalvae = model.optimize_goalvae16(
                    goal_loss,
                    goal_vae_optimizer,
                    goal_vae_scaler,
                    global_step,
                    writer,
                    cfg.optimize.goal_vae.weight_decay,
                )
                goal_vae_scaler.update()
            else:
                grad_norm_goalvae = model.optimize_goalvae16(
                    goal_loss,
                    goal_vae_optimizer,
                    scaler,
                    global_step,
                    writer,
                    cfg.optimize.goal_vae.weight_decay,
                )

            if cfg.optimize.seperate_scaler:
                actor_optimizer, actor_scaler = optimizers["actor_optimizer"]
                value_optimizer, value_scaler = optimizers["value_optimizer"]
                mgr_actor_optimizer, mgr_actor_scaler = optimizers[
                    "mgr_actor_optimizer"
                ]
                mgr_value_optimizer, mgr_value_scaler = optimizers[
                    "mgr_value_optimizer"
                ]
            else:
                actor_optimizer = optimizers["actor_optimizer"][0]
                value_optimizer = optimizers["value_optimizer"][0]
                mgr_actor_optimizer = optimizers["mgr_actor_optimizer"][0]
                mgr_value_optimizer = optimizers["mgr_value_optimizer"][0]
            actor_optimizer.zero_grad()
            value_optimizer.zero_grad()
            mgr_actor_optimizer.zero_grad()
            mgr_value_optimizer.zero_grad()
            with autocast():
                (
                    actor_loss,
                    value_loss,
                    mgr_actor_loss,
                    mgr_value_loss,
                    actor_value_logs,
                ) = model.actor_and_value_loss(global_step, post_state)

            if cfg.optimize.seperate_scaler:
                grad_norm_actor = model.optimize_actor16(
                    actor_loss,
                    actor_optimizer,
                    actor_scaler,
                    global_step,
                    writer,
                    cfg.optimize.actor.weight_decay,
                )
                actor_scaler.update()

            else:
                grad_norm_actor = model.optimize_actor16(
                    actor_loss,
                    actor_optimizer,
                    scaler,
                    global_step,
                    writer,
                    cfg.optimize.actor.weight_decay,
                )

            if cfg.optimize.seperate_scaler:
                grad_norm_value = model.optimize_value16(
                    value_loss,
                    value_optimizer,
                    value_scaler,
                    global_step,
                    writer,
                    cfg.optimize.value.weight_decay,
                )
                value_scaler.update()
            else:
                grad_norm_value = model.optimize_value16(
                    value_loss,
                    value_optimizer,
                    scaler,
                    global_step,
                    writer,
                    cfg.optimize.value.weight_decay,
                )
            if not cfg.arch.worker_only:
                if cfg.optimize.seperate_scaler:
                    grad_norm_mgr_actor = model.optimize_mgr_actor16(
                        mgr_actor_loss,
                        mgr_actor_optimizer,
                        mgr_actor_scaler,
                        global_step,
                        writer,
                        cfg.optimize.mgr_actor.weight_decay,
                    )
                    mgr_actor_scaler.update()
                else:
                    grad_norm_mgr_actor = model.optimize_mgr_actor16(
                        mgr_actor_loss,
                        mgr_actor_optimizer,
                        scaler,
                        global_step,
                        writer,
                        cfg.optimize.mgr_actor.weight_decay,
                    )

                if cfg.optimize.seperate_scaler:
                    grad_norm_mgr_value = model.optimize_mgr_value16(
                        mgr_value_loss,
                        mgr_value_optimizer,
                        mgr_value_scaler,
                        global_step,
                        writer,
                        cfg.optimize.mgr_value.weight_decay,
                    )
                    mgr_value_scaler.update()
                else:
                    grad_norm_mgr_value = model.optimize_mgr_value16(
                        mgr_value_loss,
                        mgr_value_optimizer,
                        scaler,
                        global_step,
                        writer,
                        cfg.optimize.mgr_value.weight_decay,
                    )

            if not cfg.optimize.seperate_scaler:
                scaler.update()

            if global_step % cfg.train.log_every_step == 0:
                with torch.no_grad():
                    logs = {}
                    logs.update(model_logs)
                    logs.update(actor_value_logs)
                    logs.update(goal_logs)
                    model.write_logs(logs, traj, global_step, writer)

                grad_norm = dict(
                    grad_norm_model=grad_norm_model,
                    grad_norm_actor=grad_norm_actor,
                    grad_norm_value=grad_norm_value,
                    grad_norm_goalvae=grad_norm_goalvae,
                )
                if not cfg.arch.worker_only:
                    grad_norm.update(
                        {
                            "grad_norm_mgr_actor": grad_norm_mgr_actor,
                            "grad_norm_mgr_value": grad_norm_mgr_value,
                        }
                    )

                for k, v in grad_norm.items():
                    writer.add_scalar(
                        "train_grad_norm/" + k, v, global_step=global_step
                    )

        # evaluate RL
        if global_step % cfg.train.eval_every_step == 0:

            with autocast():
                simulate_test(model, test_env, cfg, global_step, device)

        if global_step % cfg.train.checkpoint_every_step == 0:
            env_step = count_steps(datadir, cfg)
            checkpointer.save("", model, optimizers, global_step, env_step)

        global_step += 1


def train_32(model, cfg, device):

    print("======== Settings ========")
    pprint(cfg)
    input()

    print("======== Model ========")
    pprint(model)
    input()

    model = model.to(device)

    optimizers = get_optimizer(cfg, model)
    checkpointer = Checkpointer(
        os.path.join(
            cfg.checkpoint.checkpoint_dir, cfg.exp_name, cfg.env.name, cfg.run_id
        ),
        max_num=cfg.checkpoint.max_num,
    )

    if cfg.resume:
        checkpoint = checkpointer.load(cfg.resume_ckpt)

        if checkpoint:
            model.load_state_dict(checkpoint["model"])
            for k, v in optimizers.items():
                v.load_state_dict(checkpoint[k])
            env_step = checkpoint["env_step"]
            global_step = checkpoint["global_step"]

        else:
            env_step = 0
            global_step = 0

    else:
        env_step = 0
        global_step = 0

    writer = SummaryWriter(
        log_dir=os.path.join(cfg.logdir, cfg.exp_name, cfg.env.name, cfg.run_id),
        flush_secs=30,
    )

    datadir = os.path.join(
        cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, "train_episodes"
    )
    test_datadir = os.path.join(
        cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, "test_episodes"
    )
    train_env = make_env(cfg, writer, "train", datadir, store=True)
    test_env = make_env(cfg, writer, "test", test_datadir, store=True)

    # fill in length of 5000 frames
    train_env.reset()
    steps = count_steps(datadir, cfg)
    length = 0
    while steps < cfg.arch.prefill:
        action = train_env.sample_random_action()
        next_obs, reward, done, info = train_env.step(action)
        length += 1
        steps += done * length
        length = length * (1.0 - done)
        if done:
            train_env.reset()

    steps = count_steps(datadir, cfg)
    print(f"collected {steps} steps. Start training...")
    train_ds = EnvIterDataset(datadir, cfg.train.train_steps, cfg.train.batch_length)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=4)
    train_iter = iter(train_dl)
    global_step = max(global_step, steps)

    obs = train_env.reset()
    state = None
    action = torch.zeros(1, cfg.env.action_size).float()

    while global_step < cfg.total_steps:

        with torch.no_grad():
            model.eval()
            image = torch.tensor(obs["image"])
            action, state = model.policy(
                image.to(device),
                action.to(device),
                global_step,
                state,
                prior=cfg.rollout_prior,
            )
            next_obs, reward, done, info = train_env.step(
                action[0].detach().cpu().numpy()
            )
            obs = next_obs
            if done:
                train_env.reset()
                state = None
                action = torch.zeros(1, cfg.env.action_size).float()

        if global_step % cfg.train.train_every == 0:

            model.train()

            traj = next(train_iter)
            for k, v in traj.items():
                traj[k] = v.to(device)

            logs = {}

            model_optimizer = optimizers["model_optimizer"]
            model_optimizer.zero_grad()
            prior_state, post_state = model.dynamic(traj, None)
            model_loss, model_logs = model.world_model_loss(
                global_step, traj, prior_state, post_state
            )
            grad_norm_model = model.optimize_world_model32(model_loss, model_optimizer)

            actor_optimizer = optimizers["actor_optimizer"]
            value_optimizer = optimizers["value_optimizer"]
            actor_optimizer.zero_grad()
            value_optimizer.zero_grad()
            actor_loss, value_loss, actor_value_logs = model.actor_and_value_loss(
                global_step, post_state
            )
            grad_norm_actor = model.optimize_actor32(actor_loss, actor_optimizer)
            grad_norm_value = model.optimize_value32(value_loss, actor_optimizer)

            if global_step % cfg.train.log_every_step == 0:

                logs.update(model_logs)
                logs.update(actor_value_logs)
                model.write_logs(logs, traj, global_step, writer)

                grad_norm = dict(
                    grad_norm_model=grad_norm_model,
                    grad_norm_actor=grad_norm_actor,
                    grad_norm_value=grad_norm_value,
                )

                for k, v in grad_norm.items():
                    writer.add_scalar(
                        "train_grad_norm/" + k, v, global_step=global_step
                    )

        # evaluate RL
        if global_step % cfg.train.eval_every_step == 0:
            simulate_test(model, test_env, cfg, global_step, device)

        if global_step % cfg.train.checkpoint_every_step == 0:
            env_step = count_steps(datadir, cfg)
            checkpointer.save("", model, optimizers, global_step, env_step)

        global_step += 1

