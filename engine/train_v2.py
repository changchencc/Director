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

    with torch.no_grad():
        while not done:
            image = torch.tensor(obs["image"])
            action, state = model.policy(
                image.to(device), action.to(device), global_step, state, training=False
            )
            next_obs, reward, done, info = test_env.step(action[0].cpu().numpy())
            obs = next_obs


def train_16_with_real_reward(model, cfg, device):

    print("======== Settings ========")
    pprint(cfg)
    input()

    model = model.to(device)

    print("======== Model ========")
    pprint(model)
    input()

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
    train_envs = [
        make_env(cfg, writer, "train", datadir, store=i == 0, seed=i + 1)
        for i in range(cfg.train.batch_size)
    ]
    test_env = make_env(
        cfg, writer, "test", test_datadir, store=False, seed=cfg.train.batch_size
    )

    # fill in length of 5000 frames
    # train_env.reset()
    # steps = count_steps(datadir, cfg)
    # length = 0
    # while steps < cfg.arch.prefill:
    #   action = train_env.sample_random_action()
    #   next_obs, reward, done, info = train_env.step(action)
    #   length += 1
    #   steps += done * length
    #   length = length * (1. - done)
    #   if done:
    #     train_env.reset()

    steps = count_steps(datadir, cfg)
    print(f"collected {steps} steps. Start training...")
    # train_ds = EnvIterDataset(datadir, cfg.train.train_steps, cfg.train.batch_length)
    # train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=8)
    # train_iter = iter(train_dl)
    global_step = max(global_step, steps)

    # maintain a data buffer
    # B, T, C, H, W
    image_t0 = (
        torch.stack(
            [
                torch.tensor(train_env.reset()["image"]).float()
                for train_env in train_envs
            ],
            dim=0,
        )
        .float()
        .unsqueeze(1)
    )
    actions = torch.cat(
        [
            torch.zeros(1, 1, cfg.env.action_size).float()
            for _ in range(cfg.train.batch_size)
        ],
        dim=0,
    )  # B, T, C
    actions[:, 0, 0] = 1.0

    reward_mem = (
        torch.tensor([0.0] * cfg.train.batch_size)
        .float()
        .reshape(cfg.train.batch_size, 1)
    )  # B, T
    obs_mem = image_t0
    done_mem = (
        torch.tensor([0.0] * cfg.train.batch_size)
        .float()
        .reshape(cfg.train.batch_size, 1)
    )  # B, T
    action_mem = actions
    state_mem = model.dynamic.init_state(cfg.train.batch_size, device)
    for k, v in state_mem.items():
        state_mem[k] = v.unsqueeze(1)  # extent temporal dim

    mem_size = obs_mem.shape[1]

    scaler = GradScaler()

    while global_step < cfg.total_steps:

        with autocast():
            while mem_size < cfg.train.batch_length:
                with torch.no_grad():
                    model.eval()

                    for steps in range(cfg.train.batch_length):

                        # predict a_{t+1} given a_t, o_t
                        state_t = {
                            k: v[:, steps].to(device) for k, v in state_mem.items()
                        }
                        actions, state_step = model.policy(
                            image_t0[:, 0].to(device),
                            action_mem[:, steps].to(device),
                            global_step,
                            state_t,
                        )
                        # for i in range(cfg.train.batch_size):
                        #   if state_mem is None:
                        #     s_t, a_t = model.policy(image_t0[i].to(device), action_mem[i:i + 1, i].to(device), global_step, None)
                        #   else:
                        #     state_t = {k: v[i:i+1, i].to(device) for k, v in state_mem.items()}
                        #     s_t, a_t = model.policy(image_t0[i].to(device), action_mem[i:i + 1, i].to(device), global_step, state_t)
                        #
                        #   new_actions.append(a_t)
                        #   for k, v in s_t.items():
                        #     new_states[k].append(v)

                        # actions = torch.cat(new_actions, dim=0).cpu().unsqueeze(1)  # B, T, 18
                        # for k, v in new_states.items():
                        #   new_states[k] = torch.cat(v, dim=0).unsqueeze(1)
                        actions = actions.unsqueeze(1).cpu()  # B, T, 18

                        for k, v in state_mem.items():
                            state_mem[k] = torch.cat(
                                [v, state_step[k].unsqueeze(1)], dim=1
                            )  # B, T, H

                        reward_t = []
                        obs_t1 = []
                        done_t = []
                        for i, train_env in enumerate(train_envs):
                            obs, reward, done, info = train_env.step(
                                actions[i, -1].detach().cpu().numpy()
                            )

                            obs_t1.append(obs["image"])
                            reward_t.append(reward)
                            done_t.append(done)

                        for i in range(cfg.train.batch_size):
                            if done_t[i]:
                                train_envs[i].reset()
                                actions[i] = torch.zeros(
                                    1, 1, cfg.env.action_size
                                ).float()  # B, T, C
                                actions[i, 0, 0] = 1.0
                        # print(f'collecting data, global step: {global_step}')

                        reward_t = (
                            torch.tensor(reward_t)
                            .float()
                            .reshape(cfg.train.batch_size, 1)
                        )  # B, T
                        image_t1 = torch.tensor(obs_t1).float().unsqueeze(1)
                        done_t = (
                            torch.tensor(done_t)
                            .float()
                            .reshape(cfg.train.batch_size, 1)
                        )  # B, T

                        reward_mem = torch.cat([reward_mem, reward_t], dim=1)
                        obs_mem = torch.cat([obs_mem, image_t1], dim=1)
                        done_mem = torch.cat([done_mem, done_t], dim=1)
                        action_mem = torch.cat(
                            [action_mem, actions], dim=1
                        )  # B, T, 18, {a_{t-1}, o_t, r_t}

                        image_t0 = image_t1

                    mem_size = obs_mem.shape[1]

        traj = {
            "image": obs_mem[:, -cfg.train.batch_length :].to(device),
            "action": action_mem[:, -cfg.train.batch_length :].to(device),
            "reward": reward_mem[:, -cfg.train.batch_length :].to(device),
            "done": done_mem[:, -cfg.train.batch_length :].to(device),
            "discount": 1.0 - done_mem[:, -cfg.train.batch_length :].to(device),
        }

        if global_step % cfg.train.train_every == 0:
            model.train()

            model_optimizer = optimizers["model_optimizer"]
            model_optimizer.zero_grad()

            with autocast():
                prior_state, post_state = model.dynamic(traj, None)
                model_loss, model_logs = model.world_model_loss(
                    global_step, traj, prior_state, post_state
                )

            grad_norm_model = model.optimize_world_model16(
                model_loss, model_optimizer, scaler
            )

            actor_optimizer = optimizers["actor_optimizer"]
            value_optimizer = optimizers["value_optimizer"]
            actor_optimizer.zero_grad()
            value_optimizer.zero_grad()
            with autocast():
                (
                    actor_loss,
                    value_loss,
                    actor_value_logs,
                    imag_traj,
                ) = model.actor_and_value_loss_with_real_reward(
                    traj, global_step, post_state, train_envs
                )

            grad_norm_actor = model.optimize_actor16(
                actor_loss, actor_optimizer, scaler
            )
            grad_norm_value = model.optimize_value16(
                value_loss, value_optimizer, scaler
            )

            scaler.update()

            imag_obs = imag_traj["image"]
            imag_reward = imag_traj["reward"]
            imag_action = imag_traj["action"]
            imag_done = imag_traj["done"]
            # imag_state = imag_traj['state']

            reward_mem = torch.cat([reward_mem, imag_reward], dim=1)[
                :, -(cfg.train.batch_length + cfg.arch.H) :
            ]
            obs_mem = torch.cat([obs_mem, imag_obs], dim=1)[
                :, -(cfg.train.batch_length + cfg.arch.H) :
            ]
            done_mem = torch.cat([done_mem, imag_done], dim=1)[
                :, -(cfg.train.batch_length + cfg.arch.H) :
            ]
            action_mem = torch.cat([action_mem, imag_action], dim=1)[
                :, -(cfg.train.batch_length + cfg.arch.H) :
            ]  # B, T, 18
            # for k, v in state_mem.items():
            #   state_mem = torch.cat([v, imag_state[k]], dim=1)[:, -(cfg.train.batch_length + cfg.arch.H):]  # B, T, 18

            mem_size = obs_mem.shape[1]

            if global_step % cfg.train.log_every_step == 0:
                logs = {}
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
        # if global_step % cfg.train.eval_every_step == 0:
        #
        #   with autocast():
        #     simulate_test(model, test_env, cfg, global_step, device)

        if global_step % cfg.train.checkpoint_every_step == 0:
            env_step = count_steps(datadir, cfg)
            checkpointer.save("", model, optimizers, global_step, env_step)

        global_step += 1


def train_16(model, cfg, device):

    print("======== Settings ========")
    pprint(cfg)
    input()

    model = model.to(device)

    print("======== Model ========")
    pprint(model)
    input()

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
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=8)
    train_iter = iter(train_dl)
    global_step = max(global_step, steps)

    obs = train_env.reset()
    state = None
    action = torch.zeros(1, cfg.env.action_size).float()
    action[0, 0] = 1.0

    scaler = GradScaler()

    while global_step < cfg.total_steps:

        with autocast():
            # print(f'collecting data, global step: {global_step}')
            with torch.no_grad():
                model.eval()
                image = torch.tensor(obs["image"])
                action, state = model.policy(
                    image.to(device), action.to(device), global_step, state, prior=False
                )
                next_obs, reward, done, info = train_env.step(
                    action[0].detach().cpu().numpy()
                )
                obs = next_obs
                if done:
                    obs = train_env.reset()
                    state = None
                    action = torch.zeros(1, cfg.env.action_size).float()
                    action[0, 0] = 1.0

        if global_step % cfg.train.train_every == 0:
            model.train()
            model.requires_grad_(True)

            traj = next(train_iter)
            for k, v in traj.items():
                traj[k] = v.to(device)

            model_optimizer = optimizers["model_optimizer"]
            model_optimizer.zero_grad()

            with autocast():
                (
                    model_loss,
                    model_logs,
                    prior_state,
                    post_state,
                ) = model.world_model_loss(global_step, traj)

            grad_norm_model = model.world_model.optimize_world_model16(
                model_loss, model_optimizer, scaler, global_step, writer
            )

            actor_optimizer = optimizers["actor_optimizer"]
            value_optimizer = optimizers["value_optimizer"]
            actor_optimizer.zero_grad()
            value_optimizer.zero_grad()
            with autocast():
                actor_loss, value_loss, actor_value_logs = model.actor_and_value_loss(
                    global_step, post_state
                )

            grad_norm_actor = model.optimize_actor16(
                actor_loss, actor_optimizer, scaler, global_step, writer
            )
            grad_norm_value = model.optimize_value16(
                value_loss, value_optimizer, scaler, global_step, writer
            )

            scaler.update()

            if global_step % cfg.train.log_every_step == 0:
                with torch.no_grad():
                    logs = {}
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

