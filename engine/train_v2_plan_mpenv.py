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
from time import sleep
from collections import defaultdict
import torch.multiprocessing as mp


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


def simulate_test(model, test_env, cfg, device):

    model.eval()

    obs = test_env.reset()
    action = torch.zeros(1, cfg.env.action_size).float()
    state = None
    done = False
    goal = None
    K = 0

    with torch.no_grad():
        while not done:
            image = torch.tensor(obs["image"])
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
    test_env = make_env(cfg, writer, "test", datadir, store=True)

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
    goal = None
    K = 0
    action = torch.zeros(1, cfg.env.action_size).float()
    action[0, 0] = 1.0

    while global_step < cfg.total_steps:

        with autocast():
            # print(f'collecting data, global step: {global_step}')
            with torch.no_grad():
                model.eval()
                image = torch.tensor(obs["image"])
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

            model_optimizer, model_scaler = optimizers["model_optimizer"]
            model_optimizer.zero_grad()

            with autocast():
                (
                    model_loss,
                    model_logs,
                    prior_state,
                    post_state,
                ) = model.world_model_loss(global_step, traj)

            pdb.set_trace()
            grad_norm_model = model.world_model.optimize_world_model16(
                model_loss, model_optimizer, model_scaler, global_step, writer
            )

            goal_vae_optimizer, goal_vae_scaler = optimizers["goal_vae_optimizer"]
            goal_vae_optimizer.zero_grad()
            with autocast():
                goal_loss, goal_logs = model.goal_vae_loss(post_state)
            grad_norm_goalvae = model.optimize_goalvae16(
                goal_loss, goal_vae_optimizer, goal_vae_scaler, global_step, writer
            )

            actor_optimizer, actor_scaler = optimizers["actor_optimizer"]
            value_optimizer, value_scaler = optimizers["value_optimizer"]
            mgr_actor_optimizer, mgr_actor_scaler = optimizers["mgr_actor_optimizer"]
            mgr_value_optimizer, mgr_value_scaler = optimizers["mgr_value_optimizer"]
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

            grad_norm_actor = model.optimize_actor16(
                actor_loss, actor_optimizer, actor_scaler, global_step, writer
            )
            grad_norm_value = model.optimize_value16(
                value_loss, value_optimizer, value_scaler, global_step, writer
            )
            if not cfg.arch.worker_only:
                grad_norm_mgr_actor = model.optimize_mgr_actor16(
                    mgr_actor_loss,
                    mgr_actor_optimizer,
                    mgr_actor_scaler,
                    global_step,
                    writer,
                )
                grad_norm_mgr_value = model.optimize_mgr_value16(
                    mgr_value_loss,
                    mgr_value_optimizer,
                    mgr_value_scaler,
                    global_step,
                    writer,
                )

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
                simulate_test(model, test_env, cfg, device)

        if global_step % cfg.train.checkpoint_every_step == 0:
            env_step = count_steps(datadir, cfg)
            checkpointer.save("", model, optimizers, global_step, env_step)

        global_step += 1


class Learner(mp.Process):
    def __init__(self, model, completion, cfg, device, run_step):
        super().__init__()
        self.model = model
        self.completion = completion
        self.cfg = cfg
        self.device = device
        self.run_step = run_step

    def run(self,):

        cfg = self.cfg
        device = self.device

        print("======== Settings ========")
        pprint(cfg)

        model = self.model

        print("======== Model ========")
        pprint(model)

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

        steps = count_steps(datadir, cfg)
        while steps < cfg.arch.prefill:
            print(f"Learner wait {steps} steps.")
            steps = count_steps(datadir, cfg)
            sleep(10)

        train_ds = EnvIterDataset(
            datadir, cfg.train.train_steps, cfg.train.batch_length
        )
        train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=4)
        train_iter = iter(train_dl)
        scaler = GradScaler()
        self.run_step.value = 0

        print(f"collected {steps} steps. Start training...")
        try:
            while global_step < cfg.total_steps:

                model.train()
                model.requires_grad_(True)

                traj = next(train_iter)
                for k, v in traj.items():
                    traj[k] = v.to(device)

                model_optimizer, model_scaler = optimizers["model_optimizer"]
                model_optimizer.zero_grad()

                with autocast():
                    (
                        model_loss,
                        model_logs,
                        prior_state,
                        post_state,
                    ) = model.world_model_loss(global_step, traj)

                grad_norm_model = model.world_model.optimize_world_model16(
                    model_loss, model_optimizer, model_scaler, global_step, writer
                )

                goal_vae_optimizer, goal_vae_scaler = optimizers["goal_vae_optimizer"]
                goal_vae_optimizer.zero_grad()
                with autocast():
                    goal_loss, goal_logs = model.goal_vae_loss(post_state)
                grad_norm_goalvae = model.optimize_goalvae16(
                    goal_loss, goal_vae_optimizer, goal_vae_scaler, global_step, writer
                )

                actor_optimizer, actor_scaler = optimizers["actor_optimizer"]
                value_optimizer, value_scaler = optimizers["value_optimizer"]
                mgr_actor_optimizer, mgr_actor_scaler = optimizers[
                    "mgr_actor_optimizer"
                ]
                mgr_value_optimizer, mgr_value_scaler = optimizers[
                    "mgr_value_optimizer"
                ]
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

                grad_norm_actor = model.optimize_actor16(
                    actor_loss, actor_optimizer, actor_scaler, global_step, writer
                )
                grad_norm_value = model.optimize_value16(
                    value_loss, value_optimizer, value_scaler, global_step, writer
                )
                if not cfg.arch.worker_only:
                    grad_norm_mgr_actor = model.optimize_mgr_actor16(
                        mgr_actor_loss,
                        mgr_actor_optimizer,
                        mgr_actor_scaler,
                        global_step,
                        writer,
                    )
                    grad_norm_mgr_value = model.optimize_mgr_value16(
                        mgr_value_loss,
                        mgr_value_optimizer,
                        mgr_value_scaler,
                        global_step,
                        writer,
                    )

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
                    print(f"Learner run_step: {self.run_step.value}")

                if global_step % cfg.train.checkpoint_every_step == 0:
                    env_step = count_steps(datadir, cfg)
                    checkpointer.save("", model, optimizers, global_step, env_step)

                global_step += 1
                self.run_step.value += 1

            self.completion.set()

        except KeyboardInterrupt:
            self.completion.set()
        except Exception as e:
            self.completion.set()
            raise e


class Actor(mp.Process):
    def __init__(self, idx, model, cfg, device, run_step, completion, actor_steps):
        super().__init__()
        self.idx = idx
        self.model = model
        self.cfg = cfg
        self.device = device
        self.completion = completion
        self.run_step = run_step
        self.actor_steps = actor_steps

    def run(self):

        cfg = self.cfg
        device = self.device
        model = self.model

        writer = SummaryWriter(
            log_dir=os.path.join(
                cfg.logdir, cfg.exp_name, cfg.env.name, cfg.run_id, f"actor-{self.idx}"
            ),
            flush_secs=30,
        )

        datadir = os.path.join(
            cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, "train_episodes"
        )
        test_datadir = os.path.join(
            cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id, "test_episodes"
        )
        train_env = make_env(cfg, writer, "train", datadir, store=True)
        test_env = make_env(cfg, writer, "test", datadir, store=True)
        verbose = True

        try:
            train_env.reset()
            steps = count_steps(datadir, cfg)
            while steps < cfg.arch.prefill:
                action = train_env.sample_random_action()
                next_obs, reward, done, info = train_env.step(action)
                steps = count_steps(datadir, cfg)
                if done:
                    train_env.reset()
            # fill in length of 5000 frames
            model.eval()
            obs = train_env.reset()
            state = None
            goal = None
            K = 0
            action = torch.zeros(1, cfg.env.action_size).float()
            action[0, 0] = 1.0
            global_step = steps
            while not self.completion.is_set():

                total_actor_step = sum([a.value for a in self.actor_steps])
                if total_actor_step > self.run_step.value * self.cfg.train.train_every:
                    if verbose:
                        print(
                            f"Actor-{self.idx} waiting for model updating: learner runs {self.run_step.value}, actors collect {total_actor_step} steps"
                        )
                    sleep(10)
                    continue

                else:
                    verbose = False
                    with torch.no_grad():
                        with autocast():
                            image = torch.tensor(obs["image"])
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
                        global_step += 1
                        self.actor_steps[self.idx].value += 1
                        if done:
                            obs = train_env.reset()
                            state = None
                            goal = None
                            K = 0
                            action = torch.zeros(1, cfg.env.action_size).float()
                # evaluate RL
                if global_step % (cfg.train.log_every_step * 100) == 0:
                    print(
                        f"Actor-{self.idx}: learner runs {self.run_step.value}, actors collect {total_actor_step} steps"
                    )

                if global_step % cfg.train.eval_every_step == 0:
                    with autocast():
                        simulate_test(model, test_env, cfg, device)

        except KeyboardInterrupt:
            train_env.close()
            test_env.close()
        except Exception as e:
            train_env.close()
            test_env.close()
            raise e

