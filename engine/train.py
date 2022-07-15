import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils
from utils import Checkpointer
from solver import get_optimizer
from envs import make_env, count_steps
from data import EnvIterDataset
from torch.utils.data import DataLoader
from vis.vis_logger import log_vis
import os
import numpy as np
from pprint import pprint
import pdb
import torch.autograd.profiler as profiler
from time import time

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

def train(model, cfg, device):
  print("======== Settings ========")
  pprint(cfg)
  input()

  # train_dl = get_dataloader(cfg.data.dataset, 'train', cfg.train.batch_size, cfg.data.data_root_prefix)
  # val_dl = get_dataloader(cfg.data.dataset, 'val', cfg.train.batch_size, cfg.data.data_root_prefix)

  model = model.to(device)

  print("======== Model ========")
  pprint(model)
  input()

  optimizers = get_optimizer(cfg, model)
  checkpointer = Checkpointer(os.path.join(cfg.checkpoint.checkpoint_dir, cfg.exp_name, cfg.env.name, cfg.run_id), max_num=cfg.checkpoint.max_num)

  if cfg.resume:
    checkpoint = checkpointer.load(cfg.resume_ckpt)

    if checkpoint:
      model.load_state_dict(checkpoint['model'])
      for k, v in optimizers.items():
        v.load_state_dict(checkpoint[k])
      env_step = checkpoint['env_step']
      global_step = checkpoint['global_step']

    else:
      env_step = 0
      global_step = 0

  else:
    env_step = 0
    global_step = 0

  num_gpus = torch.cuda.device_count()
  if num_gpus > 1:
    model = nn.DataParallel(model)

  writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, cfg.exp_name, cfg.env.name, cfg.run_id), flush_secs=30)

  datadir = os.path.join(cfg.data.datadir, cfg.exp_name, cfg.env.name, cfg.run_id)
  train_env = make_env(cfg, writer, 'train', datadir, store=True)
  test_env = make_env(cfg, writer, 'test', datadir, store=False)

  # fill in length of 5000 frames
  train_env.reset()
  steps = count_steps(datadir, cfg)
  length = 0
  while steps < cfg.arch.prefill / cfg.env.action_repeat:
    action = train_env.sample_random_action()
    next_observation, reward, done, info = train_env.step(action)
    length += 1
    steps += done * length
    length = length * (1. - done)
    if done:
      train_env.reset()

  steps = count_steps(datadir, cfg)
  print(f'collecting {steps} env_steps. Start training...')
  train_ds = EnvIterDataset(datadir, cfg.train.train_steps, cfg.train.batch_length)
  train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=16)
  train_iter = iter(train_dl)
  while global_step < 1000000:
    gradient_step = 0.
    while gradient_step < cfg.train.train_steps:
      model.train()

      traj = next(train_iter)

      for k, v in traj.items():
        traj[k] = v.to(device)

      steps = count_steps(datadir, cfg)
      logs = model(traj, optimizers, global_step, steps, writer)

      rec_img = logs.pop('dec_img')
      gt_img = logs.pop('gt_img') # B, T, C, H, W

      if global_step % cfg.train.log_every_step == 0:

        writer.add_video('train/rec - gt',
                            torch.cat([gt_img[:4], rec_img[:4]], dim=-2).clamp(0., 1.),
                            global_step=global_step)

        for k, v in logs.items():

          if 'loss' in k:
            writer.add_scalar('train_loss/' + k, v, global_step=global_step)
          if 'grad_norm' in k:
            writer.add_scalar('train_grad_norm/' + k, v, global_step=global_step)
          if 'entropy' in k:
            writer.add_scalar('train_entropy/' + k, v, global_step=global_step)
        writer.flush()

      # evaluate
      if global_step % cfg.train.evaluate_every_step == 0:

        # generation
        gt_img = gt_img[:6]
        rec_img = rec_img[:6, :5] # B, T, C, H, W
        action = traj['action'][:6, 5:]
        with torch.no_grad():
          prev_stoch = logs['prior_state']['stoch'][:6, 4].detach()
          prev_deter = logs['prior_state']['deter'][:6, 4].detach()
          rnn_features = []
          for t in range(cfg.train.batch_length - 5):
            prev_deter = model.dynamic.rnn_forward(action[:, t], prev_stoch, prev_deter)
            prior = model.dynamic.infer_prior_stoch(prev_deter)
            rnn_features.append(model.dynamic.get_feature(prior))

          rnn_features = torch.stack(rnn_features, dim=1) # B, T-5, H
          pred_imgs = model.img_dec(rnn_features).mean + 0.5 # B, T-5, C, H, W

        imgs_act = torch.cat([rec_img, pred_imgs], dim=1) # B, T, C, H, W
        err = gt_img - imgs_act
        imgs = torch.cat([gt_img, imgs_act, err], dim=3).cpu()
        if cfg.env.grayscale:
          imgs = imgs.expand(-1, -1, 3, -1, -1)
        color_bar = torch.zeros([*imgs.shape[:4]] + [5])
        color_bar[:, :5, 0] = 1
        color_bar[:, 5:, 1] = 1
        final_vis = torch.cat([color_bar, imgs], dim=4) # B, T, C, H, W+5
        writer.add_video('test/gt - rec - gen',
                         final_vis.clamp(0., 1.),
                         global_step=global_step)

        # RL
        obs = test_env.reset()
        action = torch.zeros(1, cfg.env.action_size).float()
        state = None
        done = False

        with torch.no_grad():
          while not done:
            image = torch.tensor(obs['image'])
            action, state, expl_amount = model.policy(image.to(device), action.to(device), global_step, state, training=False)
            next_obs, reward, done, info = test_env.step(action[0].cpu().numpy())
            obs = next_obs

      if global_step % cfg.train.checkpoint_every_step == 0:
        env_step = count_steps(datadir, cfg)
        checkpointer.save('', model, optimizers, global_step, env_step)

      gradient_step += 1
      global_step += 1


    # collecting data
    print(f'{global_step} gradient steps, collecting data...')
    obs = train_env.reset()
    action = torch.zeros(1, cfg.env.action_size).float()
    state = None
    done = False

    with torch.no_grad():
      while not done:
        image = torch.tensor(obs['image'])
        action, state, expl_amount = model.policy(image.to(device), action.to(device), global_step, state)
        next_obs, reward, done, info = train_env.step(action[0].cpu().numpy())
        obs = next_obs

        # lr = anneal_learning_rate(global_step, cfg)
        # for param_group in optimizer.param_groups:
        #   param_group['lr'] = lr

      writer.add_scalar('hy/expl_amount', expl_amount, global_step=global_step)
      writer.flush()

