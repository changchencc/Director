from collections import defaultdict
from .modules import RSSMWorldModel, DenseDecoder, ActionDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from time import time
import numpy as np


class Dreamer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.world_model = RSSMWorldModel(cfg)

        self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
        self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
        if self.stoch_discrete:
            dense_input_size = (
                cfg.arch.world_model.RSSM.deter_size
                + self.stoch_size * self.stoch_discrete
            )
        else:
            dense_input_size = cfg.arch.world_model.RSSM.deter_size + self.stoch_size

        self.actor = ActionDecoder(
            dense_input_size,
            cfg.env.action_size,
            4,
            cfg.arch.actor.num_units,
            dist=cfg.arch.actor.dist,
            init_std=cfg.arch.actor.init_std,
            act=cfg.arch.actor.act,
            norm=(cfg.arch.actor.norm != "none"),
        )

        self.value = DenseDecoder(
            dense_input_size,
            4,
            cfg.arch.value.num_units,
            (1,),
            act=cfg.arch.value.act,
            norm=(cfg.arch.value.norm != "none"),
        )
        self.slow_value = DenseDecoder(
            dense_input_size,
            4,
            cfg.arch.value.num_units,
            (1,),
            act=cfg.arch.value.act,
            norm=(cfg.arch.value.norm != "none"),
        )

        self.discount = cfg.rl.discount
        self.lambda_ = cfg.rl.lambda_

        self.actor_loss_type = cfg.arch.actor.actor_loss_type
        self.grad_clip = cfg.optimize.grad_clip
        self.action_size = cfg.env.action_size
        self.log_every_step = cfg.train.log_every_step
        self.batch_length = cfg.train.batch_length
        self.grayscale = cfg.env.grayscale
        self.slow_update = 0
        self.n_sample = cfg.train.n_sample
        self.log_grad = cfg.train.log_grad
        self.ent_scale = cfg.loss.ent_scale

        self.r_transform = dict(tanh=torch.tanh, sigmoid=torch.sigmoid,)[
            cfg.rl.r_transform
        ]

    def forward(self):
        raise NotImplementedError

    def write_logs(self, logs, traj, global_step, writer):

        tag = "train"

        rec_img = logs.pop("dec_img")
        gt_img = logs.pop("gt_img")  # B, T, C, H, W

        writer.add_video(
            "train/rec - gt",
            torch.cat([gt_img[:4], rec_img[:4]], dim=-2).clamp(0.0, 1.0).cpu(),
            global_step=global_step,
        )

        for k, v in logs.items():

            if "loss" in k:
                writer.add_scalar(tag + "_loss/" + k, v, global_step=global_step)
            if "grad_norm" in k:
                writer.add_scalar(tag + "_grad_norm/" + k, v, global_step=global_step)
            if "hp" in k:
                writer.add_scalar(tag + "_hp/" + k, v, global_step=global_step)
            if "ACT" in k:
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            writer.add_histogram(
                                tag + "_ACT/" + k + "_" + kk,
                                vv,
                                global_step=global_step,
                            )
                            writer.add_scalar(
                                tag + "_mean_ACT/" + k + "_" + kk,
                                vv.mean(),
                                global_step=global_step,
                            )
                        if isinstance(vv, float):
                            writer.add_scalar(
                                tag + "_ACT/" + k + "_" + kk,
                                vv,
                                global_step=global_step,
                            )
                else:
                    if isinstance(v, torch.Tensor):
                        writer.add_histogram(
                            tag + "_ACT/" + k, v, global_step=global_step
                        )
                        writer.add_scalar(
                            tag + "_mean_ACT/" + k, v.mean(), global_step=global_step
                        )
                    if isinstance(v, float):
                        writer.add_scalar(tag + "_ACT/" + k, v, global_step=global_step)
            if "imag_value" in k:
                writer.add_scalar(
                    tag + "_values/" + k, v.mean(), global_step=global_step
                )
                writer.add_histogram(tag + "_ACT/" + k, v, global_step=global_step)
            if "actor_target" in k:
                writer.add_scalar(tag + "actor_target/" + k, v, global_step=global_step)

        return self.world_model.gen_samples(
            traj, logs, gt_img, rec_img, global_step, writer
        )

    def optimize_actor16(
        self, actor_loss, actor_optimizer, scaler, global_step, writer
    ):

        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        grad_norm_actor = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.grad_clip
        )

        if (global_step % self.log_every_step == 0) and self.log_grad:
            for n, p in self.actor.named_parameters():
                if p.requires_grad:
                    try:
                        writer.add_scalar("grads/" + n, p.grad.norm(2), global_step)
                    except:
                        pdb.set_trace()

        scaler.step(actor_optimizer)

        return grad_norm_actor.item()

    def optimize_value16(
        self, value_loss, value_optimizer, scaler, global_step, writer
    ):

        scaler.scale(value_loss).backward()
        scaler.unscale_(value_optimizer)
        grad_norm_value = torch.nn.utils.clip_grad_norm_(
            self.value.parameters(), self.grad_clip
        )

        if (global_step % self.log_every_step == 0) and self.log_grad:
            for n, p in self.value.named_parameters():
                if p.requires_grad:
                    try:
                        writer.add_scalar("grads/" + n, p.grad.norm(2), global_step)
                    except:
                        pdb.set_trace()

        scaler.step(value_optimizer)

        return grad_norm_value.item()

    # def optimize_actor32(self, actor_loss, actor_optimizer):
    #
    #   actor_loss.backward()
    #   grad_norm_actor = torch.nn.utils.clip_grad_norm_(get_parameters(self.actor_modules), self.grad_clip)
    #   actor_optimizer.step()
    #
    #   return grad_norm_actor.item()
    #
    # def optimize_value32(self, value_loss, value_optimizer):
    #
    #   value_loss.backward()
    #   grad_norm_value = torch.nn.utils.clip_grad_norm_(get_parameters(self.value_modules), self.grad_clip)
    #   value_optimizer.step()
    #
    #   return grad_norm_value.item()

    def world_model_loss(self, global_step, traj):
        return self.world_model.compute_loss(traj, global_step)

    def actor_and_value_loss(self, global_step, post_state):
        self.update_slow_target(global_step)

        self.value.eval()
        self.value.requires_grad_(False)

        (
            imag_feat,
            imag_state,
            imag_action,
            imag_reward,
            imag_disc,
        ) = self.world_model.imagine_ahead(post_state, self.actor)

        target, weights = self.compute_target(
            imag_feat, imag_reward, imag_disc
        )  # B*T, H-1, 1

        actor_dist = self.actor(imag_feat.detach())  # B*T, H
        indices = imag_action.max(-1)[1]
        actor_logprob = actor_dist._categorical.log_prob(indices)

        if self.actor_loss_type == "dynamic":
            actor_loss = target

        elif self.actor_loss_type == "reinforce":
            baseline = self.value(imag_feat[:, :-1]).mean
            advantage = (target - baseline).detach()
            actor_loss = actor_logprob[:, :-1].unsqueeze(2) * advantage

        elif self.actor_loss_type == "both":
            baseline = self.value(imag_feat[:, :-1]).mean
            advantage = (target - baseline).detach()
            actor_loss = actor_logprob[:, :-1].unsqueeze(2) * advantage

            mix = 0.1
            actor_loss = mix * target + (1.0 - mix) * actor_loss

        actor_entropy = actor_dist.entropy()
        ent_scale = self.ent_scale
        actor_loss = ent_scale * actor_entropy[:, :-1].unsqueeze(2) + actor_loss
        actor_loss = -(weights[:, :-1] * actor_loss).mean()

        self.value.train()
        self.value.requires_grad_(True)
        imagine_value = self.value(imag_feat[:, :-1].detach())
        log_prob = -imagine_value.log_prob(target.detach())
        value_loss = weights[:, :-1] * log_prob.unsqueeze(2)
        value_loss = value_loss.mean()

        if global_step % self.log_every_step == 0:
            imagine_dist = self.world_model.dynamic.get_dist(imag_state)
            logs = {
                "value_loss": value_loss.detach().item(),
                "actor_loss": actor_loss.detach().item(),
                "ACT_imag_state": {k: v.detach() for k, v in imag_state.items()},
                "ACT_imag_entropy": imagine_dist.entropy().mean().detach().item(),
                "ACT_actor_entropy": actor_entropy.mean().item(),
                "ACT_actor_logprob": actor_logprob.mean().item(),
                "ACT_action_samples": imag_action.argmax(dim=-1).float().detach(),
                "ACT_image_discount": imag_disc.detach(),
                "ACT_imag_value": imagine_value.mean.detach(),
                "ACT_actor_target": target.mean().detach(),
                "ACT_actor_baseline": baseline.mean().detach(),
            }
            if self.actor_loss_type != "dynamic":
                logs.update(
                    {"ACT_advantage": advantage.detach().mean().item(),}
                )
        else:
            logs = {}

        return actor_loss, value_loss, logs

    def compute_target(self, imag_feat, reward, discount_arr):

        self.slow_value.eval()
        self.slow_value.requires_grad_(False)
        value = self.slow_value(imag_feat).mean  # B*T, H, 1

        target = self.lambda_return(
            reward[:, :-1].float(),
            value[:, :-1].float(),
            discount_arr[:, :-1].float(),
            value[:, -1].float(),
            self.lambda_,
        )

        discount_arr = torch.cat(
            [torch.ones_like(discount_arr[:, :1]), discount_arr[:, :-1]], dim=1
        )
        weights = torch.cumprod(discount_arr, 1).detach()  # B, T 1
        return target, weights

    def policy(
        self, obs, action, gradient_step, state=None, training=True, prior=False
    ):

        obs = obs.unsqueeze(0) / 255.0 - 0.5
        obs_emb = self.world_model.dynamic.img_enc(obs)

        if state is None:
            state = self.world_model.dynamic.init_state(obs.shape[0], obs.device)

        deter, stoch = state["deter"], state["stoch"]
        deter = self.world_model.dynamic.rnn_forward(action, stoch, deter)
        world_state = self.world_model.dynamic.infer_post_stoch(obs_emb, deter)
        rnn_feature = self.world_model.dynamic.get_feature(world_state)
        pred_action_pdf = self.actor(rnn_feature)
        if training:
            action = pred_action_pdf.sample()
            # action, expl_amount = self.exploration(action, gradient_step)
        else:
            action = pred_action_pdf.mean
            index = action.argmax(dim=-1)[0]
            action = torch.zeros_like(action)
            action[..., index] = 1

        return action, world_state

    def lambda_return(
        self, imagine_reward, imagine_value, discount, bootstrap, lambda_
    ):
        """
    https://github.com/juliusfrost/dreamer-pytorch/blob/47bd509ab5cffa95ec613fd788d7ae1fe664ecd5/dreamer/algos/dreamer_algo.py
    """
        # todo: discount v.s. pcont
        # todo: dimension
        # Setting lambda=1 gives a discounted Monte Carlo return.
        # Setting lambda=0 gives a fixed 1-step return.
        next_values = torch.cat([imagine_value[:, 1:], bootstrap[:, None]], 1)
        target = imagine_reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(imagine_reward.shape[1] - 1, -1, -1))

        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:

            inp = target[:, t]
            discount_factor = discount[:, t]

            accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
            outputs.append(accumulated_reward)

        returns = torch.flip(torch.stack(outputs, dim=1), [1])
        return returns

    def update_slow_target(self, global_step):
        with torch.no_grad():
            if self.slow_update % 100 == 0:
                self.slow_value.load_state_dict(self.value.state_dict())

            self.slow_update += 1

