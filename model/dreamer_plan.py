from collections import defaultdict
from .modules_plan import (
    RSSMWorldModel,
    DenseDecoder,
    ActionDecoder,
    GoalVAE,
    GoalEncoder,
    ManagerValue,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from time import time
import numpy as np
from .utils import RunningMeanStd
from torch.distributions import OneHotCategorical
import torchvision.utils as vutils


class DreamerPlan(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.world_model = RSSMWorldModel(cfg)

        self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
        self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
        self.deter_size = cfg.arch.world_model.RSSM.deter_size
        self.worker_only = cfg.arch.worker_only
        if self.stoch_discrete:
            dense_input_size = self.deter_size + self.stoch_size * self.stoch_discrete
            if not self.worker_only:
                dense_input_size = (
                    dense_input_size + self.deter_size  # conditioned on goal
                )
        else:
            dense_input_size = self.deter_size + self.stoch_size
            if not self.worker_only:
                dense_input_size = (
                    dense_input_size + self.deter_size
                )  # conditioned on goal

        self.actor = ActionDecoder(
            cfg,
            dense_input_size,
            cfg.env.action_size,
            cfg.arch.actor.layers,
            cfg.arch.actor.num_units,
            dist=cfg.arch.actor.dist,
            init_std=cfg.arch.actor.init_std,
            act=cfg.arch.actor.act,
            norm=(cfg.arch.actor.norm != "none"),
        )

        self.value = DenseDecoder(
            dense_input_size,
            cfg.arch.value.layers,
            cfg.arch.value.num_units,
            (1,),
            act=cfg.arch.value.act,
            norm=(cfg.arch.value.norm != "none"),
        )
        self.slow_value = DenseDecoder(
            dense_input_size,
            cfg.arch.value.layers,
            cfg.arch.value.num_units,
            (1,),
            act=cfg.arch.value.act,
            norm=(cfg.arch.value.norm != "none"),
        )
        self.goal_vae = GoalVAE(cfg)

        if self.stoch_discrete:
            mgr_input = self.stoch_size * self.stoch_discrete + self.deter_size
        else:
            mgr_input = self.stoch_size + self.deter_size
        self.mgr_actor = GoalEncoder(
            mgr_input,
            cfg.arch.manager.actor.action_num,
            cfg.arch.manager.actor.action_size,
            cfg.arch.manager.actor.layers,
            cfg.arch.manager.actor.num_units,
            dist=cfg.arch.manager.actor.dist,
            act=cfg.arch.manager.actor.act,
            norm=(cfg.arch.manager.actor.norm != "none"),
        )
        self.mgr_value = ManagerValue(cfg.arch.manager, cfg)
        self.mgr_slow_value = ManagerValue(cfg.arch.manager, cfg)
        self.K = cfg.arch.manager.K
        self.H = cfg.arch.H
        self.cum_intri = cfg.arch.manager.cum_intri
        self.norm_target = cfg.arch.norm_target
        if self.norm_target:
            self.extri_target_rms = RunningMeanStd()
            self.intri_target_rms = RunningMeanStd()
        self.norm_reward = cfg.arch.norm_reward
        if self.norm_reward:
            self.extri_reward_rms = RunningMeanStd()
            self.intri_reward_rms = RunningMeanStd()
        self.mgr_exp = cfg.loss.mgr_exp
        self.mgr_extr = cfg.loss.mgr_extr

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
        self.mgr_ent_scale = cfg.loss.mgr_ent_scale
        self.goal_rec_scale = cfg.loss.goal_rec_scale
        self.debug_worker = cfg.arch.debug_worker

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
        self.plot_imagine(logs, writer, global_step)

        if self.norm_target:
            writer.add_scalar(
                tag + "_RMS_Target/mgr_extr_rms",
                np.sqrt(self.extri_target_rms.var),
                global_step=global_step,
            )
            writer.add_scalar(
                tag + "_RMS_Target/mgr_intri_rms",
                np.sqrt(self.intri_target_rms.var),
                global_step=global_step,
            )
        if self.norm_reward:
            writer.add_scalar(
                tag + "_RMS_Reward/mgr_extr_rms",
                np.sqrt(self.extri_reward_rms.var),
                global_step=global_step,
            )
            writer.add_scalar(
                tag + "_RMS_Reward/mgr_intri_rms",
                np.sqrt(self.intri_reward_rms.var),
                global_step=global_step,
            )

        for k, v in logs.items():

            if "Loss" in k:
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
                        writer.add_scalar(
                            tag + "_max_ACT/" + k, v.max(), global_step=global_step
                        )
                        writer.add_scalar(
                            tag + "_min_ACT/" + k, v.min(), global_step=global_step
                        )
                    if isinstance(v, float):
                        writer.add_scalar(tag + "_ACT/" + k, v, global_step=global_step)
        return self.world_model.gen_samples(
            traj, logs, gt_img, rec_img, global_step, writer
        )

    def plot_imagine(self, logs, writer, global_step):
        imag_state = logs["ACT_imag_state"]  # n_sample, H, L
        imag_goal = logs["ACT_mgr_imag_goal"][:4]

        imag_feature = self.world_model.dynamic.get_feature(imag_state)[:4]
        imag_img = self.world_model.img_dec(imag_feature).mean + 0.5

        goal_latent_state = self.world_model.dynamic.infer_prior_stoch(imag_goal)
        goal_feature = self.world_model.dynamic.get_feature(goal_latent_state)
        goal_img = self.world_model.img_dec(goal_feature).mean + 0.5
        mask = goal_img.new_zeros(goal_img.shape)
        mask[:, :: self.K] = 1.0
        goal_img = mask * goal_img

        writer.add_video(
            "train/imag - goal",
            torch.cat([imag_img, goal_img], dim=-2).clamp(0.0, 1.0).cpu(),
            global_step=global_step,
        )
        imgs = (
            torch.stack([imag_img, goal_img], dim=1)
            .clamp(0.0, 1.0)
            .cpu()
            .flatten(end_dim=2)
        )
        writer.add_image(
            "train/imag - goal",
            vutils.make_grid(imgs, nrow=self.H, pad_value=1),
            global_step=global_step,
        )
        # todo: plot reward

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
                        writer.add_scalar(
                            "grads/worker_actor_" + n, p.grad.norm(2), global_step
                        )
                    except:
                        pdb.set_trace()

        scaler.step(actor_optimizer)
        scaler.update()

        return grad_norm_actor.item()

    def optimize_mgr_actor16(
        self, actor_loss, actor_optimizer, scaler, global_step, writer
    ):

        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        grad_norm_actor = torch.nn.utils.clip_grad_norm_(
            self.mgr_actor.parameters(), self.grad_clip
        )

        if (global_step % self.log_every_step == 0) and self.log_grad:
            for n, p in self.mgr_actor.named_parameters():
                if p.requires_grad:
                    try:
                        writer.add_scalar(
                            "grads/mgr_actor_" + n, p.grad.norm(2), global_step
                        )
                    except:
                        pdb.set_trace()

        scaler.step(actor_optimizer)
        scaler.update()

        return grad_norm_actor.item()

    def optimize_mgr_value16(
        self, value_loss, value_optimizer, scaler, global_step, writer
    ):

        scaler.scale(value_loss).backward()
        scaler.unscale_(value_optimizer)
        grad_norm_value = torch.nn.utils.clip_grad_norm_(
            self.mgr_value.parameters(), self.grad_clip
        )

        if (global_step % self.log_every_step == 0) and self.log_grad:
            for n, p in self.mgr_value.named_parameters():
                if p.requires_grad:
                    try:
                        writer.add_scalar(
                            "grads/mgr_value_" + n, p.grad.norm(2), global_step
                        )
                    except:
                        pdb.set_trace()

        scaler.step(value_optimizer)
        scaler.update()

        return grad_norm_value.item()

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
                        writer.add_scalar(
                            "grads/worker_value_" + n, p.grad.norm(2), global_step
                        )
                    except:
                        pdb.set_trace()

        scaler.step(value_optimizer)
        scaler.update()

        return grad_norm_value.item()

    def optimize_goalvae16(
        self, goal_vae_loss, goal_vae_optimizer, scaler, global_step, writer
    ):

        scaler.scale(goal_vae_loss).backward()
        scaler.unscale_(goal_vae_optimizer)
        grad_norm_value = torch.nn.utils.clip_grad_norm_(
            self.goal_vae.parameters(), self.grad_clip
        )

        if (global_step % self.log_every_step == 0) and self.log_grad:
            for n, p in self.goal_vae.named_parameters():
                if p.requires_grad:
                    try:
                        writer.add_scalar(
                            "grads/goal_vae_" + n, p.grad.norm(2), global_step
                        )
                    except:
                        pdb.set_trace()

        scaler.step(goal_vae_optimizer)
        scaler.update()

        return grad_norm_value.item()

    def world_model_loss(self, global_step, traj):
        return self.world_model.compute_loss(traj, global_step)

    def goal_vae_loss(self, state):
        """
        state: B, T, S
        """
        s_t = state["deter"].detach()
        z_dist = self.goal_vae.enc(s_t)
        z = z_dist.sample()  # B, T, K, L
        z = z + z_dist.mean - z_dist.mean.detach()
        goal_dist = self.goal_vae.dec(z.flatten(start_dim=-2))  # B, S

        kl_loss = self.goal_vae.kl_loss(z_dist).sum(-1)
        kl_loss = kl_loss.mean()

        rec_loss = -goal_dist.log_prob(s_t).mean()  # B, T
        goal_mse = ((s_t - goal_dist.mean) ** 2).sum(-1).mean()
        goal_loss = kl_loss + self.goal_rec_scale * rec_loss
        logs = {
            "Loss_goal_rec_loss": rec_loss.detach().item(),
            "Loss_goal_kl_loss": kl_loss.detach().item(),
            "ACT_goal_mse_loss": goal_mse.detach().item(),
            "ACT_goalvae_latent_entropy": z_dist.entropy().sum(-1).mean().detach(),
            "ACT_goalvae_latent_sample": z_dist.mean.argmax(-1).float().detach(),
        }
        return goal_loss, logs

    def compute_manager_reward(self, imag_reward, imag_state):
        """
        imag_reward: B, H, 1
        imag_goal: B, H, S
        imag_state: B, H, S
        discount_arr: B, H, 1
        """

        B, T = imag_reward.shape[:2]

        def compute_mgr_extri_reward(task_reward):
            cum_r = task_reward.reshape(B, -1, self.K).sum(-1)
            return cum_r

        def compute_mgr_intri_reward(imag_s):

            z_dist = self.goal_vae.enc(imag_s.detach())
            z = z_dist.sample()  # B, T, K, L
            goal_dist = self.goal_vae.dec(z.flatten(start_dim=-2))  # B, S
            rec_s = goal_dist.mean
            intri_r = ((rec_s - imag_s) ** 2).sum(-1)

            if self.cum_intri:
                intri_r = intri_r.reshape(B, -1, self.K).sum(-1)
            else:
                intri_r = intri_r[:, :: self.K]

            return intri_r

        mgr_extri_reward = compute_mgr_extri_reward(imag_reward)
        mgr_intri_reward = compute_mgr_intri_reward(imag_state)

        if self.norm_reward:

            mean, std, count = (
                mgr_extri_reward.mean().cpu().numpy(),
                mgr_extri_reward.std().cpu().numpy(),
                mgr_extri_reward.shape[0],
            )
            self.extri_reward_rms.update_from_moments(mean, std ** 2, count)
            mgr_extri_reward = mgr_extri_reward / np.sqrt(self.extri_reward_rms.var)

            mean, std, count = (
                mgr_intri_reward.mean().cpu().numpy(),
                mgr_intri_reward.std().cpu().numpy(),
                mgr_intri_reward.shape[0],
            )
            self.intri_reward_rms.update_from_moments(mean, std ** 2, count)
            mgr_intri_reward = mgr_intri_reward / np.sqrt(self.intri_reward_rms.var)

        return mgr_extri_reward.unsqueeze(-1), mgr_intri_reward.unsqueeze(-1)

    def compute_worker_reward(self, imag_goal, imag_state):
        """
        imag_goal: B, H, S
        imag_goal: B, H, S
        """
        g_norm = torch.linalg.norm(imag_goal, dim=-1)
        s_norm = torch.linalg.norm(imag_state, dim=-1)

        norm = torch.maximum(g_norm, s_norm)  # B, H, S

        normed_g = imag_goal / norm.unsqueeze(-1)
        normed_s = imag_state / norm.unsqueeze(-1)

        reward = torch.matmul(normed_g, normed_s.transpose(-1, -2))  # B, H, H
        reward = reward[:, :: self.K]
        reward = torch.cat([reward[:, 0, : self.K], reward[:, 1, self.K :]], dim=-1)

        return reward.unsqueeze(-1)

    def compute_mgr_and_worker_reward(self, imag_g, imag_state, imag_r):

        # todo: not detaching state for worker reward
        worker_r = self.compute_worker_reward(
            imag_g[:, :-1], imag_state["deter"][:, 1:].detach()
        )
        mgr_ext_r, mgr_intr_r = self.compute_manager_reward(
            imag_r[:, 1:], imag_state["deter"][:, 1:].detach()
        )

        return worker_r, mgr_ext_r, mgr_intr_r

    def actor_and_value_loss(self, global_step, post_state):
        self.update_slow_target(global_step)

        self.value.eval()
        self.value.requires_grad_(False)
        self.mgr_value.eval()
        self.mgr_value.requires_grad_(False)
        self.goal_vae.eval()
        self.goal_vae.requires_grad_(False)
        self.mgr_actor.eval()
        self.mgr_actor.requires_grad_(False)

        (
            imag_feat,
            imag_state,
            imag_action,
            imag_reward,
            imag_disc,
            imag_goal,
            imag_mgr_action,
        ) = self.world_model.imagine_ahead(post_state, self.actor, self.manager_policy)

        if not self.worker_only:
            with torch.no_grad():
                """
                worker_r: starts from r_{t+1}
                """
                worker_r, mgr_ext_r, mgr_intri_r = self.compute_mgr_and_worker_reward(
                    imag_goal.detach(), imag_state, imag_reward.detach(),
                )
                if self.debug_worker:
                    worker_r += imag_reward[:, 1:]
        else:
            worker_r = imag_reward[:, 1:]

        self.slow_value.eval()
        self.slow_value.requires_grad_(False)
        if self.worker_only:
            worker_imag_feat = imag_feat
        else:
            worker_imag_feat = torch.cat([imag_feat, imag_goal], dim=-1)
        worker_value = self.slow_value(worker_imag_feat).mean  # B*T, H, 1

        # if not self.worker_only:
        # target_1, weights_1 = self.compute_target(
        #     worker_r[:, : self.K],
        #     imag_disc[:, : self.K + 1],
        #     worker_value[:, : self.K + 1],
        # )  # B*T, H-1, 1
        # target_2, weights_2 = self.compute_target(
        #     worker_r[:, self.K :],
        #     imag_disc[:, self.K :],
        #     worker_value[:, self.K :],
        # )  # B*T, H-1, 1
        # target = torch.cat([target_1, target_2], dim=1)
        # weights = torch.cat([weights_1[:, :-1], weights_2], dim=1)
        # else:
        target, weights = self.compute_target(
            worker_r, imag_disc, worker_value
        )  # B*T, H-1, 1

        if self.actor_loss_type == "reinforce":
            actor_dist = self.actor(imag_feat.detach(), imag_goal.detach())  # B*T, H
            indices = imag_action.max(-1)[1]
            actor_logprob = actor_dist._categorical.log_prob(indices)

            baseline = self.value(worker_imag_feat[:, :-1]).mean
            advantage = (target - baseline).detach()
            actor_loss = actor_logprob[:, :-1].unsqueeze(2) * advantage

        elif self.actor_loss_type == "dynamic":
            actor_loss = target
        else:
            raise NotImplemented

        actor_entropy = actor_dist.entropy()
        ent_scale = self.ent_scale
        actor_loss = ent_scale * actor_entropy[:, :-1].unsqueeze(2) + actor_loss
        actor_loss = -(weights[:, :-1] * actor_loss).mean()

        self.value.train()
        self.value.requires_grad_(True)
        imagine_value = self.value(worker_imag_feat[:, :-1].detach())
        log_prob = -imagine_value.log_prob(target.detach())
        value_loss = weights[:, :-1] * log_prob.unsqueeze(2)
        value_loss = value_loss.mean()

        # mgr loss

        # worker only is used for sanity check
        if not self.worker_only:
            self.mgr_slow_value.eval()
            self.mgr_slow_value.requires_grad_(False)
            mgr_extr_value, mgr_intri_value = self.mgr_slow_value(
                imag_feat[:, :: self.K]
            )  # B*T, H, 1
            mgr_extr_value = mgr_extr_value.mean
            mgr_intri_value = mgr_intri_value.mean

            mgr_extri_target, mgr_weights = self.compute_target(
                mgr_ext_r, imag_disc[:, :: self.K], mgr_extr_value
            )  # B*T, H-1, 1

            mgr_intri_target, mgr_weights = self.compute_target(
                mgr_intri_r, imag_disc[:, :: self.K], mgr_intri_value
            )  # B*T, H-1, 1
            if self.norm_target:
                mean, std, count = (
                    mgr_extri_target.detach().mean().cpu().numpy(),
                    mgr_extri_target.detach().std().cpu().numpy(),
                    mgr_extri_target.shape[0],
                )
                self.extri_target_rms.update_from_moments(mean, std ** 2, count)
                mgr_extri_target = mgr_extri_target / np.sqrt(self.extri_target_rms.var)

                mean, std, count = (
                    mgr_intri_target.detach().mean().cpu().numpy(),
                    mgr_intri_target.detach().std().cpu().numpy(),
                    mgr_intri_target.shape[0],
                )
                self.intri_target_rms.update_from_moments(mean, std ** 2, count)
                mgr_intri_target = mgr_intri_target / np.sqrt(self.intri_target_rms.var)

            # mgr actor loss
            self.mgr_actor.train()
            self.mgr_actor.requires_grad_(True)
            mgr_actor_dist = self.mgr_actor(imag_feat[:, :: self.K].detach())  # B*T, H
            mgr_indices = imag_mgr_action[:, :: self.K].max(-1)[1]
            mgr_actor_logprob = mgr_actor_dist._categorical.log_prob(mgr_indices).sum(
                -1
            )

            mgr_extr_value, mgr_intri_value = self.mgr_value(
                imag_feat[:, :: self.K][:, :-1]
            )  # B*T, H, 1
            mgr_extr_baseline = mgr_extr_value.mean
            mgr_intri_baseline = mgr_intri_value.mean
            mgr_ext_adv = mgr_extri_target - mgr_extr_baseline
            mgr_intri_adv = mgr_intri_target - mgr_intri_baseline

            mgr_advantage = (
                self.mgr_exp * mgr_intri_adv + self.mgr_extr * mgr_ext_adv
            ).detach()
            mgr_actor_loss = mgr_actor_logprob[:, :-1].unsqueeze(2) * mgr_advantage

            mgr_actor_entropy = mgr_actor_dist.entropy().sum(-1)
            mgr_actor_loss = (
                self.mgr_ent_scale * mgr_actor_entropy[:, :-1].unsqueeze(2)
                + mgr_actor_loss
            )
            mgr_actor_loss = -(mgr_weights[:, :-1] * mgr_actor_loss).mean()

            self.mgr_value.train()
            self.mgr_value.requires_grad_(True)

            mgr_extr_value, mgr_intri_value = self.mgr_value(
                imag_feat[:, :: self.K][:, :-1].detach()
            )  # B*T, H, 1
            mgr_extr_log_prob = -mgr_extr_value.log_prob(mgr_extri_target.detach())
            mgr_extr_value_loss = mgr_weights[:, :-1] * mgr_extr_log_prob.unsqueeze(2)
            mgr_extr_value_loss = mgr_extr_value_loss.mean()

            mgr_intri_log_prob = -mgr_intri_value.log_prob(mgr_intri_target.detach())
            mgr_intri_value_loss = mgr_weights[:, :-1] * mgr_intri_log_prob.unsqueeze(2)
            mgr_intri_value_loss = mgr_intri_value_loss.mean()
            mgr_value_loss = mgr_extr_value_loss + mgr_intri_value_loss
        else:
            mgr_actor_loss = 0.0
            mgr_value_loss = 0.0

        if global_step % self.log_every_step == 0:
            imagine_dist = self.world_model.dynamic.get_dist(imag_state)
            logs = {
                "Loss_value_loss": value_loss.detach().item(),
                "Loss_actor_loss": actor_loss.detach().item(),
                "ACT_imag_state": {k: v.detach() for k, v in imag_state.items()},
                "ACT_imag_entropy": imagine_dist.entropy().mean().detach().item(),
                "ACT_actor_entropy": actor_entropy.mean().item(),
                "ACT_actor_logprob": actor_logprob.mean().item(),
                "ACT_action_samples": imag_action.argmax(dim=-1).float().detach(),
                "ACT_image_discount": imag_disc.detach(),
                "ACT_imag_value": imagine_value.mean.detach(),
                "ACT_actor_target": target.mean().detach(),
                "ACT_actor_baseline": baseline.mean().detach(),
                "ACT_actor_reward": worker_r.mean().detach(),
            }
            if self.actor_loss_type != "dynamic":
                logs.update(
                    {"ACT_advantage": advantage.detach().mean().item(),}
                )
            if not self.worker_only:
                logs.update(
                    {
                        "Loss_mgr_actor_loss": mgr_actor_loss.detach().item(),
                        "ACT_mgr_actor_entropy": mgr_actor_entropy.mean().detach(),
                        "ACT_mgr_actor_logprob": mgr_actor_logprob.mean().detach(),
                        "Loss_mgr_ext_value_loss": mgr_extr_value_loss.detach().item(),
                        "Loss_mgr_intri_value_loss": mgr_intri_value_loss.detach().item(),
                        "ACT_mgr_ext_value": mgr_extr_value.mean.detach(),
                        "ACT_mgr_intri_value": mgr_intri_value.mean.detach(),
                        "ACT_mgr_ext_target": mgr_extri_target.mean().detach(),
                        "ACT_mgr_intri_target": mgr_intri_target.mean().detach(),
                        "ACT_mgr_extr_reward": mgr_ext_r.mean().detach(),
                        "ACT_mgr_intri_reward": mgr_intri_r.mean().detach(),
                        "ACT_mgr_extr_advantage": mgr_ext_adv.mean().detach(),
                        "ACT_mgr_intri_advantage": mgr_intri_adv.mean().detach(),
                        "ACT_mgr_weights": mgr_weights.mean().detach(),
                        "ACT_mgr_imag_goal": imag_goal.detach(),
                        "ACT_mgr_imag_action": mgr_indices.float().detach(),
                    }
                )
        else:
            logs = {}

        return actor_loss, value_loss, mgr_actor_loss, mgr_value_loss, logs

    def compute_target(self, reward, discount_arr, value):

        target = self.lambda_return(
            reward.float(),
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

    def manager_policy(self, state, sample=False):

        # todo:
        # if self.mgr_actor_include_deter:
        rnn_feature = self.world_model.dynamic.get_feature(state).detach()
        z_dist = self.mgr_actor(rnn_feature)

        if sample:
            z = z_dist.sample()  # B, K, L
        else:
            probs = z_dist.probs
            index = probs.argmax(-1)
            index = index.unsqueeze(-1).expand(probs.shape)
            z = probs.new_zeros(probs.shape)
            z.scatter_(-1, index, 1.0)

        # if self.stoch_discrete:
        #     goal_dist = self.goal_vae.dec(z.flatten(start_dim=-2))  # B, S
        #     probs = goal_dist.probs  # B, S, L, L
        #     index = probs.argmax(-1)
        #     index = index.unsqueeze(-1).expand(probs.shape)
        #     goal = probs.new_zeros(probs.shape)
        #     goal.scatter_(-1, index, 1.0)
        #     goal = goal.flatten(start_dim=-2)
        goal = self.goal_vae.dec(z.flatten(start_dim=-2)).mean  # B, S

        return goal, z

    def policy(
        self, obs, action, goal, state=None, training=True, sample_goal=False,
    ):

        obs = obs.unsqueeze(0) / 255.0 - 0.5
        obs_emb = self.world_model.dynamic.img_enc(obs)

        if state is None:
            state = self.world_model.dynamic.init_state(obs.shape[0], obs.device)
        if sample_goal:
            goal, _ = self.manager_policy(state, sample=training)

        deter, stoch = state["deter"], state["stoch"]
        deter = self.world_model.dynamic.rnn_forward(action, stoch, deter)
        world_state = self.world_model.dynamic.infer_post_stoch(obs_emb, deter)
        rnn_feature = self.world_model.dynamic.get_feature(world_state)
        pred_action_pdf = self.actor(rnn_feature, goal)
        if training:
            action = pred_action_pdf.sample()
            # action, expl_amount = self.exploration(action, gradient_step)
        else:
            action = pred_action_pdf.mean
            index = action.argmax(dim=-1)[0]
            action = torch.zeros_like(action)
            action[..., index] = 1

        return action, world_state, goal

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
                self.mgr_slow_value.intri_value.load_state_dict(
                    self.mgr_value.intri_value.state_dict()
                )
                self.mgr_slow_value.extri_value.load_state_dict(
                    self.mgr_value.extri_value.state_dict()
                )

            self.slow_update += 1

