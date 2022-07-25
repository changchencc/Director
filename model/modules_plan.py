from torch.distributions.bernoulli import Bernoulli
from torch.distributions.one_hot_categorical import OneHotCategorical
from .utils import Conv2DBlock, ConvTranspose2DBlock, Linear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, Bernoulli
from torch.distributions import kl_divergence
from .distributions import ContDist, SafeTruncatedNormal
from .utils import (
    Conv2DBlock,
    ConvTranspose2DBlock,
    Linear,
    MLP,
    GRUCell,
    LayerNormGRUCell,
    LayerNormGRUCellV2,
)
from collections import defaultdict
import numpy as np
import pdb
from time import time


class RSSMWorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dynamic = RSSM(cfg)
        self.img_dec = ImgDecoder(cfg)

        self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
        self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
        self.pcont = cfg.arch.world_model.RSSM.pcont
        if self.stoch_discrete:
            dense_input_size = (
                cfg.arch.world_model.RSSM.deter_size
                + self.stoch_size * self.stoch_discrete
            )
        else:
            dense_input_size = cfg.arch.world_model.RSSM.deter_size + self.stoch_size
        self.reward = DenseDecoder(
            dense_input_size,
            cfg.arch.world_model.reward.layers,
            cfg.arch.world_model.reward.num_units,
            (1,),
            act=cfg.arch.world_model.reward.act,
            norm=(cfg.arch.world_model.reward.norm != "none"),
        )

        if self.pcont:
            self.pcont = DenseDecoder(
                dense_input_size,
                cfg.arch.world_model.pcont.layers,
                cfg.arch.world_model.pcont.num_units,
                (1,),
                dist="binary",
                act="elu",
                norm=(cfg.arch.world_model.reward.norm != "none"),
            )
        self.r_transform = torch.tanh

        self.pcont_scale = cfg.loss.pcont_scale
        self.kl_scale = cfg.loss.kl_scale
        self.kl_balance = cfg.loss.kl_balance
        self.free_nats = cfg.loss.free_nats
        self.H = cfg.arch.H
        self.grad_clip = cfg.optimize.grad_clip

        self.discount = cfg.rl.discount
        self.lambda_ = cfg.rl.lambda_

        self.log_every_step = cfg.train.log_every_step
        self.log_grad = cfg.train.log_grad
        self.n_sample = cfg.train.n_sample
        self.batch_length = cfg.train.batch_length
        self.grayscale = cfg.env.grayscale
        self.K = cfg.arch.manager.K
        self.optimizer_type = cfg.optimize.type

    def forward(self, traj):
        raise NotImplementedError

    def compute_loss(self, traj, global_step):

        self.train()
        self.requires_grad_(True)

        prior_state, post_state = self.dynamic(traj, None)
        model_loss, model_logs = self.world_model_loss(
            global_step, traj, prior_state, post_state
        )

        return model_loss, model_logs, prior_state, post_state

    def world_model_loss(self, global_step, traj, prior_state, post_state):

        obs = traj["image"].float()
        obs = obs / 255.0 - 0.5
        reward = traj["reward"]
        reward = self.r_transform(reward).float()

        rnn_feature = self.dynamic.get_feature(post_state)

        image_pred_pdf = self.img_dec(rnn_feature)  # B, T, 3, 64, 64

        reward_pred_pdf = self.reward(rnn_feature)  # B, T, 1

        if self.pcont:
            pred_pcont = self.pcont(rnn_feature)  # B, T, 1
            pcont_target = self.discount * (1.0 - traj["done"].float())  # B, T
            pcont_loss = -pred_pcont.log_prob(
                (pcont_target.unsqueeze(2) > 0.5).float()
            ).mean()  #
            pcont_loss = self.pcont_scale * pcont_loss
            discount_acc = (pred_pcont.mean == pcont_target.unsqueeze(2)).float().mean()

        image_pred_loss = -image_pred_pdf.log_prob(obs).mean().float()  # B, T
        mse_loss = (
            F.mse_loss(image_pred_pdf.mean, obs, reduction="none")
            .flatten(start_dim=-3)
            .sum(-1)
            .mean()
            .detach()
        )
        reward_pred_loss = -reward_pred_pdf.log_prob(reward.unsqueeze(2)).mean()  # B, T
        reward_nonzero_loss = (
            ((-reward > 0.0).float() * reward_pred_pdf.log_prob(reward.unsqueeze(2)))
            .mean()
            .detach()
        )  # for monitoring
        pred_reward = reward_pred_pdf.mean

        prior_dist = self.dynamic.get_dist(prior_state)
        post_dist = self.dynamic.get_dist(post_state)
        value_lhs = kl_divergence(
            post_dist, self.dynamic.get_dist(prior_state, detach=True)
        )
        value_rhs = kl_divergence(
            self.dynamic.get_dist(post_state, detach=True), prior_dist
        )
        loss_lhs = torch.maximum(
            value_lhs.mean(),
            value_lhs.new_ones(value_lhs.mean().shape) * self.free_nats,
        )
        loss_rhs = torch.maximum(
            value_rhs.mean(),
            value_rhs.new_ones(value_rhs.mean().shape) * self.free_nats,
        )
        mix = 1.0 - self.kl_balance
        kl_loss = mix * loss_lhs + (1.0 - mix) * loss_rhs
        kl_loss = self.kl_scale * kl_loss

        model_loss = image_pred_loss + reward_pred_loss + kl_loss
        if self.pcont:
            model_loss = model_loss + pcont_loss

        if global_step % self.log_every_step == 0:
            logs = {
                "Loss_model_loss": model_loss.detach().item(),
                "Loss_model_kl_loss": kl_loss.detach().item(),
                "Loss_model_reward_logprob_loss": reward_pred_loss.detach().item(),
                "Loss_model_reward_nonzero_logprob_loss": reward_nonzero_loss.item(),
                "Loss_model_image_loss": image_pred_loss.detach().item(),
                "Loss_model_mse_loss": mse_loss.detach(),
                "ACT_prior_state": {k: v.detach() for k, v in prior_state.items()},
                "ACT_prior_entropy": prior_dist.entropy().mean().detach().item(),
                "ACT_post_state": {k: v.detach() for k, v in post_state.items()},
                "ACT_post_entropy": post_dist.entropy().mean().detach().item(),
                "dec_img": image_pred_pdf.mean.detach() + 0.5,  # B, T, 3, 64, 64
                "gt_img": obs + 0.5,
                "ACT_pred_reward": pred_reward.detach().squeeze(-1),
                "ACT_gt_reward": reward,
            }
            if self.pcont:
                logs.update(
                    {
                        "pred_discount": pred_pcont.mean.detach(),
                        "discount_acc": discount_acc.detach(),
                        "Loss_model_discount_logprob_loss": pcont_loss.detach().item(),
                    }
                )

        else:
            logs = {}

        return model_loss, logs

    def apply_weight_decay(self, varibs, wd):
        for var in varibs:
            var.data = (1 - wd) * var.data

    def optimize_world_model16(
        self, model_loss, model_optimizer, scaler, global_step, writer, wd
    ):

        scaler.scale(model_loss).backward()
        scaler.unscale_(model_optimizer)
        grad_norm_model = torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.grad_clip
        )

        if (global_step % self.log_every_step == 0) and self.log_grad:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    try:
                        writer.add_scalar(
                            "grads/world_model_" + n, p.grad.norm(2), global_step
                        )
                    except:
                        pdb.set_trace()
        scaler.step(model_optimizer)

        return grad_norm_model.item()

    def imagine_ahead(self, post_state, actor, mgr_actor):
        """
        post_state:
          mean: mean of q(s_t | h_t, o_t), (B*T, H)
          std: std of q(s_t | h_t, o_t), (B*T, H)
          stoch: s_t sampled from q(s_t | h_t, o_t), (B*T, H)
          deter: h_t, (B*T, H)
        """

        self.eval()
        self.requires_grad_(False)

        def flatten(tensor):
            """
            flatten the temporal dimension and the batch dimension.
            tensor: B, T, *
            """
            shape = tensor.shape
            tensor = tensor.reshape([shape[0] * shape[1]] + [*tensor.shape[2:]])
            return tensor

        post_state_flatten = {k: flatten(v).detach() for k, v in post_state.items()}

        pred_state = defaultdict(list)

        pred_prior = post_state_flatten
        rnn_feat_list = []
        action_list = []
        goal_list = []
        mgr_action_list = []

        for t in range(self.H):

            rnn_feature = self.dynamic.get_feature(pred_prior)
            if t % self.K == 0:
                g, z = mgr_actor(pred_prior)
            goal_list.append(g)
            mgr_action_list.append(z)

            pred_action_pdf = actor(rnn_feature.detach(), g.detach())
            action = pred_action_pdf.sample()
            if self.pcont:
                action = (
                    action + pred_action_pdf.probs - pred_action_pdf.probs.detach()
                )  # straight through

            pred_deter = self.dynamic.rnn_forward(
                action, pred_prior["stoch"], pred_prior["deter"]
            )
            pred_prior = self.dynamic.infer_prior_stoch(pred_deter)

            for k, v in pred_prior.items():
                pred_state[k].append(v)
            rnn_feat_list.append(rnn_feature)
            action_list.append(action)

        for k, v in pred_state.items():
            pred_state[k] = torch.cat(
                [post_state_flatten[k].unsqueeze(1), torch.stack(v, 1)[:, :-1]], dim=1
            )
        actions = torch.stack(action_list, dim=1)
        rnn_features = torch.stack(rnn_feat_list, dim=1)
        goal_list = torch.stack(goal_list, dim=1)
        mgr_action_list = torch.stack(mgr_action_list, dim=1)

        reward = self.reward(rnn_features).mean  # B*T, H, 1
        if self.pcont:
            discount = self.discount * self.pcont(rnn_features).mean  # B*T, H, 1
        else:
            discount = self.discount * reward.new_ones(reward.shape)  # B*T, H, 1

        return (
            rnn_features,
            pred_state,
            actions,
            reward,
            discount,
            goal_list,
            mgr_action_list,
        )

    def gen_samples(self, traj, logs, gt_img, rec_img, global_step, writer):

        self.eval()
        self.requires_grad_(False)

        gt_img = gt_img
        rec_img = rec_img[:, : self.n_sample]  # B, T, C, H, W
        pred_action = traj["action"][:, self.n_sample :]
        reward = traj["reward"]
        reward = self.r_transform(reward)
        with torch.no_grad():
            prev_stoch = logs["ACT_prior_state"]["stoch"][:, self.n_sample - 1].detach()
            prev_deter = logs["ACT_prior_state"]["deter"][:, self.n_sample - 1].detach()
            rnn_features = []
            stoch_state = []
            for t in range(self.batch_length - self.n_sample):
                prev_deter = self.dynamic.rnn_forward(
                    pred_action[:, t], prev_stoch, prev_deter
                )
                prior = self.dynamic.infer_prior_stoch(prev_deter)
                rnn_features.append(self.dynamic.get_feature(prior))
                stoch_state.append(prior["stoch"])

            rnn_features = torch.stack(rnn_features, dim=1)  # B, T-n_sample, H

            pred_imgs = self.img_dec(rnn_features).mean + 0.5  # B, T, 3, 64, 64

            reward_pred_pdf = self.reward(rnn_features)  # B, T, 1
            pred_reward_pred_loss = -reward_pred_pdf.log_prob(
                reward[:, self.n_sample :].unsqueeze(2)
            ).mean()  # B, T
            pred_reward_nonzero_loss = (
                (
                    -(reward[:, self.n_sample :] > 0.0).float()
                    * reward_pred_pdf.log_prob(reward[:, self.n_sample :].unsqueeze(2))
                )
                .mean()
                .detach()
            )  # for monitoring
            pred_reward = reward_pred_pdf.mean
            gt_reward = reward[:, self.n_sample :]

            if self.pcont:
                pred_pcont = self.pcont(rnn_features)  # B, T, 1
                pcont_target = self.discount * (1.0 - traj["done"].float())  # B, T
                pcont_loss = -pred_pcont.log_prob(
                    (pcont_target.unsqueeze(2)[:, self.n_sample :] > 0.5).float()
                ).mean()  #
                pred_pcont_loss = self.pcont_scale * pcont_loss
                discount_acc = (
                    (pred_pcont.mean == pcont_target[:, self.n_sample :].unsqueeze(2))
                    .float()
                    .mean()
                )

        imgs_act = torch.cat([rec_img, pred_imgs], dim=1)  # B, T, C, H, W
        err = gt_img - imgs_act
        gen_mse = (err[:, self.n_sample :] ** 2.0).flatten(start_dim=-3).sum(-1).mean()
        imgs = torch.cat([gt_img, imgs_act, err], dim=3).cpu()[:6]
        if self.grayscale:
            imgs = imgs.expand(-1, -1, 3, -1, -1)
        color_bar = torch.zeros([*imgs.shape[:4]] + [5])
        color_bar[:, : self.n_sample, 0] = 1
        color_bar[:, self.n_sample :, 1] = 1
        final_vis = torch.cat([color_bar, imgs], dim=4)  # B, T, C, H, W+5
        writer.add_video(
            "test/gt - rec - gen - err",
            final_vis.clamp(0.0, 1.0),
            global_step=global_step,
        )
        writer.add_scalar("Gen-losses/mse", gen_mse, global_step=global_step)
        writer.add_scalar(
            "Gen-losses/reward_logprob_loss",
            pred_reward_pred_loss,
            global_step=global_step,
        )
        writer.add_scalar(
            "Gen-losses/reward_nonzero_logprob_loss",
            pred_reward_nonzero_loss,
            global_step=global_step,
        )
        if self.pcont:
            writer.add_scalar(
                "Gen-losses/discount_logprob_loss",
                pred_pcont_loss,
                global_step=global_step,
            )
            writer.add_scalar(
                "Gen-losses/discount_acc", discount_acc, global_step=global_step
            )
        writer.flush()

        return {
            "pred_reward": pred_reward.detach().squeeze(-1),
            "gt_reward": gt_reward,
        }


class RSSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        act = cfg.arch.world_model.RSSM.act
        deter_size = cfg.arch.world_model.RSSM.deter_size
        hidden_size = cfg.arch.world_model.RSSM.hidden_size
        action_size = cfg.env.action_size
        self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
        self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
        self.norm = cfg.arch.world_model.norm
        self.bias = cfg.arch.world_model.RSSM.bias

        self.img_enc = ImgEncoder(cfg)

        weight_init = cfg.arch.world_model.RSSM.weight_init
        # RNN cell
        self.cell = LayerNormGRUCellV2(hidden_size, deter_size, bias=self.bias)
        if self.stoch_discrete:
            latent_size = self.stoch_size * self.stoch_discrete
        else:
            latent_size = 2 * self.stoch_size

        # MLP layers
        if self.stoch_discrete:
            self.act_stoch_mlp = Linear(
                action_size + latent_size, hidden_size, weight_init=weight_init,
            )
        else:
            self.act_stoch_mlp = Linear(
                action_size + latent_size // 2, hidden_size, weight_init=weight_init,
            )

        self.post_stoch_mlp = MLP(
            [1536 + deter_size, hidden_size, latent_size,],
            act=act,
            weight_init=weight_init,
            norm=self.norm,
        )
        self.prior_stoch_mlp = MLP(
            [deter_size, hidden_size, latent_size],
            act=act,
            weight_init=weight_init,
            norm=self.norm,
        )

    def init_state(self, batch_size, device):
        deter = self.cell.init_state(batch_size)
        if self.stoch_discrete:
            stoch = torch.zeros(
                (batch_size, self.stoch_size, self.stoch_discrete), device=device
            )
        else:
            stoch = torch.zeros((batch_size, self.stoch_size), device=device)

        state = {
            "deter": deter,
            "stoch": stoch,
        }
        return state

    def forward(self, traj, prev_state):
        """
        traj:
          observations: embedding of observed images, B, T, C
          actions: (one-hot) vector in action space, B, T, d_act
          dones: scalar, B, T

        prev_state:
          deter: GRU hidden state, B, h1
          stoch: RSSM stochastic state, B, h2
        """

        obs = traj["image"]
        obs = obs / 255.0 - 0.5
        obs_emb = self.img_enc(obs)  # B, T, C

        actions = traj["action"]
        dones = traj["done"]

        if prev_state is None:
            prev_state = self.init_state(obs_emb.shape[0], obs_emb.device)

        prior, post = self.infer_states(obs_emb, actions, dones, prev_state)

        return prior, post

    def get_feature(self, state):

        if self.stoch_discrete:
            shape = state["stoch"].shape
            stoch = state["stoch"].reshape(
                [*shape[:-2]] + [self.stoch_size * self.stoch_discrete]
            )
            return torch.cat([stoch, state["deter"]], dim=-1)  # B, T, 2H

        else:
            return torch.cat([state["stoch"], state["deter"]], dim=-1)  # B, T, 2H

    def get_dist(self, state, detach=False):
        if self.stoch_discrete:
            return self.get_discrete_dist(state, detach)
        else:
            return self.get_normal_dist(state, detach)

    def get_normal_dist(self, state, detach):

        mean = state["mean"]
        std = state["std"]

        if detach:
            mean = mean.detach()
            std = std.detach()

        return Independent(Normal(mean, std), 1)

    def get_discrete_dist(self, state, detach):

        logits = state["logits"]

        if detach:
            logits = logits.detach()

        return Independent(OneHotCategorical(logits=logits), 1)

    def rnn_forward(self, action, prev_stoch, prev_deter):

        if self.stoch_discrete:
            shape = prev_stoch.shape
            prev_stoch = prev_stoch.reshape(
                [*shape[:-2]] + [self.stoch_size * self.stoch_discrete]
            )

        act_sto_emb = self.act_stoch_mlp(torch.cat([action, prev_stoch], dim=-1))
        act_sto_emb = F.elu(act_sto_emb)
        deter = self.cell(act_sto_emb, prev_deter)

        return deter

    def infer_states(self, observations, actions, dones, prev_state):

        prev_deter, prev_stoch = prev_state["deter"], prev_state["stoch"]

        prior_states = defaultdict(list)
        post_states = defaultdict(list)

        T = observations.shape[1]
        for t in range(T):

            prev_deter = self.rnn_forward(actions[:, t], prev_stoch, prev_deter)

            prior = self.infer_prior_stoch(prev_deter)

            post = self.infer_post_stoch(observations[:, t], prev_deter)

            prev_stoch = post["stoch"]

            for k, v in prior.items():
                prior_states[k].append(v)
            for k, v in post.items():
                post_states[k].append(v)

        for k, v in prior_states.items():
            prior_states[k] = torch.stack(v, dim=1)
        for k, v in post_states.items():
            post_states[k] = torch.stack(v, dim=1)  # B, T, C

        prior_states["stoch_int"] = prior_states["stoch"].argmax(-1).float()
        post_states["stoch_int"] = post_states["stoch"].argmax(-1).float()

        return prior_states, post_states

    def infer_prior_stoch(self, deter):

        logits = self.prior_stoch_mlp(deter)
        logits = logits.float()

        if self.stoch_discrete:

            logits = logits.reshape(
                [*logits.shape[:-1]] + [self.stoch_size, self.stoch_discrete]
            )
            dist = Independent(OneHotCategorical(logits=logits), 1)
            stoch = dist.sample()
            stoch = stoch + dist.mean - dist.mean.detach()

            prior_state = {
                "logits": logits,
                "stoch": stoch,
                "deter": deter,
            }
        else:
            mean, std = logits.chunk(2, dim=-1)
            std = 2.0 * torch.sigmoid(std / 2.0) + 0.1
            pdf = Normal(mean, std)

            stoch = pdf.rsample()
            prior_state = {
                "mean": mean,
                "std": std,
                "stoch": stoch,
                "deter": deter,
            }

        return prior_state

    def infer_post_stoch(self, observation, prev_deter):

        logits = self.post_stoch_mlp(torch.cat([observation, prev_deter], dim=-1))
        logits = logits.float()

        if self.stoch_discrete:

            logits = logits.reshape(
                [*logits.shape[:-1]] + [self.stoch_size, self.stoch_discrete]
            )
            dist = Independent(OneHotCategorical(logits=logits), 1)

            stoch = dist.sample()
            stoch = stoch + dist.mean - dist.mean.detach()

            post_state = {
                "logits": logits,
                "stoch": stoch,
                "deter": prev_deter,
            }
        else:
            mean, std = logits.chunk(2, dim=-1)
            std = 2.0 * torch.sigmoid(std / 2.0) + 0.1
            pdf = Normal(mean, std)

            stoch = pdf.rsample()
            post_state = {
                "mean": mean,
                "std": std,
                "stoch": stoch,
                "deter": prev_deter,
            }

        return post_state


class ImgEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        depth = 48
        c_in = 1 if cfg.env.grayscale else 3
        n_group_norm = 1 if cfg.arch.world_model.norm != "none" else 0
        self.enc = nn.Sequential(
            Conv2DBlock(
                c_in,
                depth,
                4,
                2,
                0,
                num_groups=n_group_norm,
                bias=True,
                non_linearity=True,
                act="elu",
                weight_init="xavier",
            ),
            Conv2DBlock(
                depth,
                2 * depth,
                4,
                2,
                0,
                num_groups=n_group_norm,
                bias=True,
                non_linearity=True,
                act="elu",
                weight_init="xavier",
            ),
            Conv2DBlock(
                2 * depth,
                4 * depth,
                4,
                2,
                0,
                num_groups=n_group_norm,
                bias=True,
                non_linearity=True,
                act="elu",
                weight_init="xavier",
            ),
            Conv2DBlock(
                4 * depth,
                8 * depth,
                4,
                2,
                0,
                num_groups=n_group_norm,
                bias=True,
                non_linearity=True,
                act="elu",
                weight_init="xavier",
            ),
        )

    def forward(self, ipts):
        """
    ipts: tensor, (B, T, 3, 64, 64)
    return: tensor, (B, T, 1024)
    """

        shapes = ipts.shape
        o = self.enc(ipts.view([-1] + [*shapes[-3:]]))
        o = o.reshape([*shapes[:-3]] + [1536])

        return o


class ImgDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        depth = 48
        self.c_out = 1 if cfg.env.grayscale else 3
        self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
        if self.stoch_discrete:
            input_size = (
                cfg.arch.world_model.RSSM.deter_size
                + cfg.arch.world_model.RSSM.stoch_size * self.stoch_discrete
            )
        else:
            input_size = (
                cfg.arch.world_model.RSSM.deter_size
                + cfg.arch.world_model.RSSM.stoch_size
            )
        self.fc = Linear(input_size, 1536, bias=True, weight_init="xavier")
        n_group_norm = 1 if cfg.arch.world_model.norm != "none" else 0
        if cfg.arch.decoder.dec_type == "conv":
            self.dec = nn.Sequential(
                ConvTranspose2DBlock(
                    1536,
                    4 * depth,
                    5,
                    2,
                    0,
                    num_groups=n_group_norm,
                    bias=True,
                    non_linearity=True,
                    act="elu",
                    weight_init="xavier",
                ),
                ConvTranspose2DBlock(
                    4 * depth,
                    2 * depth,
                    5,
                    2,
                    0,
                    num_groups=n_group_norm,
                    bias=True,
                    non_linearity=True,
                    act="elu",
                    weight_init="xavier",
                ),
                ConvTranspose2DBlock(
                    2 * depth,
                    depth,
                    6,
                    2,
                    0,
                    num_groups=n_group_norm,
                    bias=True,
                    non_linearity=True,
                    act="elu",
                    weight_init="xavier",
                ),
                ConvTranspose2DBlock(
                    depth,
                    self.c_out,
                    6,
                    2,
                    0,
                    num_groups=0,
                    bias=True,
                    non_linearity=False,
                    weight_init="xavier",
                ),
            )

        elif cfg.dec_type == "pixelshuffle":
            pass

        else:
            raise ValueError(f"decoder type {cfg.dec_type} is not supported.")

        self.shape = (self.c_out, 64, 64)

    def forward(self, ipts):
        """
        ipts: tensor, (B, T, C)
        """

        shape = ipts.shape

        fc_o = self.fc(ipts)
        dec_o = self.dec(fc_o.reshape(shape[0] * shape[1], 1536, 1, 1))
        dec_o = dec_o.reshape([*shape[:2]] + [self.c_out, 64, 64])

        dec_pdf = Independent(
            Normal(dec_o, dec_o.new_ones(dec_o.shape)), len(self.shape)
        )

        return dec_pdf


class DenseDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        layers,
        units,
        output_shape,
        weight_init="xavier",
        dist="normal",
        act="relu",
        norm=False,
        output_dim=1,
        learn_std=False,
    ):
        super().__init__()

        acts = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "celu": nn.CELU,
        }
        module_list = []

        for i in range(layers):

            if i == 0:
                dim_in = input_size
            else:
                dim_in = units
            dim_out = units

            module_list.append(Linear(dim_in, dim_out, weight_init=weight_init))
            if norm:
                module_list.append(nn.LayerNorm(dim_out))
            module_list.append(acts[act]())

        self.learn_std = learn_std
        if learn_std:
            output_dim = output_dim * 2
        module_list.append(Linear(dim_out, output_dim, weight_init=weight_init))
        self.dec = nn.Sequential(*module_list)

        self.dist = dist
        self.output_shape = output_shape

    def forward(self, inpts):

        logits = self.dec(inpts)
        logits = logits.float()

        if self.dist == "normal":
            if self.learn_std:

                mean, std = logits.chunk(2, dim=-1)
                std = 2.0 * torch.sigmoid(std / 2.0) + 0.1
                pdf = Independent(Normal(mean, std), len(self.output_shape))
            else:
                pdf = Independent(Normal(logits, 1), len(self.output_shape))

        elif self.dist == "binary":
            pdf = Independent(Bernoulli(logits=logits), len(self.output_shape))

        elif self.dist == "none":
            return logits
        else:
            raise NotImplementedError(self.dist)

        return pdf


class ActionDecoder(nn.Module):
    def __init__(
        self,
        cfg,
        input_size,
        action_size,
        layers,
        units,
        dist="onehot",
        act="relu",
        min_std=0.1,
        init_std=5,
        mean_scale=1,
        weight_init="xavier",
        norm=False,
    ):
        super().__init__()

        acts = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "celu": nn.CELU,
        }
        module_list = []

        for i in range(layers):

            if i == 0:
                dim_in = input_size
            else:
                dim_in = units
            dim_out = units

            module_list.append(Linear(dim_in, dim_out, weight_init=weight_init))
            if norm:
                module_list.append(nn.LayerNorm(dim_out))
            module_list.append(acts[act]())

        if dist == "trunc_normal":
            module_list.append(
                Linear(dim_out, 2 * action_size, weight_init=weight_init)
            )

        elif dist == "onehot":
            module_list.append(Linear(dim_out, action_size, weight_init=weight_init))

        else:
            raise NotImplementedError(self.dist)

        self.dec = nn.Sequential(*module_list)
        self.dist = dist
        self.raw_init_std = np.log(np.exp(init_std) - 1)
        self.min_std = min_std
        self.mean_scale = mean_scale
        self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
        self.worker_only = cfg.arch.worker_only

    def forward(self, inpts, goal):

        if not self.worker_only:
            inpts = torch.cat([inpts, goal], dim=-1)

        if self.dist == "trunc_normal":

            logits = self.dec(inpts).float()
            mean, std = torch.chunk(logits, 2, -1)
            mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
            std = 2.0 * F.softplus(std / 2.0) + self.min_std
            dist = SafeTruncatedNormal(mean, std, -1, 1)
            dist = ContDist(torch.distributions.Independent(dist, 1))

        if self.dist == "onehot":

            logits = self.dec(inpts)
            logits = logits.float()
            dist = OneHotCategorical(logits=logits)

        return dist


class GoalEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        action_num,
        action_size,
        layers,
        units,
        dist="onehot",
        act="relu",
        norm="none",
        weight_init="xavier",
    ):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "celu": nn.CELU,
        }
        module_list = []
        for i in range(layers):
            module_list.append(
                Linear(input_size if i == 0 else units, units, weight_init=weight_init)
            )
            if norm:
                module_list.append(nn.LayerNorm(units))
            module_list.append(acts[act]())
        module_list.append(
            Linear(units, action_num * action_size, weight_init=weight_init)
        )
        self.enc = nn.Sequential(*module_list)
        self.action_num = action_num
        self.action_size = action_size

    def forward(self, inputs):
        """
        inputs: B, S
        """

        logits = (
            self.enc(inputs)
            .reshape(*inputs.shape[:-1], self.action_num, self.action_size)
            .float()
        )
        dist = OneHotCategorical(logits=logits)

        return dist


class GoalDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
        self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
        self.input_type = cfg.arch.manager.input_type
        self.deter_size = cfg.arch.world_model.RSSM.deter_size

        if self.input_type == "latent":
            if self.stoch_discrete:
                output_dim = input_size = self.stoch_size * self.stoch_discrete
            else:
                output_dim = input_size = self.stoch_size
        elif self.input_type == "deter":
            output_dim = input_size = self.deter_size

        self.dec = DenseDecoder(
            cfg.arch.manager.actor.action_num * cfg.arch.manager.actor.action_size,
            cfg.arch.manager.actor.layers,
            cfg.arch.manager.actor.num_units,
            (1,),
            output_dim=output_dim,
            dist="none",
            act=cfg.arch.manager.actor.act,
            norm=(cfg.arch.manager.actor.norm != "none"),
            learn_std=cfg.arch.manager.learn_vae_std,
        )
        self.action_num = cfg.arch.manager.actor.action_num
        self.action_size = cfg.arch.manager.actor.action_size
        self.learn_std = cfg.arch.manager.learn_vae_std
        self.output_shape = (1,)

    def forward(self, inpts):

        logits = self.dec(inpts)
        logits = logits.float()

        if self.input_type == "latent":
            if self.stoch_discrete:
                pdf = OneHotCategorical(
                    logits=logits.reshape(
                        *logits.shape[:-1], self.stoch_discrete, self.stoch_size
                    )
                )
            else:
                if self.learn_std:

                    mean, std = logits.chunk(2, dim=-1)
                    std = 2.0 * torch.sigmoid(std / 2.0) + 0.1
                    pdf = Independent(Normal(mean, std), len(self.output_shape))
                else:
                    pdf = Independent(Normal(logits, 1), len(self.output_shape))
        elif self.input_type == "deter":

            if self.learn_std:

                mean, std = logits.chunk(2, dim=-1)
                std = 2.0 * torch.sigmoid(std / 2.0) + 0.1
                pdf = Independent(Normal(mean, std), len(self.output_shape))
            else:
                pdf = Independent(Normal(logits, 1), len(self.output_shape))

        return pdf


class GoalVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
        self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
        self.input_type = cfg.arch.manager.input_type
        self.deter_size = cfg.arch.world_model.RSSM.deter_size

        if self.input_type == "latent":
            if self.stoch_discrete:
                output_dim = input_size = self.stoch_size * self.stoch_discrete
            else:
                output_dim = input_size = self.stoch_size
        elif self.input_type == "deter":
            output_dim = input_size = self.deter_size

        self.enc = GoalEncoder(
            input_size,
            cfg.arch.manager.actor.action_num,
            cfg.arch.manager.actor.action_size,
            cfg.arch.manager.actor.layers,
            cfg.arch.manager.actor.num_units,
            dist=cfg.arch.manager.actor.dist,
            act=cfg.arch.manager.actor.act,
            norm=(cfg.arch.manager.actor.norm != "none"),
        )

        self.dec = GoalDecoder(cfg)
        self.action_num = cfg.arch.manager.actor.action_num
        self.action_size = cfg.arch.manager.actor.action_size

    def prior(self, pred_mean, batch_shape):

        dim = len(pred_mean.shape) - len(batch_shape) - 1

        uni_probs = (
            pred_mean.new_ones(*batch_shape, self.action_size) / self.action_size
        )
        return OneHotCategorical(probs=uni_probs)

    def kl_loss(self, pred_dist):
        return kl_divergence(
            pred_dist, self.prior(pred_dist.mean, pred_dist.batch_shape)
        )


class ManagerValue(nn.Module):
    def __init__(self, manager_cfg, cfg):
        super().__init__()

        self.stoch_discrete = cfg.arch.world_model.RSSM.stoch_discrete
        self.stoch_size = cfg.arch.world_model.RSSM.stoch_size
        self.input_type = cfg.arch.manager.input_type
        deter_size = cfg.arch.world_model.RSSM.deter_size

        if self.stoch_discrete:
            input_size = self.stoch_size * self.stoch_discrete
        else:
            input_size = self.stoch_size

        input_size = input_size + deter_size
        self.intri_value = DenseDecoder(
            input_size,
            manager_cfg.intri_value.layers,
            manager_cfg.intri_value.num_units,
            (1,),
            output_dim=1,
            act=manager_cfg.intri_value.act,
            norm=(manager_cfg.intri_value.norm != "none"),
        )

        self.extri_value = DenseDecoder(
            input_size,
            manager_cfg.extri_value.layers,
            manager_cfg.extri_value.num_units,
            (1,),
            output_dim=1,
            act=manager_cfg.extri_value.act,
            norm=(manager_cfg.extri_value.norm != "none"),
        )

    def forward(self, state):

        extri_value = self.extri_value(state)  # B*T, H, 1
        intri_value = self.intri_value(state)  # B*T, H, 1

        return extri_value, intri_value

