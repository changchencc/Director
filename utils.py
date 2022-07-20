import os
import pdb
import pickle
import torch.nn as nn
import torch
import numpy as np
from model.utils import spatial_transformer


class Checkpointer(object):
    def __init__(self, checkpoint_dir, max_num):
        self.max_num = max_num
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.model_list_path = os.path.join(checkpoint_dir, "model_list.pkl")

        if not os.path.exists(self.model_list_path):
            model_list = []
            with open(self.model_list_path, "wb") as f:
                pickle.dump(model_list, f)

    def load(self, path):
        if path == "":
            with open(self.model_list_path, "rb") as f:
                model_list = pickle.load(f)

            if len(model_list) == 0:
                print("Start training from scratch.")
                return None

            else:
                path = model_list[-1]
                print(f"Load checkpoint from {path}")

                checkpoint = torch.load(path)
                return checkpoint
        else:

            assert os.path.exists(path), f"checkpoint {path} not exits."
            checkpoint = torch.load(path)

            return checkpoint

    def save(self, path, model, optimizers, global_step, env_step):

        if path == "":
            path = os.path.join(
                self.checkpoint_dir, "model_{:09}.pth".format(global_step + 1)
            )

            with open(self.model_list_path, "rb+") as f:
                model_list = pickle.load(f)
                if len(model_list) >= self.max_num:
                    if os.path.exists(model_list[0]):
                        os.remove(model_list[0])

                    del model_list[0]
                model_list.append(path)
            with open(self.model_list_path, "rb+") as f:
                pickle.dump(model_list, f)

        if isinstance(model, nn.DataParallel):
            model = model.module

        checkpoint = {
            "model": model.state_dict(),
            "global_step": global_step,
            "env_step": env_step,
        }
        for k, v in optimizers.items():
            if v is not None:
                state_dict = []
                for vv in v:
                    state_dict.append(vv.state_dict())

                checkpoint.update(
                    {k: state_dict,}
                )

        assert path.endswith(".pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            torch.save(checkpoint, f)

        print(f"Saved checkpoint to {path}")


# def sample_random_pos(T, H, W, step, patch_size, repeat):
#   h = w = patch_size
#
#   h_min = h // 2
#   h_max = H - h // 2
#   assert (h_max - h_min) % step == 0, ''
#   h_range = np.arange(h_min, h_max + 1, step)
#
#   w_min = w // 2
#   w_max = W - w_min
#   assert (w_max - w_min) % step == 0, ''
#   w_range = np.arange(w_min, w_max + 1, step)
#
#   w = np.random.choice(w_range, T).repeat(repeat)[:T]
#   h = np.random.choice(h_range, T).repeat(repeat)[:T]
#
#   return np.stack([w, h], axis=1)


def sample_random_pos(grid_n, T, repeat, start_pixel, step):

    h = np.random.randint(grid_n, size=T) * step + start_pixel
    w = np.random.randint(grid_n, size=T) * step + start_pixel

    h = h.repeat(repeat)[:T]
    w = w.repeat(repeat)[:T]

    return np.stack([w, h], axis=1)


def sample_ordered_pos(grid_n, T, repeat, start_pixel, step, last_idx):
    hh = np.arange(grid_n).reshape(grid_n, 1)
    ww = np.arange(grid_n).reshape(1, grid_n)
    ww, hh = np.meshgrid(ww, hh)
    ww, hh = start_pixel + ww * step, start_pixel + hh * step
    pos = np.stack([hh, ww], axis=-1)

    # starting from bottom left
    perm_idx = [2, 1, 0]
    pos = pos[perm_idx]
    pos = pos.reshape(grid_n ** 2, 2).repeat(repeat, axis=0)

    idx = last_idx + np.arange(T)
    num_pos = repeat * (grid_n ** 2)
    pos = pos[idx % num_pos]
    last_idx = idx[-1] + 1  # T, 2

    return pos, last_idx


def crop_obs(
    images, patch_size, random_crop, last_idx, repeat=4, pos=None, resize=True
):

    T, C, H, W = images.shape[:4]
    if patch_size == 48:
        step = 8
        start_pixel = 24
        grid_n = 3
    if patch_size == 32:
        step = 16
        start_pixel = 16
        grid_n = 3
    if patch_size == 24:
        step = 10
        start_pixel = 12
        grid_n = 5

    scales = patch_size / H * np.ones((T, 2)).astype(np.float32)

    if pos is None:
        if random_crop:
            pos = sample_random_pos(grid_n, T, repeat, start_pixel, step)
            last_idx = 0

        else:
            assert last_idx is not None, ""
            pos, last_idx = sample_ordered_pos(
                grid_n, T, repeat, start_pixel, step, last_idx
            )

    pos_crop = np.concatenate(
        [pos[:, 1:], pos[:, :1]], axis=1
    )  # convert (x, y) to (h, w)
    if resize:
        crops = spatial_transformer(
            torch.tensor(pos_crop.astype(np.float32) / (H / 2.0) - 1.0),
            torch.tensor(scales),
            torch.tensor(images).float(),
            H,
            W,
        )

    else:
        crops = spatial_transformer(
            torch.tensor(pos_crop.astype(np.float32) / (H / 2.0) - 1.0),
            torch.tensor(scales),
            crops,
            patch_size,
            patch_size,
            inverse=True,
        )

    return crops, pos_crop, last_idx

