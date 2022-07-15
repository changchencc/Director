import atexit
import functools
import sys
import threading
import traceback

import gym
import numpy as np
from PIL import Image
import crafter


class Crafter:

    LOCK = threading.Lock()

    def __init__(self, name, size=(84, 84), seed=0):
        assert size[0] == size[1]
        with self.LOCK:
            env = crafter.Env(
                area=(64, 64), view=(9, 9), size=size, length=10000, seed=seed
            )
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None
        self._env = env
        self._size = size
        self.action_size = 17
        self._achievements = crafter.constants.achievements.copy()

    @property
    def observation_space(self):
        shape = (1 if self._grayscale else 3,) + self._size
        return gym.spaces.Dict({"image": gym.spaces.Box(0, 255, shape, np.uint8),})

    @property
    def action_space(self):
        return {"action": self._env.action_space}

    def close(self):
        return self._env.close()

    def reset(self):
        with self.LOCK:
            image = self._env.reset()
        image = np.transpose(image, (2, 0, 1))  # 3, 64, 64
        obs = {
            "image": image,
        }
        return obs

    def step(self, action):
        image, reward, done, info = self._env.step(action)
        image = np.transpose(image, (2, 0, 1))  # 3, 64, 64
        obs = {
            "image": image,
        }

        return obs, reward, done, info

    def render(self, mode):
        return self._env.render(mode)

