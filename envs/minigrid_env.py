import atexit
import threading

import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import cv2
import torch
from torch import distributions as torchd
import numpy as np
import pdb


class GymGridEnv:
    LOCK = threading.Lock()

    def __init__(self, name, action_repeat, max_steps=245, life_done=False):

        with self.LOCK:
            env = gym.make(name)
            env = RGBImgPartialObsWrapper(
                env, tile_size=9
            )  # Get pixel observations, (63, 63, 3)
            self._env = ImgObsWrapper(env)  # Get rid of the 'mission' field
            self._env.max_steps = max_steps
        self.action_repeat = action_repeat
        self._step_counter = 0
        self._random = np.random.RandomState(seed=None)
        self.life_done = life_done
        self.max_steps = max_steps
        self.action_size = 6

    def reset(self):

        self._step_counter = 0  # Reset internal timer
        with self.LOCK:
            observation = self._env.reset()

        # observation = self._env.render(mode='rgb_array')
        observation = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_LINEAR)
        observation = np.clip(observation, 0, 255).astype(np.uint8)
        observation = np.transpose(observation, (2, 0, 1))  # 3, 64, 64
        self._step_counter = 0

        return {"image": observation}

    def step(self, action):
        reward = 0
        RESET = False
        for k in range(self.action_repeat):
            observation, reward_k, done, info = self._env.step(action)
            reward += reward_k
            self._step_counter += 1  # Increment internal timer

            if done:
                observation = self._env.reset()
                RESET = True

            if self.life_done:
                done = self._step_counter == self.max_steps

            if RESET:
                break

        # observation = self._env.render(mode='rgb_array')
        observation = cv2.resize(observation, (64, 64), interpolation=cv2.INTER_LINEAR)
        observation = np.clip(observation, 0, 255).astype(np.uint8)
        observation = np.transpose(observation, (2, 0, 1))  # 3, 64, 64

        return {"image": observation}, reward, done, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        shape = (3, 64, 64)
        space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        return gym.spaces.Dict({"image": space})

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.action_size)

