import atexit
import functools
import pdb
import sys
import threading
import traceback

import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from gym_unity.envs import UnityToGymWrapper
import gym


class IMaze3D:
  LOCK = threading.Lock()

  def __init__(self, name, env_file, id, action_repeat, action_size, top_view,
               size=(64, 64), grayscale=False, seed=0, life_done=False):
    assert size[0] == size[1]

    with self.LOCK:
      env = UnityEnvironment(file_name=env_file, worker_id=id, seed=seed, timeout_wait=1000)
      env = UnityToGymWrapper(env, True,  allow_multiple_obs=top_view)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    self._env = env
    self._grayscale = grayscale
    self._size = size
    self.action_size = action_size
    self.action_repeat = action_repeat
    self.top_view = top_view
    self.life_done = life_done

  @property
  def observation_space(self):
    shape = (1 if self._grayscale else 3,) + self._size
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      obs = self._env.reset()
    if self._grayscale:
      if self.top_view:
        image = obs[0][..., None]
        top_view = obs[1][..., None]
      else:
        image = obs[..., None]
    else:
      image = obs[0]
      top_view = obs[1]

    image = np.transpose(image, (2, 0, 1))  # 3, 64, 64

    if self.top_view:
      top_view = np.transpose(top_view, (2, 0, 1))  # 3, 64, 64
      return {'image': image, 'top_view': top_view}
    else:
      return {'image': image}

  def step(self, action):
    action = action + 1 # unity reserve action 0 as staying still
    reward = 0
    for _ in range(self.action_repeat):
      obs, r, done, info = self._env.step(action)
      reward = reward + r
      if done:
        if self.life_done:
          done = False
          obs = self._env.reset()
        break
    if self._grayscale:
      if self.top_view:
        image = obs[0][..., None]
        top_view = obs[1][..., None]
      else:
        image = obs[..., None]
    else:
      image = obs[0]
      top_view = obs[1]
    image = np.transpose(image, (2, 0, 1))  # 3, 64, 64
    if self.top_view:
      top_view = np.transpose(top_view, (2, 0, 1))  # 3, 64, 64
      obs = {'image': image, 'top_view': top_view}
    else:
      obs = {'image': image}

    return obs, reward, done, info
