import atexit
import traceback
import numpy as np
import os
import pdb
from multiprocessing import Pool, Pipe, Process
import multiprocessing as mp
import cloudpickle
import pickle
from torch import distributions as torchd
import torch
import gym
import sys


class OneHotAction:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def sample_random_action(self):
        action = np.zeros((self._env.action_space["action"].n,), dtype=np.float)
        idx = np.random.randint(0, self._env.action_space["action"].n, size=(1,))[0]
        action[idx] = 1
        return action


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._step = None
        self._duration = duration

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self._env.step(action)
        self._step += 1

        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class Collect:
    def __init__(self, env, callbacks=None, precision=32):
        self._env = env
        self._callbacks = callbacks or ()
        self._precision = precision
        self._episode = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {k: self._convert(v) for k, v in obs.items()}
        transition = obs.copy()
        transition["action"] = action
        transition["reward"] = reward
        transition["discount"] = info.get("discount", np.array(1 - float(done)))
        transition["done"] = float(done)
        self._episode.append(transition)
        if done:
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            info["episode"] = episode
            for callback in self._callbacks:
                callback(episode)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        transition["action"] = np.zeros(self.action_size)
        transition["discount"] = 1.0
        transition["reward"] = 0.0
        transition["done"] = 0.0
        self._episode = [transition]
        return obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = np.float32
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = np.int32
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            pdb.set_trace()
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)


class RewardObs:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert "reward" not in spaces
        spaces["reward"] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs["reward"] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs["reward"] = 0.0
        return obs


class NormalizeActions:
    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

        self.random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(
                torch.Tensor(env.action_space.low)[None],
                torch.Tensor(env.action_space.high)[None],
            ),
            1,
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)

    def sample_random_action(self):
        return self.random_actor.sample()[0].numpy()


# class AsyncEnv:

#     _ACCESS = 1
#     _CALL = 2
#     _RESULT = 3
#     _CLOSE = 4
#     _EXCEPTION = 5

#     def __init__(self, constructor, strategy="thread"):
#         self._pickled_ctor = cloudpickle.dumps(constructor)
#         if strategy == "process":
#             contxt = mp.get_context("spawn")
#         elif strategy == "thread":
#             import multiprocessing.dummy as context
#         else:
#             raise NotImplementedError(strategy)

#         self._strategy = strategy
#         self._conn, conn = context.Pipe()
#         self._process = context.Process(target=self._worker, args=(conn,))
#         atexit.register(self.close)
#         self._process.start()
#         self._receive()
#         self._observation_space = None
#         self._action_space = None

#     def _worker(self, conn):
#         try:
#             ctor = cloudpickle.load(self._pickled_ctor)
#             env = ctor()
#             conn.send((self._RESULT, None))
#             while True:
#                 try:
#                     if not conn.poll(0.1):
#                         continue
#                     message, payload = conn.recv()
#                     if message == self._ACCESS:
#                         name = payload
#                         result = getattr(env, name)
#                         conn.send((self._RESULT, result))
#                         continue
#                     if message == self._CALL:
#                         name, args, kwargs = payload
#                         result = getattr(env, name)(args, kwargs)
#                         conn.send((self._RESULT, result))
#                     if message == self._CLOSE:
#                         break
#                     else:
#                         raise KeyError(f"Received message of unkonw type{message}")

#                 except (EOFError, KeyboardInterrupt):
#                     break
#         except Exception:
#             stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
#             conn.send(self._EXCEPTION, stacktrace)
#         finally:
#             try:
#                 conn.close()
#             except IOError:
#                 pass

