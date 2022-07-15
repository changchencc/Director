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


class IMaze3D16Area:
  LOCK = threading.Lock()

  def __init__(self, name, env_file, id, action_size, callbacks, max_steps, num_area=16,
               size=(64, 64), seed=0, life_done=False, precision=32):
    assert size[0] == size[1]

    with self.LOCK:
      env = UnityEnvironment(file_name=env_file, worker_id=id, seed=seed, timeout_wait=1000)
      env.reset()
    # Avoid unnecessary rendering in inner env.
    # Tell wrapper that the inner env has no action repeat.
    self._env = env
    self._size = size
    self.action_size = action_size
    self.life_done = life_done
    self.num_area = num_area
    self.behavior_names = list(env.behavior_specs)[0]
    self._episodes = [[] for _ in range(self.num_area)]
    self._callbacks = callbacks
    self._steps = 0
    self.max_steps = max_steps
    self._precision = precision

  @property
  def observation_space(self):
    shape = (1 if self._grayscale else 3,) + self._size
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      pdb.set_trace()
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)

  def sample_random_action(self):
    action = np.zeros((self.num_area, self.action_size,), dtype=np.float)
    idx = np.random.randint(0, self.action_size, size=(self.num_area,))
    action[np.arange(self.num_area), idx] = 1
    return action

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      self._env.reset()

    decision_steps, terminal_steps = self._env.get_steps(self.behavior_names)

    images = decision_steps.obs[0] * 255.  # N, H, W, C
    images = np.transpose(images, (0, 3, 1, 2)).astype(np.uint8)
    top_views = decision_steps.obs[1] * 255.
    top_views = np.transpose(top_views, (0, 3, 1, 2)).astype(np.uint8)
    rewards = decision_steps.reward

    for i in range(self.num_area):
      transition = {}
      transition['image'] = images[i]
      transition['top_view'] = top_views[i]
      transition['reward'] = rewards[i]
      transition['discount'] = 1.0
      transition['done'] = 0.0
      transition['action'] = np.array([1, 0, 0, 0, 0], dtype=np.float32)
      transition['episode_done'] = 0.
      transition['succeed'] = 0.
      self._episodes[i] = [transition]

    return {'image': images, 'top_view': top_views}

  def step(self, action):
    self._steps += 1
    action_scalar = action.argmax(axis=-1).reshape(self.num_area, 1)
    action_scalar = action_scalar + 1  # unity reserve action 0 as staying still
    action_tuple = ActionTuple()
    action_tuple.add_discrete(action_scalar)

    self._env.set_actions(self.behavior_names, action_tuple)
    self._env.step()
    decision_steps, terminal_steps = self._env.get_steps(self.behavior_names)

    if not self.life_done:
      if len(terminal_steps) > 0:
        agent_ids = terminal_steps.agent_id
        for agent_id in agent_ids:
          image = terminal_steps.obs[0][agent_id] * 255.
          image = np.transpose(image, (2, 0, 1)).astype(np.uint8)
          top_view = terminal_steps.obs[1][agent_id]
          top_view = np.transpose(top_view, (2, 0, 1)).astype(np.uint8)
          reward = terminal_steps.reward[agent_id]

          transition = {}
          transition['image'] = image
          transition['top_view'] = top_view
          transition['action'] = action[agent_id]
          transition['reward'] = reward
          transition['done'] = 1.
          transition['discount'] = 0.
          self._episodes[agent_id].append(transition)
          episode = {k: [t[k] for t in self._episodes[agent_id]] for k in self._episodes[agent_id][0]}
          episode = {k: self._convert(v) for k, v in episode.items()}
          for callback in self._callbacks:
            callback(episode)

          self._episodes[agent_id] = []
          self._env.step()
          decision_steps, terminal_steps = self._env.get_steps(self.behavior_names)

      images = decision_steps.obs[0] * 255.  # N, H, W, C
      images = np.transpose(images, (0, 3, 1, 2)).astype(np.uint8)
      top_views = decision_steps.obs[1] * 255.
      top_views = np.transpose(top_views, (0, 3, 1, 2)).astype(np.uint8)
      rewards = decision_steps.reward

      for i in range(self.num_area):
        transition = {}
        transition['image'] = images[i]
        transition['top_view'] = top_views[i]
        transition['reward'] = rewards[i]
        transition['discount'] = 1.0
        transition['done'] = 0.0
        transition['action'] = action[i]
        self._episodes[i].append(transition)

      done = 0.0

    else:
      done = 0.0
      term_agent_ids = terminal_steps.agent_id
      if len(terminal_steps) > 0:  # it might because some agents have terminated
        if len(terminal_steps) == self.num_area:
          done = 1.0
          # print(f'steps: {self._steps}')
          self._steps = 0
        images = terminal_steps.obs[0] * 255.  # N, H, W, C
        images = np.transpose(images, (0, 3, 1, 2)).astype(np.uint8)
        top_views = terminal_steps.obs[1] * 255.
        top_views = np.transpose(top_views, (0, 3, 1, 2)).astype(np.uint8)
        term_rewards = terminal_steps.reward

        # print(f'{len(term_agent_ids)} terminated')
        # print(f'terminated reward: {term_rewards}')
        for i, agent_id in enumerate(term_agent_ids):
          transition = {}
          transition['image'] = images[i]
          transition['top_view'] = top_views[i]
          transition['reward'] = term_rewards[i]
          transition['discount'] = 1.0
          transition['done'] = done
          transition['action'] = action[agent_id]
          transition['episode_done'] = 1.
          transition['succeed'] = np.float32((term_rewards[i] > 2.0))
          self._episodes[agent_id].append(transition)

      if done:
        for i in range(self.num_area):
          for callback in self._callbacks:
            episode = {k: [t[k] for t in self._episodes[i]] for k in self._episodes[i][0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            callback(episode)
          self._episodes[i] = []

        obs = {'image': images, 'top_view': top_views}

        return obs, term_rewards, done

        # then retrive new data if not done
      else:

        if self._steps > self.max_steps:
          done = 1.0
          self._steps = 0

        i = 0
        while (len(decision_steps) == 0):
          self._env.step()
          decision_steps, terminal_steps = self._env.get_steps(self.behavior_names)
          i = i+1
          if i > 5:
            print(f'decision_steps: {len(decision_steps)}')
            print(f'terminal_steps: {len(terminal_steps)}')
            print(f'break! ')
            break

        images = decision_steps.obs[0] * 255.  # N, H, W, C
        images = np.transpose(images, (0, 3, 1, 2)).astype(np.uint8)
        top_views = decision_steps.obs[1] * 255.
        top_views = np.transpose(top_views, (0, 3, 1, 2)).astype(np.uint8)
        rewards = decision_steps.reward

        for i in range(self.num_area):
          transition = {}
          transition['image'] = images[i]
          transition['top_view'] = top_views[i]
          transition['reward'] = rewards[i]
          transition['discount'] = 1.0
          transition['done'] = done
          transition['action'] = action[i]
          transition['episode_done'] = 0.
          transition['succeed'] = 0.0
          self._episodes[i].append(transition)

        obs = {'image': images, 'top_view': top_views}

        if done:
          for i in range(self.num_area):
            for callback in self._callbacks:
              episode = {k: [t[k] for t in self._episodes[i]] for k in self._episodes[i][0]}
              episode = {k: self._convert(v) for k, v in episode.items()}
              callback(episode)
            self._episodes[i] = []

        return obs, rewards, done
