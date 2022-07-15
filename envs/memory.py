import torch
import numpy as np
import pdb

class Memory:
  def __init__(
    self,
    size,
    action_size,
  ):
    self.idx = 0
    self.steps = 0
    self.episodes = 0
    self.size = size
    self.full = False
    self.action_size = action_size

    self.observations = torch.zeros((size, 3, 64, 64), dtype=torch.float)
    self.actions = torch.zeros((size, action_size), dtype=torch.float)
    self.rewards = torch.zeros((size,), dtype=torch.float)
    self.dones = torch.zeros((size,), dtype=torch.float)

  def add( self, observation, action, reward, done):
    """

    :param observation: tensor, (3, 64, 64)
    :param action:  tensor, (7, )
    :param reward: scalar
    :param done: boolean
    :return:
    """

    self.observations[self.idx].copy_(observation)
    self.actions[self.idx].copy_(action)
    self.rewards[self.idx].copy_(torch.tensor(reward).float())
    self.dones[self.idx].copy_(torch.tensor(done).float())

    self.idx = (self.idx + 1) % self.size
    self.full = self.full or (self.idx == 0)
    self.steps = self.steps + 1
    self.episodes = self.episodes + 1 if done else self.episodes
  
  def sample_idx(self, length):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - length)
      idxs = np.arange(idx, idx + length) % self.size
      valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
    return idxs

  def retrieve_batch(self, indices, batch_size, length):

    vec_idxs = indices.reshape(-1)
    o = self.observations[vec_idxs]
    o = o.reshape(batch_size, length, 3, 64, 64)
    a = self.actions[vec_idxs].reshape(batch_size, length, self.action_size)
    d = self.dones[vec_idxs].reshape(batch_size, length)
    r = self.rewards[vec_idxs].reshape(batch_size, length)

    traj = {
      'observations': o,
      'actions': a,
      'dones': d,
      'rewards': r
    }
    return traj 

  def sample(self, batch_size, length):

    batch = self.retrieve_batch(np.asarray([self.sample_idx(length) for _ in range(batch_size)]), batch_size, length)
    return batch
