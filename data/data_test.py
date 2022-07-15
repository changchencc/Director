import pdb
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import os
import pathlib
import numpy as np
import glob



class EnvIterDatasetTest(IterableDataset):
  data_list = np.arange(20)

  def __init__(self, ):
    super(EnvIterDatasetTest, self).__init__()

  def load_episodes(self, seed=0):
    random = np.random.RandomState(seed)
    while True:
      with open('./data_list.npy', 'rb') as f:
        data_list = np.load(f)
      print(data_list)

      for index in random.choice(len(data_list), 16):

        yield index

  def __iter__(self):
    return self.load_episodes()

if __name__ == '__main__':

  ds = EnvIterDatasetTest()
  dl = DataLoader(ds, batch_size=4, num_workers=2)
  dl_iter = iter(dl)

  total_steps = 1000
  data_list = np.arange(20)
  with open('./data_list.npy', 'wb') as f:
    np.save(f, data_list)

  for i in range(total_steps):

    if i % 16 == 0:
      pdb.set_trace()
      data = next(dl_iter)
      print(f'step: {i},\t data: {data}')

    data_list = np.append(data_list, 101+i)
    with open('./data_list.npy', 'wb') as f:
      np.save(f, data_list)



