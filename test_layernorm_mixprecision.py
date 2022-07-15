from model.utils import LayerNormGRUCell, get_parameters, get_named_parameters, FreezeParameters
import torch.nn as nn
import torch
import pdb




class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc_in = torch.nn.Linear(20, 20)
    self.rnn_cell = LayerNormGRUCell(20, 10)
    self.fc_out = torch.nn.Linear(10, 10)
    self.loss = nn.MSELoss()
    self.module_1 = [self.fc_in, self.rnn_cell]
    self.module_2 = [self.fc_out]

  def forward(self, ipts, gt):

    o = self.fc_in(ipts)
    for t in range(3):
      if t == 0:
        h = self.rnn_cell.init_state(o.shape[0])
      h = self.rnn_cell(o, h)

    loss_1 = self.loss(h, gt)

    with FreezeParameters(self.module_1):
      o = self.fc_in(ipts)
      # h = h.detach()
      for t in range(3):
        h = self.rnn_cell(o, h)
      o = self.fc_out(h)
      loss_2 = self.loss(o, gt)

    return loss_1, loss_2

device = torch.device('cuda')

dummy_input = torch.randn(3, 20).to(device)
dummy_target = torch.randn(3, 10).to(device)

model = Net()
model = model.to(device)
named_params = get_named_parameters([model])
optimizer_1 = torch.optim.Adam(get_parameters(model.module_1), lr=1e-3)
optimizer_2 = torch.optim.Adam(get_parameters(model.module_2), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()

for i in range(38):
  optimizer_1.zero_grad()
  optimizer_2.zero_grad()

  loss_1, loss_2 = model(dummy_input, dummy_target)
  scaler.scale(loss_1).backward()
  scaler.unscale_(optimizer_1)
  grad_norm_value = torch.nn.utils.clip_grad_norm_(get_parameters(model.module_1), 100.)
  scaler.step(optimizer_1)

  scaler.scale(loss_2).backward()
  scaler.unscale_(optimizer_2)
  grad_norm_value = torch.nn.utils.clip_grad_norm_(get_parameters(model.module_2), 100.)
  scaler.step(optimizer_2)

  scaler.update()
  print(f'iter-{i}, loss: {loss_1.detach() + loss_2.detach()}')

