import torch
import torchvision.utils as vutils
from model.utils import spatial_transformer
import pdb

COLORS = torch.tensor([
    [255, 255, 255], # white
    [255, 128, 0], # orange
    [0, 255, 0], # lime
    [0, 0, 255], # blue
    [255, 255, 0], # yellow
    [0, 255, 255], # cyan
    [0, 128, 255],
    [255, 0, 255], # magenta
    [127, 0, 255],
    [255, 0, 127],
]) / 255.0
N_COLORS = COLORS.shape[0]

def get_colors(ids):
  return COLORS[ids.long() % N_COLORS]

def get_bbox(color, bbox_h, bbox_w, line_width=1):
  bbox = color[..., None, None].repeat(1, 1, bbox_h, bbox_w)
  bbox[..., line_width:-line_width, line_width:-line_width] = 0.
  return bbox

def draw_bboxes(img, st_xy, st_wh, z_pres, cfg):

  bs, C, H, W = img.shape
  if C == 1:
    img = img.expand(-1, 3, -1, -1)
    C = 3
  K = st_xy.shape[1]

  st_xy = st_xy.reshape(bs*K, 2)
  st_wh = st_wh.reshape(bs*K, 2)

  bbox_color = get_colors(torch.ones(K).reshape(1, K).expand(bs, -1).reshape(bs*K))
  bbox = get_bbox(bbox_color, cfg.PATCH_HW, cfg.PATCH_HW, line_width=3)

  bboxes = spatial_transformer(st_xy,
                                 st_wh,
                                 bbox,
                                 H,
                                 W,
                                 inverse=True)
  bboxes = bboxes.reshape(bs, K, C, H, W)
  img_with_bbox = (img + (bboxes * z_pres.reshape(bs, K, 1, 1, 1).sigmoid()).sum(1)).clamp_max(1.0)

  return img_with_bbox

def log_vis_with_bbox(writer, vis_log, sequence, global_step, cfg, train):

  s_pos = vis_log['s_pos'].cpu()
  s_scale = vis_log['s_scale'].cpu()
  z_pres = vis_log['z_pres'].cpu()
  rec = vis_log['img'].cpu()
  sequence = sequence.cpu()
  if cfg.train.model == 'TGNMConvdraw':
    rec = rec[:, :, -1]
    s_pos = s_pos[:, :, -1]
    s_scale = s_scale[:, :, -1]

  if len(s_pos.shape) == 4:
    bs, _, H, W = s_pos.shape
  elif len(s_pos.shape) == 5:
    bs, T, _, H, W = s_pos.shape
  else:
    raise ValueError('Invalid tensor shape in log_vis_with_bbox')

  if train:
    tag = 'Rec/tgt_vs_rec'

  else:
    tag = 'Gen_x2/tgt_vs_rec'

  if len(s_pos.shape) == 4:
    tgt_with_bbox = draw_bboxes(sequence,
                              s_pos.permute(0, 2, 3, 1).reshape(bs, H * W, cfg.arch.z_pos_dim),
                              s_scale.permute(0, 2, 3, 1).reshape(bs, H * W, cfg.arch.z_scale_dim),
                              z_pres.reshape(bs, H * W, 1), cfg)

    img_vis = vutils.make_grid(
      torch.cat([tgt_with_bbox[:4], rec[:4].expand(-1, 3, -1, -1)], dim=0).clamp(0., 1.),
      nrow=1, pad_value=1)
    writer.add_image(tag, img_vis, global_step + 1)

  else:
    img_vis_list = []
    for t in range(T):
      tgt_with_bbox = draw_bboxes(sequence[:, t],
                                  s_pos[:, t].permute(0, 2, 3, 1).reshape(bs, H*W, cfg.arch.z_pos_dim),
                                  s_scale[:, t].permute(0, 2, 3, 1).reshape(bs, H*W, cfg.arch.z_scale_dim),
                                  z_pres[:, t].reshape(bs, H*W, 1), cfg)

      img_vis = vutils.make_grid(
        torch.stack([tgt_with_bbox[:4], rec[:4, t].expand(-1, 3, -1, -1)], dim=1).clamp(0., 1.).reshape(8, 3, *rec.shape[3:]),
        nrow=1, pad_value=1)
      img_vis_list.append(img_vis)

    writer.add_image(tag, torch.cat(img_vis_list, dim=2), global_step+1)

def log_vis(writer, vis_log, sequence, global_step, cfg, train=True):

  if cfg.train.model in ['GNM', 'TGNM', 'TGNMConvdraw', 'TGNMGatedCNN', 'TGNMWPS', 'TGNMZLocalEnc',
                         'TGNMZLocalEncWithTransitRNN', 'TGNMZLocalEncWithTransitRNNAction',
                         'TGNMZLocalEncWithTransitRIMAction']:
    log_vis_with_bbox(writer, vis_log, sequence, global_step, cfg, train)

  else:

    if train:
      rec = vis_log['rec_x'][:4].cpu() # 4, T, C, H, W
    else:
      rec = vis_log['gen_x'][:4].cpu() # 4, T, C, H, W

    _, T, C, H, W = rec.shape

    tgt = sequence[:4, :T].cpu() # 4, T, C, H, W

    tgt_rec_vis = vutils.make_grid(
        torch.stack([tgt, rec], dim=1).reshape(-1, C, H, W), nrow=T, pad_value=1)
    if train:
      writer.add_image('Rec/tgt-rec-step-{}'.format(global_step), tgt_rec_vis.clamp(0., 1.), global_step)

    else:
      writer.add_image('Gen/tgt-rec-step-{}'.format(global_step), tgt_rec_vis.clamp(0., 1.), global_step)

