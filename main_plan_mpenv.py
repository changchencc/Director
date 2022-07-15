import torch
from configs import cfg
from engine.train_v2_plan_mpenv import Actor, Learner
from model import get_model
import os
import argparse
import pdb
import torch.multiprocessing as mp
from torch.multiprocessing import Event, Value


def get_config():
    parser = argparse.ArgumentParser(description="args for plan_dreamer project")
    parser.add_argument("--config-file", type=str, default="", help="config file")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="using command line to modify configs.",
    )

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    if args.opts:
        cfg.merge_from_list(args.opts)

    if cfg.exp_name == "":
        if not args.config_file:
            raise ValueError(
                "exp name can not be empty when config file is not provided"
            )
        else:
            cfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]

    return cfg


if __name__ == "__main__":
    cfg = get_config()

    mp.set_start_method("spawn")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(cfg, device, cfg.seed).to(device)
    model.share_memory()

    run_step = Value("Q", 0)
    completion = Event()
    learner = Learner(model, completion, cfg, device, run_step)

    actor_steps = []
    for i in range(cfg.env.num_actor):
        actor_steps.append(Value("Q", 0))
    actors = []
    for i in range(cfg.env.num_actor):
        actors.append(Actor(i, model, cfg, device, run_step, completion, actor_steps))

    learner.start()
    for a in actors:
        a.start()

    completion.wait()

    learner.join()
    for a in actors:
        a.join()

