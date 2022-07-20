from yacs.config import CfgNode as CN

cfg = CN(
    {
        "exp_name": "",
        "logdir": "/common/users/cc1547/projects/dreamer/dreamer_pytorch/CC/plan_dreamer",
        "resume": True,
        "resume_ckpt": "",
        "debug": False,
        "seed": 0,
        "run_id": "run_0",
        "total_steps": 1e7,
        "model": "dreamer_plan",
        "arch": {
            "use_pcont": True,
            "mem_size": 100000,
            "prefill": 50000,
            "H": 17,
            "norm_target": True,
            "norm_reward": False,
            "worker_only": False,
            "world_model": {
                "norm": "ln",
                "RSSM": {
                    "act": "elu",
                    "weight_init": "xavier",
                    "stoch_size": 32,
                    "stoch_discrete": 32,
                    "deter_size": 1024,
                    "hidden_size": 1024,
                    "rnn_type": "LayerNormGRUV2",
                    "ST": True,
                },
                "reward": {
                    "num_units": 512,
                    "act": "elu",
                    "dist": "normal",
                    "layers": 4,
                    "norm": "ln",
                },
                "pcont": {
                    "num_units": 512,
                    "dist": "binary",
                    "act": "elu",
                    "layers": 4,
                    "norm": "ln",
                },
            },
            "actor": {
                "num_units": 512,
                "act": "elu",
                "init_std": 5.0,
                "dist": "onehot",
                "layers": 4,
                "norm": "none",
                "actor_loss_type": "reinforce",
                "norm": "ln",
            },
            "value": {
                "num_units": 512,
                "act": "elu",
                "dist": "normal",
                "layers": 4,
                "norm": "ln",
            },
            "decoder": {"dec_type": "conv",},
            "manager": {
                "K": 8,
                "cum_intri": True,
                "learn_vae_std": False,
                "actor": {
                    "num_units": 512,
                    "act": "elu",
                    "norm": "ln",
                    "init_std": 5.0,
                    "dist": "onehot",
                    "layers": 4,
                    "aggregator": "none",
                    "action_num": 8,
                    "action_size": 8,
                    "include_deter": False,
                },
                "intri_value": {
                    "num_units": 512,
                    "act": "elu",
                    "norm": "ln",
                    "dist": "normal",
                    "layers": 4,
                },
                "extri_value": {
                    "num_units": 512,
                    "act": "elu",
                    "norm": "ln",
                    "dist": "normal",
                    "layers": 4,
                },
            },
        },
        "loss": {
            "pcont_scale": 5.0,
            "kl_scale": 0.1,
            "free_nats": 0.0,
            "kl_balance": 0.8,
            "mgr_ent_scale": 1e-3,
            "goal_rec_scale": 1.0,
        },
        "env": {
            "action_size": 18,
            "name": "atari_boxing",
            "action_repeat": 1,
            "max_steps": 1000,
            "life_done": False,
            "precision": 16,
            "time_limit": 108000,
            "grayscale": True,
        },
        "rl": {
            "discount": 0.99,
            "lambda_": 0.999,
            "expl_amount": 0.0,
            "expl_decay": 200000.0,
            "expl_min": 0.0,
            "expl_type": "epsilon_greedy",
            "r_transform": "tanh",
        },
        "data": {
            "datadir": "/common/users/cc1547/projects/dreamer/dreamer_pytorch/CC/plan_dreamer",
        },
        "train": {
            "batch_length": 50,
            "batch_size": 50,
            "train_steps": 100,
            "train_every": 16,
            "print_every_step": 2000,
            "log_every_step": 1e4,
            "checkpoint_every_step": 1e4,
            "eval_every_step": 1e5,
            "n_sample": 10,
            "imag_last_T": False,
        },
        "optimize": {
            "model_lr": 1e-4,
            "value_lr": 1e-4,
            "actor_lr": 1e-4,
            "goal_vae_lr": 1e-4,
            "mgr_value_lr": 1e-4,
            "mgr_actor_lr": 1e-4,
            "optimizer": "adam",
            "grad_clip": 100.0,
            "weight_decay": 1e-2,
            "eps": 1e-6,
            "reward_scale": 1.0,
            "discount_scale": 5.0,
        },
        "checkpoint": {
            "checkpoint_dir": "/common/users/cc1547/projects/dreamer/dreamer_pytorch/CC/plan_dreamer",
            "max_num": 10,
        },
    }
)

