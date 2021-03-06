import sys, os
sys.path.insert(0, os.getcwd())
import json
import time
import argparse
from copy import deepcopy
from easydict import EasyDict as edict
from main.train import train_with_config
import multiprocessing as mp

available_gpus = []
n_concurrent = 0
default_args = edict({
    "n_epochs": 15,
    "n_workers": 4,
    "seed": 999,
    "val_freq": 1,
    "grad_freq": 0
})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--item', type=str, help="item to tune, e.g. 'learning_rate'")
    parser.add_argument('--config', type=str, default='configs/vgg19-512-14-7-7-GRU-300d-1layer-5-0-256-128-0.2-0.2-1.0-1001-gt-311-100000-1e-4-0.8.json')
    parser.add_argument('--gpus', type=str, help="comma separated gpu_ids to use, e.g. '2,5,6'")
    parser.add_argument('--np', type=int, help="number of concurrent processes")
    parser.add_argument('--preload', action="store_true")
    args = parser.parse_args()
    args.gpus = args.gpus.split(",")
    default_args.preload = args.preload
    return args

def get_cfg_name(cfg):
    cfg_name = ""
    cfg_name += cfg.vision_model.backbone
    cfg_name += "-%d" % cfg.vision_model.emb_dim
    cfg_name += "-%d" % cfg.vision_model.feature_height
    cfg_name += "-%d" % cfg.vision_model.rel_crop_size
    cfg_name += "-%d" % cfg.vision_model.ent_crop_size
    cfg_name += "-%s" % cfg.language_model.rnn_type
    cfg_name += "-%d" % cfg.language_model.word_emb_dim
    cfg_name += "-%dlayer" % cfg.language_model.n_layers
    cfg_name += "-%d" % cfg.language_model.tokens_length
    cfg_name += "-%d" % cfg.language_model.n_attrs
    cfg_name += "-%d" % cfg.train.batch_size
    cfg_name += "-%d" % cfg.ent_loss.n_neg
    cfg_name += "-%.1f" % cfg.ent_loss.margin
    cfg_name += "-%.1f" % cfg.rel_loss.margin
    cfg_name += "-%.1f" % cfg.ent_loss.similarity_norm
    cfg_name += "-"
    cfg_name += "1" if cfg.ent_loss.x_tr else "0"
    cfg_name += "1" if cfg.ent_loss.x_trsm else "0"
    cfg_name += "1" if cfg.ent_loss.y_tr else "0"
    cfg_name += "1" if cfg.ent_loss.y_trsm else "0"
    cfg_name += "-%s" % cfg.train.box_source
    cfg_name += "-%d" % cfg.n_preds
    cfg_name += "-%d" % cfg.n_rel_max
    cfg_name += "-%.0e" % cfg.train.learning_rate
    cfg_name += "-%.1f" % cfg.train.learning_rate_decay
    return cfg_name

def tune_backbone(base_cfg):
    values = ["vgg19", "resnet101"]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.vision_model.backbone = value
        cfg_exp.vision_model.cache_dir = "cache/gqa_%s_%dx%d" % (
            value, cfg_exp.vision_model.feature_height, cfg_exp.vision_model.feature_width)
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/backbone"
    run_configs(args, cfgs)

def tune_crop_size(base_cfg):
    sizes = [ (7, 7), (7, 5), (7, 3), (5, 5), (5, 3) ]
    cfgs = []
    for rel_size, ent_size in sizes:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.vision_model.rel_crop_size = rel_size
        cfg_exp.vision_model.ent_crop_size = ent_size
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/crop_size"
    run_configs(args, cfgs)

def tune_rnn_type(base_cfg):
    values = [ "GRU", "LSTM" ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.language_model.rnn_type = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/rnn_type"
    run_configs(args, cfgs)

def tune_rnn_layers(base_cfg):
    values = [ 1, 2, 3 ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.language_model.n_layers = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/rnn_layers"
    run_configs(args, cfgs)

def tune_n_attrs(base_cfg):
    values = [ 2, 3, 4, 5 ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.language_model.n_attrs = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/n_attrs"
    run_configs(args, cfgs)

def tune_sampling(base_cfg):
    values = [ 32, 128, 256  ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.ent_loss.n_neg = int(value/2)
        cfg_exp.rel_loss.n_neg = int(value / 2)
        cfg_exp.train.batch_size = value
        cfg_exp.val.batch_size = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/sampling"
    run_configs(args, cfgs)

def tune_n_neg(base_cfg):
    values = [ 32, 64, 128  ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.ent_loss.n_neg = value
        cfg_exp.rel_loss.n_neg = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/n_neg"
    run_configs(args, cfgs)

def tune_margin(base_cfg):
    values = [ 0.05, 0.1, 0.2, 0.4 ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.ent_loss.margin = value
        cfg_exp.rel_loss.margin = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/margin"
    run_configs(args, cfgs)

def tune_similarity_norm(base_cfg):
    values = [ 1.0, 2.5, 5.0, 7.5, 10.0 ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.ent_loss.similarity_norm = value
        cfg_exp.rel_loss.similarity_norm = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/similarity_norm"
    run_configs(args, cfgs)

def tune_learning_rate(base_cfg):
    values = [ 0.01, 0.005, 0.001, 0.0005 ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.train.learning_rate = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/learning_rate"
    run_configs(args, cfgs)

def tune_learning_rate_decay(base_cfg):
    values = [ 0.8, 0.6, 0.4, 0.2 ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.train.learning_rate_decay = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/learning_rate_decay"
    run_configs(args, cfgs)

def tune_loss_composition(base_cfg):
    values = [ (1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0), (1, 1, 1, 1) ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.ent_loss.x_tr = bool(value[0])
        cfg_exp.ent_loss.x_trsm = bool(value[1])
        cfg_exp.ent_loss.y_tr = bool(value[2])
        cfg_exp.ent_loss.y_trsm = bool(value[3])
        cfg_exp.rel_loss.x_tr = bool(value[0])
        cfg_exp.rel_loss.x_trsm = bool(value[1])
        cfg_exp.rel_loss.y_tr = bool(value[2])
        cfg_exp.rel_loss.y_trsm = bool(value[3])
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/loss_composition"
    run_configs(args, cfgs)

def tune_data_distribution(base_cfg):
    values = [ 10000, 20000, 50000, 100000 ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.n_rel_max = value
        cfg_exp.train.triples_path = "cache/train_triples_gt_boxes_use_none_311_max_%d.pkl" % value
        cfg_exp.val.triples_path = "cache/val_triples_gt_boxes_use_none_311_max_%d.pkl" % value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/data_distribution"
    run_configs(args, cfgs)

def run_configs(args, cfgs):
    tasks = []
    for cfg in cfgs:
        cfg_name = get_cfg_name(cfg)
        args = edict(deepcopy(args))
        args.cfg_name = cfg_name
        tasks.append((args, cfg))
    pool = []
    gpu_indices = []
    gpu_status = [ 0 for _ in range(len(available_gpus)) ]
    for task in tasks:
        args, cfg = task
        if len(pool) < n_concurrent:
            gpu_idx = gpu_status.index(min(gpu_status))
            gpu_status[gpu_idx] += 1
            args.gpu_id = available_gpus[gpu_idx]
            runner = mp.Process(target=train_with_config, args=(args, cfg))
            pool.append(runner)
            gpu_indices.append(gpu_idx)
            pool[-1].start()
        else:
            started = False
            while not started:
                time.sleep(10)
                for i in range(len(pool)):
                    if not pool[i].is_alive():
                        gpu_status[gpu_indices[i]] -= 1
                        gpu_idx = gpu_status.index(min(gpu_status))
                        gpu_indices[i] = gpu_idx
                        gpu_status[gpu_idx] += 1
                        args.gpu_id = available_gpus[gpu_idx]
                        pool[i] = mp.Process(target=train_with_config, args=(args, cfg))
                        pool[i].start()
                        started = True
                        break
    for runner in pool:
        runner.join()

tune_fns = {
    "backbone": tune_backbone,
    "crop_size": tune_crop_size,
    "rnn_type": tune_rnn_type,
    "rnn_layers": tune_rnn_layers,
    "n_attrs": tune_n_attrs,
    "sampling": tune_sampling,
    "margin": tune_margin,
    "similarity_norm": tune_similarity_norm,
    "learning_rate": tune_learning_rate,
    "learning_rate_decay": tune_learning_rate_decay,
    "loss_composition": tune_loss_composition,
    "data_distribution": tune_data_distribution,
    "n_neg": tune_n_neg
}

if __name__ == "__main__":
    run_args = parse_args()

    for arg in vars(run_args):
        print("%10s: %s" % (arg, str(getattr(run_args, arg))))
    print()

    available_gpus = run_args.gpus
    n_concurrent = run_args.np
    base_config_path = run_args.config
    tune_fn = tune_fns[run_args.item]
    base_cfg = edict(json.load(open(base_config_path)))
    tune_fn(base_cfg)