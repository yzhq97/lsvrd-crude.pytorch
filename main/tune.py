import sys, os
sys.path.insert(0, os.getcwd())
import json
import time
from copy import deepcopy
from easydict import EasyDict as edict
from main.train import train_with_config
import multiprocessing as mp

default_args = edict({
    "n_epochs": 20,
    "n_workers": 2,
    "seed": 999,
    "val_freq": 1,
    "grad_freq": 100,
})

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
    cfg_name += "-%d" % cfg.train.batch_size
    cfg_name += "-%d" % cfg.loss_model.n_neg
    cfg_name += "-%.1f" % cfg.loss_model.margin
    cfg_name += "-%.1f" % cfg.loss_model.similarity_norm
    cfg_name += "-"
    cfg_name += "1" if cfg.loss_model.x_tr else "0"
    cfg_name += "1" if cfg.loss_model.x_trsm else "0"
    cfg_name += "1" if cfg.loss_model.y_tr else "0"
    cfg_name += "1" if cfg.loss_model.y_trsm else "0"
    cfg_name += "-%s" % cfg.train.box_source
    cfg_name += "-%d" % cfg.n_preds
    cfg_name += "-%d" % cfg.n_rel_max
    return cfg_name

def tune_backbone(base_cfg):
    values = ["vgg19", "resnet101"]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.vision_model.backbone = value
        cfg_exp.vision_model.cache_dir = "gqa_%s_%dx%d" % (
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

def tune_rnn_types(base_cfg):
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

def tune_sampling(base_cfg):
    values = [ 32, 128, 256  ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.loss_model.n_neg = int(value/2)
        cfg_exp.train.batch_size = value
        cfg_exp.val.batch_size = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/sampling"
    run_configs(args, cfgs)

def tune_margin(base_cfg):
    values = [ 0.1, 0.2, 0.3, 0.4, 0.5 ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.loss_model.margin = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/margin"
    run_configs(args, cfgs)

def tune_similarity_norm(base_cfg):
    values = [ 1.0, 2.5, 5.0, 7.5, 10.0 ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.loss_model.similarity_norm = value
        cfgs.append(cfg_exp)
    args = edict(deepcopy(default_args))
    args.out_dir = "out/similarity_norm"
    run_configs(args, cfgs)

def tune_loss_composition(base_cfg):
    values = [ (1, 0, 0, 1), (0, 1, 1, 0), (1, 1, 1, 1) ]
    cfgs = []
    for value in values:
        cfg_exp = edict(deepcopy(base_cfg))
        cfg_exp.loss_model.x_tr = bool(value[0])
        cfg_exp.loss_model.x_trsm = bool(value[1])
        cfg_exp.loss_model.y_tr = bool(value[2])
        cfg_exp.loss_model.y_trsm = bool(value[3])
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

def run_configs(args, cfgs, n_concurrent=4):
    tasks = []
    for cfg in cfgs:
        cfg_name = get_cfg_name(cfg)
        args = edict(deepcopy(args))
        args.cfg_name = cfg_name
        tasks.append((args, cfg))
    pool = []
    for task in tasks:
        args, cfg = task
        if len(pool) < n_concurrent:
            runner = mp.Process(target=train_with_config, args=(args, cfg))
            pool.append(runner)
            pool[-1].start()
        else:
            while True:
                time.sleep(10)
                for i in range(len(pool)):
                    if not pool[i].is_alive():
                        pool[i] = mp.Process(target=train_with_config, args=(args, cfg))
                        pool[i].start()
                        break
    for runner in pool:
        runner.join()


if __name__ == "__main__":
    base_config_path = "configs/vgg19-512-14-7-7-GRU-300d-1layer-64-32-0.2-5.0-1001-gt-311-100000.json"
    base_cfg = edict(json.load(open(base_config_path)))
    tune_sampling(base_cfg)