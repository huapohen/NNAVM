import os
import sys
import ipdb
from easydict import EasyDict
from yacs.config import CfgNode as CN


def common_config(cfg):
    if "linux" in sys.platform:
        cfg.data_dir = "/home/data/lwb/data/dybev"
    else:  # windows
        cfg.data_dir = ""
    if not os.path.exists(cfg.data_dir):
        raise ValueError
    cfg.exp_root = 'experiments'
    cfg.exp_name = 'eeavm'
    exp_dir = os.path.join(cfg.exp_root, cfg.exp_name)
    cfg.model_dir = os.path.join(exp_dir, f"exp_{cfg.exp_id}")
    cfg.tb_path = os.path.join(exp_dir, 'tf_log', f'exp_{cfg.exp_id}')
    return cfg


def train_config(cfg):
    cfg.camera_list = ["front"]
    # cfg.camera_list = ["front", 'back']
    # cfg.camera_list = ["front", 'back', 'left', 'right']
    cfg.exp_id = 2
    cfg.gpu_used = '2'
    cfg.num_workers = 16
    cfg.train_batch_size = 32
    cfg.num_epochs = 24
    cfg.train_data_ratio = [["v3", 1]]
    # cfg.src_num_mode = "single_calibrate_image"
    cfg.src_num_mode = "multiple_driving_images"
    cfg.scale_undist = 0.5
    # cfg.model_train_type = "supervised"
    cfg.model_train_type = "unsupervised"
    # cfg.second_stage_image_supervised = True  # exp_8
    # cfg.bev_mask_mode = True # exp_9
    # cfg.eval_freq = 1
    cfg.train_visualize_save = True
    cfg.train_vis_iter_frequence = 20
    cfg.eval_freq = 10000
    cfg.major_metric = 'total_loss'
    cfg.metric_mode = 'descend'
    cfg = continue_train(cfg, 0)
    # cfg.gpu_used = '0_1_2_3_4_5_6_7' # use 8 GPUs
    return cfg


def test_config(cfg, args=None):

    cfg.exp_id = 1
    cfg.gpu_used = '2_5'
    cfg.eval_batch_size = 2
    cfg.test_data_ratio = [["v2", 1]]
    cfg.camera_list = ["front"]
    # cfg.model_train_type = "supervised"
    cfg.model_train_type = "unsupervised"
    cfg.dataset_type = "test"
    cfg.eval_visualize_save = True
    # cfg.eval_visualize_save = False
    # cfg.restore_file = 'eeavm_test_model_best.pth'
    cfg.restore_file = "eeavm_model_latest.pth"

    if 'exp_id' in vars(args):
        cfg.exp_id = args.exp_id

    if cfg.model_train_type == 'supervised':
        cfg.major_metric = 'homing_point_ratio'
        cfg.metric_mode = "ascend"
    elif cfg.model_train_type == 'unsupervised':
        cfg.major_metric = "psnr"

    return cfg


def continue_train(cfg, mode=0):
    assert mode in [0, 1]
    if mode == 1:
        # cfg.restore_file = 'eeavm_test_model_best.pth'
        cfg.restore_file = "eeavm_model_latest.pth"
        cfg.only_weights = True
        # cfg.only_weights = False
    return cfg


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d


def get_config(args=None, mode='train'):
    """Get a yacs CfgNode object with debug params values."""
    cfg = CN()

    assert mode in ['train', 'test', 'val', 'evaluate']

    if mode == 'train':
        cfg = train_config(cfg)
    else:
        cfg = test_config(cfg, args)

    cfg = common_config(cfg)

    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = cfg.clone()
    config.freeze()

    return config


'''
cfg = get_config(None, 'train'))
dic = json.loads(json.dumps(cfg))
'''
