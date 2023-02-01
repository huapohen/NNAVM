# -*- coding: utf-8 -*-
import os
import sys
import json
import thop
import time
import ipdb
import shutil
import warnings
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from ipdb import set_trace
from easydict import EasyDict

import torch
import torch.optim as optim
import torch.nn.functional as F

import dataset.data_loader as data_loader
import model.net as net

from common import utils
from common.manager import Manager
from evaluate import evaluate
from loss.losses import compute_losses
from util.preprocess import to_cuda
from util.postprocess import visulize_results
from parameters import get_config, dictToObj

warnings.filterwarnings("ignore")

''' add this line, because GTX30 serials' default torch.matmul() on cuda is uncorrected '''
torch.backends.cuda.matmul.allow_tf32 = False


def p(*arg, **kwargs):
    return print(*arg, **kwargs)


torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default=None, help="params file")
parser.add_argument(
    "--model_dir", type=str, default=None, help="Directory containing params.json"
)
parser.add_argument(
    "--restore_file",
    default=None,
    help="Optional, name of the file in --model_dir "
    + "containing weights to reload before training",
)
parser.add_argument(
    "-ow",
    "--only_weights",
    action="store_true",
    help="Only use weights to load or load all train status.",
)
parser.add_argument(
    "-gu",
    "--gpu_used",
    type=str,
    default=None,
    help="select the gpu for train or evaluation.",
)
parser.add_argument(
    "-exp", "--exp_name", type=str, default=None, help="experiment name."
)
parser.add_argument(
    "-tb",
    "--tb_path",
    type=str,
    default=None,
    help="the path to save the tensorboardx log.",
)


def train(manager):
    manager.reset_loss_status()

    torch.cuda.empty_cache()
    manager.model.train()
    params = manager.params
    params.visualize_mode = 'train'

    iter_max = len(manager.dataloaders["train"])
    assert iter_max > 0, "\t\t\t\t empty input"
    params.iter_max = iter_max

    with tqdm(total=iter_max) as t:
        for idx, data in enumerate(manager.dataloaders["train"]):
            params.current_iter = idx + 1

            data = to_cuda(params, data)

            data['offset_pred'] = manager.model(data['image'])
            data = net.second_stage(params, data)

            losses = compute_losses(params, data)

            manager.optimizer.zero_grad()
            # with torch.autograd.set_detect_anomaly(True):
            with torch.autograd.set_detect_anomaly(False):
                losses["total_loss"].backward()

            manager.update_loss_status(loss=losses, split="train")

            manager.optimizer.step()

            manager.update_step()

            if params.train_visualize_save:
                visulize_results(params, data)

            print_str = manager.print_train_info()

            t.set_description(desc=print_str)
            t.update()

            # break
            # sys.exit()

    manager.logger.info(print_str)
    manager.scheduler.step()

    manager.update_epoch()


def train_and_evaluate(manager):

    epoch_start = 0
    # reload weights from restore_file if specified
    if args.restore_file is not None:
        manager.load_checkpoints()
        epoch_start = manager.train_status["epoch"]

    if not isinstance(epoch_start, int):
        epoch_start = 0

    for epoch in range(epoch_start, manager.params.num_epochs):
        manager.params.current_epoch = epoch + 1

        # evaluate the net
        if epoch % manager.params.eval_freq == 0:
            if epoch != 0 or manager.params.is_eval_first:
                evaluate(manager)

        # Save latest model, or best model weights accroding to the params.major_metric
        manager.check_best_save_last_checkpoints(latest_freq=1)

        # compute number of batches in one epoch (one full pass over the training set)
        train(manager)

    # finished train, evaluate
    evaluate(manager)
    with open(os.path.join(manager.params.model_dir, "finish_flag.txt"), "w") as f:
        f.write("exp finish!")


if __name__ == "__main__":

    args = parser.parse_args()

    if args.params_path is not None and args.restore_file is None:
        # run train by DIY in designated diy_params.json
        '''python train.py --params diy_param_json_path'''
        params = utils.Params(args.params_path)
        exp_dir = os.path.join(params.exp_root, params.exp_name)
        params.model_dir = os.path.join(exp_dir, f"exp_{params.exp_id}")
        params.tb_path = os.path.join(exp_dir, 'tf_log', f'exp_{params.exp_id}')
    elif args.gpu_used is not None:
        # run train.py through search_hyperparams.py
        default_json_path = os.path.join("experiments", "params.json")
        params = utils.Params(default_json_path)
        try:
            shutil.rmtree(args.model_dir)
            shutil.rmtree(args.tb_path)
        except:
            pass
        model_json_path = os.path.join(args.model_dir, "params.json")
        model_params_dict = utils.Params(model_json_path).dict
        params.update(model_params_dict)
    else:
        # run by python train.py
        cfg = get_config(args, mode='train')
        dic_params = json.loads(json.dumps(cfg))
        obj_params = dictToObj(dic_params)
        default_json_path = os.path.join("experiments", "params.json")
        params = utils.Params(default_json_path)
        params.update(obj_params)
        file_name = f"{params.exp_name}_exp_{params.exp_id}.json"
        extra_config_json_path = os.path.join("experiments", 'config')
        exp_json_path1 = os.path.join(params.model_dir, "params.json")
        exp_json_path2 = os.path.join(extra_config_json_path, file_name)
        # resume
        if 'restore_file' not in obj_params:
            try:
                shutil.rmtree(params.model_dir)
                shutil.rmtree(params.tb_path)
                os.remove(exp_json_path2)
            except:
                pass
        else:
            model_json_path = os.path.join(obj_params.model_dir, "params.json")
            params_model = utils.Params(model_json_path)
            params.update(params_model.dict)

    os.makedirs(params.model_dir, exist_ok=True)
    os.makedirs(params.tb_path, exist_ok=True)
    os.makedirs(extra_config_json_path, exist_ok=True)

    # Assign dataset
    if params.eval_freq < params.num_epochs:
        params.dataset_type = 'basic'

    # Save params
    if 'restore_file' not in vars(params):
        params.save(exp_json_path1)
        params.save(exp_json_path2)

    # Set the logger
    logger = utils.set_logger(os.path.join(params.model_dir, "train.log"))

    # Set the tensorboard writer
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # use GPU if available
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if "_" in params.gpu_used:
        params.gpu_used = ",".join(params.gpu_used.split("_"))
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_used
    # cuda
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # init model
    model = net.fetch_net(params)

    # flop, parameters
    if params.is_calc_flops:
        input = torch.randn(1, params.in_channel, params.img_h, params.img_w)
        flops, parameters = thop.profile(model, inputs=(input,), verbose=False)
        model.eval()
        output = model(input)
        split_line = "=" * 31
        prt_model_info = f"""
                            {split_line}
                            Input  shape: {tuple(input.shape[1:])}
                            Output shape: {tuple(output.shape[1:])}
                            Flops: {flops / 1e6:.1f} M
                            Params: {parameters / 1e6:.1f} M
                            {split_line}"""
        logger.info(prt_model_info)

    # set gpu-mode
    if params.cuda:
        gpu_num = len(params.gpu_used.split(","))
        device_ids = range(gpu_num)
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate)

    # learning rate
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=params.step_size, gamma=params.gamma
    )

    logger.info("Loading the train datasets from {}".format(params.data_dir))

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Dataset information
    for set_mode, set_dl in dataloaders.items():
        sample_info = set_dl.sample_number
        ds_stats = ""
        for k, v in sample_info.items():
            if k != 'total_samples':
                ds_stats += f" {k}  r={v['ratio']} n={v['samples']}\t"
        logger.info(f"{set_mode} dataset: {ds_stats}")
        logger.info(f"total samples: {sample_info['total_samples']}")

    # Initial status for checkpoint manager
    assert params.metric_mode in ["ascend", "descend"]
    manager = Manager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        params=params,
        dataloaders=dataloaders,
        writer=None,
        logger=logger,
        exp_name=params.exp_name,
        tb_path=params.tb_path,
    )

    # Continue training
    if 'restore_file' in vars(params):
        manager.load_checkpoints()

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(manager)
