# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import thop
import shutil
import logging
import warnings
import datetime
import argparse
import platform
import numpy as np
from tqdm import tqdm
from ipdb import set_trace
from easydict import EasyDict

import torch
from torch.autograd import Variable

import dataset.data_loader as data_loader
import model.net as net

from common import utils
from common.manager import Manager
from loss.losses import compute_losses
from loss.benchmark import benchmark_indicator
from util.preprocess import to_cuda
from util.visualize import visulize_results
from parameters import get_config, dictToObj

warnings.filterwarnings("ignore")

''' add this line, because GTX30 serials' default torch.matmul() on cuda is uncorrected '''
torch.backends.cuda.matmul.allow_tf32 = False


result_all_exps = {}

parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default=None, help="params file")


def evaluate(manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    print("============== eval begin ==============")

    # loss status and eval status initial
    manager.reset_loss_status()
    manager.reset_metric_status("test")
    manager.model.eval()
    params = manager.params
    params.visualize_mode = 'evaluate'
    params.train_eval_inference = 'eval'

    Metric = EasyDict(
        {"total_loss": [], 'homing_point_ratio': [], 'mean_pixel_err': [], 'psnr': []}
    )

    with torch.no_grad():

        iter_max = len(manager.dataloaders["test"])
        assert iter_max > 0, "\t\t\t\t empty input"
        params.iter_max = iter_max

        with tqdm(total=iter_max) as t:
            for idx, data in enumerate(manager.dataloaders["test"]):
                params.current_iter = idx + 1

                data = to_cuda(params, data)

                data['offset_pred'] = manager.model(data['image'])
                data = net.second_stage(params, data)

                losses = compute_losses(params, data)
                indicator = benchmark_indicator(params, data)

                Metric = gather_metric(Metric, losses, indicator)

                if params.eval_visualize_save:
                    visulize_results(params, data)
                    # sys.exit()

                t.update()

                # break

        Metric = statistic_metric(Metric)

        # result_all_exps[params.exp_id] = Metric

        manager.update_metric_status(
            metrics=Metric,
            split="test",
            batch_size=params.eval_batch_size,
        )

        # update data to logger
        manager.logger.info(
            "Loss/valid epoch_{} {}: "
            "total_loss: {:.6f} | ".format("test", manager.epoch_val, Metric.total_loss)
        )

        # For each epoch, print the metric
        manager.print_metrics("test", title="test", color="green")

        # manager.epoch_val += 1
        manager.model.train()


def run_all_exps(exp_id):
    """
    Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()

    if args.params_path is not None and args.restore_file is None:
        # run test by DIY in designated diy_params.json
        '''python evaluate.py --params_path diy_param_json_path'''
        params = utils.Params(args.params_path)
        exp_dir = os.path.join(params.exp_root, params.exp_name)
        params.model_dir = os.path.join(exp_dir, f"exp_{params.exp_id}")
        params.tb_path = os.path.join(exp_dir, 'tf_log', f'exp_{params.exp_id}')
    else:
        # run by python train.py
        if exp_id is not None:
            args.exp_id = exp_id
        cfg = get_config(args, mode='test')
        dic_params = json.loads(json.dumps(cfg))
        obj_params = dictToObj(dic_params)
        params_default_path = os.path.join(obj_params.exp_root, 'params.json')
        model_json_path = os.path.join(obj_params.model_dir, "params.json")
        assert os.path.isfile(
            model_json_path
        ), "No json configuration file found at {}".format(model_json_path)
        params = utils.Params(params_default_path)
        params_model = utils.Params(model_json_path)
        params.update(params_model.dict)
        params.update(obj_params)

    # Only load model weightsn
    params.only_weights = True

    # Use GPU if available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_used
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # Get the logger
    logger = utils.set_logger(os.path.join(params.model_dir, "evaluate.log"))

    logger.info(f"exp_id: {params.exp_id}")
    logger.info("Loading the train datasets from {}".format(params.data_dir))

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Dataset information
    sample_info = dataloaders['test'].sample_number
    ds_stats = ""
    for k, v in sample_info.items():
        if k != 'total_samples':
            ds_stats += f" {k}  r={v['ratio']} n={v['samples']}\t"
    logger.info(f"save_img: {params.eval_visualize_save}")
    logger.info(f"test dataset: {ds_stats}")
    logger.info(f"total samples: {sample_info['total_samples']}")

    # model
    model = net.fetch_net(params)

    # flop, parameters
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
    logger.info(f"exp_id: {params.exp_id}")

    # Define the model and gpu
    if params.cuda:
        model = net.fetch_net(params).cuda()
        # gpu_num = len(params.gpu_used.split(","))
        # device_ids = range(gpu_num)
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids)
    else:
        model = net.fetch_net(params)

    # Initial status for checkpoint manager
    manager = Manager(
        model=model,
        optimizer=None,
        scheduler=None,
        params=params,
        dataloaders=dataloaders,
        writer=None,
        logger=logger,
    )

    # Reload weights from the saved file
    try:
        manager.load_checkpoints()
    except:
        return

    # Test the model
    logger.info("Starting test")

    # Evaluate
    evaluate(manager)


def gather_metric(Metric, losses, indicator):
    for ele in [losses, indicator]:
        for k, v in ele.items():
            Metric[k].append(v.item())
    return Metric


def statistic_metric(Metric, mode='mean'):
    if mode == 'mean':
        for k, v in Metric.items():
            Metric[k] = np.mean(v)
    return Metric


if __name__ == "__main__":

    # for i in range(5, 10):
    #     run_all_exps(i)
    run_all_exps(exp_id=None)
    # run_all_exps(exp_id=1)
