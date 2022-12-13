# -*- coding: utf-8 -*-
import argparse
import logging
import platform
import os
import cv2
import sys
import json
import thop
import shutil
import datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable

import dataset.data_loader as data_loader
import model.net as net

from common import utils
from common.manager import Manager
from loss.losses import *
from dataset.postprocess import *

result_all_exps = {}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir",
    default="experiments/rs/exp_3",
    help="Directory containing params.json",
)
parser.add_argument(
    "--restore_file",
    # default='yolors_test_model_best.pth',
    default="yolors_model_latest.pth",
    help="name of the file in --model_dir containing weights to load",
)
parser.add_argument(
    "-gu",
    "--gpu_used",
    type=str,
    default="0",
    help="select the gpu for train or evaluation.",
)


def evaluate(manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    print("==============eval begin==============")

    # loss status and eval status initial
    manager.reset_loss_status()
    manager.reset_metric_status("test")
    manager.model.eval()

    with torch.no_grad():
        # compute metrics over the dataset

        total_loss = []

        iter_max = len(manager.dataloaders["test"])
        ds_stats = ""
        for ds in manager.params.test_data_save:
            ds_stats += ds[0] + " "
        print(f"save_img: {manager.params.test_data_save}  " + f"dataset: {ds_stats}")
        manager.logger.info(f"dataset: {ds_stats} {manager.params.eval_dataset}")

        with tqdm(total=iter_max) as t:
            for idx, data_batch in enumerate(manager.dataloaders["test"]):
                images = data_batch["image"].cuda()
                labels = data_batch["label"].cuda()
                names = data_batch["name"]

                output = manager.model(images)
                loss = {}
                loss["total"] = F.mse_loss(output, labels)
                total_loss.append(loss["total"].item())

                if manager.params.test_data_save:
                    visulize_results(
                        manager.params, images, names, output, idx,
                    )

                t.update()

        total_loss = np.mean(total_loss) if len(total_loss) else 0.0

        Metric = {"total_loss": total_loss}

        # result_all_exps[manager.params.exp_id] = Metric

        manager.update_metric_status(
            metrics=Metric, split="test", batch_size=manager.params.eval_batch_size,
        )

        # update data to logger
        manager.logger.info(
            "Loss/valid epoch_{} {}: "
            "total_loss: {:.6f} | ".format("test", manager.epoch_val, total_loss)
        )

        # For each epoch, print the metric
        manager.print_metrics("test", title="test", color="green")

        # manager.epoch_val += 1
        manager.model.train()


def run_all_exps(eid, val_data_save):
    """
    Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()

    if "linux" in sys.platform:
        args.data_dir = ""
    else:  # windows
        args.data_dir = [""]
    base_path = os.path.join(os.getcwd(), "experiments", "eeavm")
    args.exp_id = eid
    args.exp_id = 1
    args.model_dir = os.path.join(base_path, f"exp_{args.exp_id}")
    # args.restore_file = 'eeavm_test_model_best.pth'
    args.restore_file = "eeavm_model_latest.pth"
    args.restore_file = "82.pth"
    args.eval_batch_size = 8
    args.gpu_used = "2"
    args.num_workers_eval = 1
    args.dataset_type = "test"
    args.eval_dataset = "PTS"
    args.test_data_ratio = [["PTS", 0.01]]
    args.val_data_save = val_data_save

    # merge params
    default_json_path = os.path.join("experiments", "params.json")
    params = utils.Params(default_json_path)
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    model_params_dict = utils.Params(json_path).dict
    params.update(model_params_dict)

    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Use GPU if available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_used
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, "evaluate.log"))

    # Create the input data pipeline
    logging.info(f"Creating the dataset {args.eval_dataset}")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # # flop, parameters
    input = torch.randn(1, params.input_channels, params.img_h, params.img_w)
    model = net.get_model(params)
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

    # Define the model and optimizer
    if params.cuda:
        model = net.get_model(params).cuda()
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids)
    else:
        model = net.get_model(params)

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


if __name__ == "__main__":

    # val_data_save = False
    val_data_save = True
    for i in range(0, 1):
        run_all_exps(0, val_data_save)
