import os
import sys
import thop
import time
import shutil
import warnings
import argparse
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

import dataset.data_loader as data_loader
import model.net as net

from common import utils
from common.manager import Manager
from evaluate import evaluate
from loss.losses import *

warnings.filterwarnings("ignore")


def p(*arg, **kwargs):
    return print(*arg, **kwargs)


torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir", default="default", help="Directory containing params.json"
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
    default="default",
    help="select the gpu for train or evaluation.",
)
parser.add_argument(
    "-exp", "--exp_name", type=str, default="default", help="experiment name."
)
parser.add_argument(
    "-tb",
    "--tb_path",
    type=str,
    default="default",
    help="the path to save the tensorboardx log.",
)


def train(manager):

    manager.reset_loss_status()

    torch.cuda.empty_cache()
    manager.model.train()

    iter_max = len(manager.dataloaders["train"])
    if iter_max == 0:
        print("\t\t\t\t empty input")
        sys.exit()
    # four vertex coordinates
    bev_coords = get_coords(manager.params)
    bev_ori_fblr = get_bev_ori(manager.params.train_batch_size)
    bev_coords = bev_coords.cuda()

    with tqdm(total=iter_max) as t:
        for i, data_batch in enumerate(manager.dataloaders["train"]):
            # data_batch = utils.tensor_gpu(data_batch, params_gpu=manager.params.cuda)
            images = data_batch["image"].cuda()
            labels = [lab.cuda() for lab in data_batch["label"]]

            delta = manager.model(images)

            losses = compute_losses(
                manager.params,
                delta,
                labels,
                bev_coords,
                bev_ori_fblr,
            )

            manager.optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                losses["total"].backward()

            manager.update_loss_status(loss=losses, split="train")

            manager.optimizer.step()

            manager.update_step()

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

        # evaluate the model
        # eval处理loss，batch内部取的1
        if epoch != 0 and epoch % manager.params.eval_freq == 0:
            evaluate(manager)

        # Save latest model, or best model weights accroding to the params.major_metric
        manager.check_best_save_last_checkpoints(latest_freq=1)

        # for finetune
        if epoch == 1600:
            sys.exit()

        # compute number of batches in one epoch (one full pass over the training set)
        train(manager)

    with open(os.path.join(manager.params.model_dir, "finish_flag.txt"), "w") as f:
        f.write("exp finish!")


if __name__ == "__main__":
    """
    python train.py --gpu_used 0 --model_dir experiments/rs/exp_1 --exp_name rs_exp_1 --tb_path experiments/rs/tf_log/
    """
    # Load the parameters from json file
    args = parser.parse_args()

    # is_continue_training = True
    is_continue_training = False
    if is_continue_training:
        exp_id = 1
        args.gpu_used = "0"
        exp_name = "eeavm"
        args.model_dir = os.path.join("experiments", exp_name, f"exp_{exp_id}")
        args.tb_path = os.path.join("experiments", exp_name, "tf_log")
        # args.restore_file = 'yolors_model_latest.pth'
        # args.only_weights = False
        # args.restore_file = '1.pth'
        # args.only_weights = True
        # args.learning_rate = 5e-7
        # args.learning_rate_new = True
        # args.num_epochs = 610

    # 直接启动 train.py，从零开始训
    train_without_search_hyperparams = True
    # train_without_search_hyperparams = False
    cacl_flop = 0

    if train_without_search_hyperparams == True:
        exp_id = 1
        args.gpu_used = "1"
        exp_name = "eeavm"
        args.exp_name = ""
        if "linux" in sys.platform:
            args.data_dir = "/home/data/lwb/data/dybev"
        else:  # windows
            args.data_dir = [""]
        args.model_dir = f"experiments/{exp_name}/exp_{exp_id}"
        args.tb_path = f"experiments/{exp_name}/tf_log/"
        args.tb_path += f"{exp_name}_exp_{exp_id}"
        try:
            shutil.rmtree(args.model_dir)
            shutil.rmtree(args.tb_path)
        except:
            pass
        # args.model_type = "yolo"
        args.model_type = "light"
        args.dataset_type = "train"
        args.learning_rate = 0.001
        args.train_batch_size = 8
        # args.num_workers = 12
        args.num_workers = 4
        args.num_epochs = 12
        args.eval_freq = 10000
        is_calc_flops = 1

    # 有新的字段，放在了默认的json里，需要读出来，以兼容之前的实验
    if "read existing default params.json":
        default_json_path = os.path.join("experiments", "params.json")
        params = utils.Params(default_json_path)

        # 读之前的实验 json
        json_path = os.path.join(args.model_dir, "params.json")
        if os.path.exists(json_path):
            model_params_dict = utils.Params(json_path).dict
            # 实验json覆盖默认json
            params.update(model_params_dict)
        else:
            os.makedirs(args.model_dir, exist_ok=True)
            params.save(os.path.join(args.model_dir, "params.json"))

    # 通过 search_hyperparams.py 启用 train.py
    if train_without_search_hyperparams == False and is_continue_training == False:
        json_path = os.path.join(args.model_dir, "params.json")
        assert os.path.isfile(
            json_path
        ), "No json configuration file found at {}".format(json_path)
        params = utils.Params(json_path)

    # Update args into params
    # 把这里args手动定义的参数更新到manager.params里
    params.update(vars(args))
    if train_without_search_hyperparams:
        params.save(os.path.join(args.model_dir, "params.json"))

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

    # model
    model = net.fetch_net(params)

    # flop, parameters
    if params.is_calc_flops:
        input = torch.randn(1, params.input_channels, params.img_h, params.img_w)
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
        sys.exit()

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

    # Create the input data pipeline
    logger.info("Loading the train datasets from {}".format(params.data_dir))
    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # initial status for checkpoint manager
    assert params.metric_mode in ["ascend", "descend"]
    manager = Manager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        params=params,
        dataloaders=dataloaders,
        writer=None,
        logger=logger,
        exp_name=args.exp_name,
        tb_path=args.tb_path,
    )

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(manager)
