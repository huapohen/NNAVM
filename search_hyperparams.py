"""Peform hyperparemeters search"""
# -*- coding: utf-8 -*-
import os
import sys
import time
import ipdb
import shutil
import argparse
import itertools
import collections
from easydict import EasyDict
from subprocess import check_call

from common import utils
# from window_segment import dispatcher_v0, tmux_v0
from window_segment import dispatcher, tmux


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_root_dir",
    default="experiments",
    help="Directory containing params.json",
)


def launch_training_job(
    exp_root_dir,
    exp_name,
    session_name,
    param_pool_dict,
    device_used,
    params,
    start_id=0,
):
    # Partition tmux windows automatically
    tmux_ops = tmux.TmuxOps()
    # Combining hyper-parameters and experiment ID automatically
    task_manager = dispatcher.Enumerate_params_dict(
        task_thread=0, if_single_id_task=True, **param_pool_dict
    )

    num_jobs = len([v for v in itertools.product(*param_pool_dict.values())])
    num_device = len(device_used)
    exp_cmds = []
    
    assert num_jobs == num_device

    for job_id in range(num_jobs):
        device_id = device_used[job_id % num_device]
        hyper_params = task_manager.get_thread(ind=job_id)[0]
        exp_id = job_id + start_id
        job_name = 'exp_{}'.format(exp_id)
        # ipdb.set_trace()
        for k in hyper_params.keys():
            params.dict[k] = hyper_params[k]

        params.dict["model_dir"] = os.path.join(exp_root_dir, exp_name, job_name)
        model_dir = params.dict["model_dir"]

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Write parameters in json file
        json_path = os.path.join(model_dir, "params.json")
        params.save(json_path)
        # ipdb.set_trace()

        # Launch training with this config
        tb_path = os.path.join(exp_root_dir, exp_name, "tf_log", job_name)
        cmd = (
            'python train.py '
            f'--gpu_used {device_id} '
            f'--exp_root_dir {exp_root_dir} '
            f'--model_dir {model_dir} '
            f'--exp_name {exp_name} '
            f'--exp_id {exp_id} '
            f'--tb_path {tb_path}'
        )

        exp_cmds.append(cmd)

    if_serial = num_jobs > num_device
    if if_serial:
        print("run task serially! ")
    else:
        print("run task parallelly! ")

    tmux_ops.run_task(
        exp_cmds, task_name=exp_name, session_name=session_name, if_serial=if_serial
    )

    if True:
        print(
            '\n terminal: \n'
            f'\t input `tmux ls` to find all running tasks \n'
            f'\t input `tmux attach -t {start_id}` to enter the tmux window \n'
            f'\t input `CTRL+C` to interupt the task \n'
            f'\t input `CTRL+D` to end the current subwindow \n'
            f'\t turn off the window to keep the `tmux` window running in the background \n'
        )
        time.sleep(10)
        check_call(f'tmux ls', shell=True)
        time.sleep(5)
        check_call(f'tmux attach -t {start_id}', shell=True)


def experiment():
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    exp_root_dir = args.exp_root_dir
    json_path = os.path.join(exp_root_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    params = utils.Params(json_path)

    exp_name = "eeavm"
    exp_start_id = 2
    session_name = f"{exp_start_id}"
    param_pool_dict = collections.OrderedDict()
    device_used = collections.OrderedDict()
    n_gpus = 1
    if "linux" in sys.platform:
        param_pool_dict["data_dir"] = ["/home/data/lwb/data/dybev"]
    else:  # windows
        param_pool_dict["data_dir"] = [""]
    param_pool_dict["train_batch_size"] = [16 * n_gpus]
    param_pool_dict["eval_batch_size"] = [128 * n_gpus]
    param_pool_dict["num_workers"] = [8 * n_gpus]
    device_used = ['6']
    param_pool_dict["camera_list"] = [['front']]
    # param_pool_dict["camera_list"] = [['front', "back", "left", "right"]]
    param_pool_dict["model_type"] = ["light"]
    param_pool_dict["channel_ratio"] = [0.125]
    param_pool_dict["learning_rate"] = [0.001]
    param_pool_dict["dataset_type"] = ["train"]
    param_pool_dict["train_data_ratio"] = [[["v2", 1]]]
    param_pool_dict["val_data_ratio"] = [[["v2", 0.1]]]
    param_pool_dict["eval_freq"] = [10000]
    param_pool_dict["num_epochs"] = [36]

    # '0', '1', '2', '3', '4', '5', '6', '7'
    # device_used = ['6']
    # device_used = ['0', '0']
    # device_used = ['0_1_2_3'] # for one experiment
    # device_used = ['0', '1', '2', '3'] # for four experiments

    for id in range(exp_start_id, exp_start_id + len(device_used)):
        try:  # cover exp
            bp = os.path.join(exp_root_dir, exp_name)
            shutil.rmtree(os.path.join(bp, f"exp_{id}"))
            path = os.path.join(bp, "tf_log", f"{exp_name}_exp_{id}")
            shutil.rmtree(path)
        except:
            pass

    launch_training_job(
        exp_root_dir,
        exp_name,
        session_name,
        param_pool_dict,
        device_used,
        params,
        exp_start_id,
    )


if __name__ == "__main__":
    experiment()
