"""Peform hyperparemeters search"""

import argparse
import collections
import itertools
import os
import sys
import shutil

from common import utils
from experiment_dispatcher import dispatcher, tmux


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument(
    "--parent_dir", default="experiments", help="Directory containing params.json"
)
parser.add_argument("--id", default=1, type=int, help="Experiment id")


def launch_training_job(
    exp_dir, exp_name, session_name, param_pool_dict, device_used, params, start_id=0
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

    for job_id in range(num_jobs):

        device_id = device_used[job_id % num_device]
        hyper_params = task_manager.get_thread(ind=job_id)[0]

        job_name = "exp_{}".format(job_id + start_id)
        for k in hyper_params.keys():
            params.dict[k] = hyper_params[k]

        params.dict["model_dir"] = os.path.join(exp_dir, exp_name, job_name)
        model_dir = params.dict["model_dir"]

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Write parameters in json file
        json_path = os.path.join(model_dir, "params.json")
        params.save(json_path)

        # Launch training with this config
        cmd = (
            "python train.py "
            "--gpu_used {} "
            "--model_dir {} "
            "--exp_name {} "
            "--tb_path {}".format(
                device_id,
                model_dir,
                exp_name + "_" + job_name,
                os.path.join(exp_dir, exp_name, "tf_log/"),
            )
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


def experiment():
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    params = utils.Params(json_path)

    if args.id == 1:
        exp_name = "eeavm"
        start_id = 1
        session_name = f"{start_id}"
        param_pool_dict = collections.OrderedDict()
        device_used = collections.OrderedDict()
        n_gpus = 1
        if "linux" in sys.platform:
            param_pool_dict["data_dir"] = [""]
        else:  # windows
            param_pool_dict["data_dir"] = [""]
        param_pool_dict["train_batch_size"] = [16 * n_gpus]
        # param_pool_dict["train_batch_size"] = [8 * n_gpus]
        param_pool_dict["eval_batch_size"] = [128 * n_gpus]
        param_pool_dict["num_workers"] = [8 * n_gpus]
        param_pool_dict["in_channel"] = [3]

        device_used = ['7']
        # device_used = ['1']
        param_pool_dict["expansion"] = [2]
        # param_pool_dict['expansion'] = [0.125]
        param_pool_dict["learning_rate"] = [0.01]
        # param_pool_dict["learning_rate"] = [0.1]
        param_pool_dict["model_type"] = ["yolo"]
        param_pool_dict["dataset_type"] = ["train"]
        param_pool_dict["train_data_ratio"] = [[["PTS", 1]]]
        param_pool_dict["val_data_ratio"] = [[["PTS", 0.1]]]
        param_pool_dict["save_every_epoch"] = [False]
        param_pool_dict["eval_freq"] = [10000]
        param_pool_dict["num_epochs"] = [36]
        # '0', '1', '2', '3', '4', '5', '6', '7'
        # device_used = ['6']
        # device_used = ['0', '0']
        # device_used = ['0_1_2_3'] # for one experiment
        # device_used = ['0', '1', '2', '3'] # for four experiments

        for id in range(start_id, start_id + len(device_used)):
            try:  # cover exp
                bp = os.path.join("experiments", exp_name)
                shutil.rmtree(os.path.join(bp, f"exp_{id}"))
                path = os.path.join(bp, "tf_log", f"{exp_name}_exp_{id}")
                shutil.rmtree(path)
            except:
                pass
    else:
        raise NotImplementedError

    launch_training_job(
        args.parent_dir,
        exp_name,
        session_name,
        param_pool_dict,
        device_used,
        params,
        start_id,
    )


if __name__ == "__main__":
    experiment()
