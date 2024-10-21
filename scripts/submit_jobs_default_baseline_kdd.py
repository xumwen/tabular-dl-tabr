import os
import time
import shlex
import subprocess
import argparse
import numpy as np
from itertools import product
from datetime import timedelta

def load_dataset_config(config_path):
    datasets = []
    with open(config_path, "r") as f:
        for line in f:
            dataset_name = line.strip()
            datasets.append(dataset_name)
    return datasets

# Load benchmark datasets
kdd_config_path = "/data/Blob_EastUS/v-zhenxu2/projects/TabFM/configs/kdd16"
tabr_config_path = "/data/Blob_EastUS/v-zhenxu2/projects/TabFM/configs/default8"
tree_config_path = "/data/Blob_EastUS/v-zhenxu2/projects/TabFM/configs/tree36"
kdd_datasets = load_dataset_config(kdd_config_path)
tabr_datasets = load_dataset_config(tabr_config_path)
tree_datasets = load_dataset_config(tree_config_path)

MODELS = ['ffn', 'ft_transformer', 'saint', 'lightgbm_', 'catboost_', 'xgboost_', 'tabr']

def run(cmds, cuda_id):
    _cur = 0

    def recycle_devices():
        for cid in cuda_id:
            if cuda_id[cid] is not None:
                proc = cuda_id[cid]
                if proc.poll() is not None:
                    cuda_id[cid] = None

    def available_device_id():
        for cid in cuda_id:
            if cuda_id[cid] is None:
                return cid

    def submit(cmd, cid):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = cid

        args = shlex.split(cmd)
        exp_dir = args[-1]
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)
        log = open('{}/log.txt'.format(exp_dir), 'w')
        print(time.asctime(), ' '.join(args))

        proc = subprocess.Popen(args[:-1], env=env, stdout=log, stderr=log)

        cuda_id[cid] = proc

    while _cur < len(cmds):
        recycle_devices()
        cid = available_device_id()

        if cid is not None:
            print(f'CUDA {cid} available for job ({_cur+1} of {len(cmds)})')
            submit(cmds[_cur], cid)
            _cur += 1

        time.sleep(1)
    
    while any([v is not None for v in cuda_id.values()]):
        recycle_devices()
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Tabular Data')
    parser.add_argument('--cuda_ids', type=str, default='0')
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--config_dir', type=str, default='/data/Blob_WestJP/xumeng/projects/TabR/exp_default_1016')

    args = parser.parse_args()
    cuda_dict = dict([(str(i), None) for i in args.cuda_ids.split('-')])
    worker_id = int(args.worker_id)
    num_workers = int(args.num_workers)
    config_dir = args.config_dir

    cmds = []
    
    job_id = 0
    for dataset in kdd_datasets:
        for model in MODELS:
            if job_id % num_workers == worker_id:
                target_config_path = os.path.join(config_dir, model, dataset, 'default-evaluation')
                cmds.append(f"python bin/go.py {target_config_path} --function bin.{model}.main {target_config_path}")
            job_id += 1
    start = time.time()
    run(cmds, cuda_dict)
    end = time.time()
    print(f'Total elapsed time: {str(timedelta(seconds=end-start))}')