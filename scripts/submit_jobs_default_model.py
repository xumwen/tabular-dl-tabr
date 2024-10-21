import os
import time
import shlex
import subprocess
import argparse
import numpy as np
from itertools import product
from datetime import timedelta

DATASETS_DEFAULT8 = [
    'house',
    'adult',
    'otto',
    'higgs-small',
    'black-friday',
    'weather-small',
    'covtype',
    'microsoft',
]

DATASETS_TREE36 = [
    "classif-cat-large-0-covertype",
    "classif-cat-large-0-road-safety",
    "classif-cat-medium-0-KDDCup09_upselling",
    "classif-cat-medium-0-compass",
    "classif-cat-medium-0-electricity",
    "classif-cat-medium-0-rl",
    "classif-num-large-0-Higgs",
    "classif-num-large-0-MiniBooNE",
    "classif-num-large-0-jannis",
    "classif-num-medium-0-credit",
    "classif-num-medium-0-kdd_ipums_la_97-small",
    "classif-num-medium-0-phoneme",
    "regression-cat-large-0-SGEMM_GPU_kernel_performance",
    "regression-cat-large-0-black_friday",
    "regression-cat-large-0-nyc-taxi-green-dec-2016",
    "regression-cat-large-0-particulate-matter-ukair-2017",
    "regression-cat-medium-0-Brazilian_houses",
    "regression-cat-medium-0-Mercedes_Benz_Greener_Manufacturing",
    "regression-cat-medium-0-OnlineNewsPopularity",
    "regression-cat-medium-0-analcatdata_supreme",
    "regression-cat-medium-0-house_sales",
    "regression-cat-medium-0-visualizing_soil",
    "regression-cat-medium-0-yprop_4_1",
    "regression-num-large-0-year",
    "regression-num-medium-0-Ailerons",
    "regression-num-medium-0-MiamiHousing2016",
    "regression-num-medium-0-cpu_act",
    "regression-num-medium-0-elevators",
    "regression-num-medium-0-fifa",
    "regression-num-medium-0-house_16H",
    "regression-num-medium-0-houses",
    "regression-num-medium-0-isolet",
    "regression-num-medium-0-medical_charges",
    "regression-num-medium-0-pol",
    "regression-num-medium-0-sulfur",
    "regression-num-medium-0-superconduct"
]

MODELS_WITH_DEFAULT_CONFIG = ['catboost_', 'lightgbm_', 'xgboost_', 'tabr']

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
    parser.add_argument('--config_dir', type=str, default='/data/Blob_WestJP/xumeng/projects/TabR/exp_default_1016')

    args = parser.parse_args()
    cuda_dict = dict([(str(i), None) for i in args.cuda_ids.split('-')])
    config_dir = args.config_dir

    cmds = []
    
    for dataset in DATASETS_TREE36:
        for model in MODELS_WITH_DEFAULT_CONFIG:
            target_config_path = os.path.join(config_dir, model, dataset, 'default-evaluation')
            cmds.append(f"python bin/go.py {target_config_path} --function bin.{model}.main {target_config_path}")
    start = time.time()
    run(cmds, cuda_dict)
    end = time.time()
    print(f'Total elapsed time: {str(timedelta(seconds=end-start))}')