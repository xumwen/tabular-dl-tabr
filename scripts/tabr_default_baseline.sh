python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/lightgbm_/house/default-evaluation --function bin.lightgbm_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/catboost_/house/default-evaluation --function bin.catboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/xgboost_/house/default-evaluation --function bin.xgboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/lightgbm_/adult/default-evaluation --function bin.lightgbm_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/catboost_/adult/default-evaluation --function bin.catboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/xgboost_/adult/default-evaluation --function bin.xgboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/lightgbm_/otto/default-evaluation --function bin.lightgbm_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/catboost_/otto/default-evaluation --function bin.catboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/xgboost_/otto/default-evaluation --function bin.xgboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/lightgbm_/higgs-small/default-evaluation --function bin.lightgbm_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/catboost_/higgs-small/default-evaluation --function bin.catboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/xgboost_/higgs-small/default-evaluation --function bin.xgboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/lightgbm_/black-friday/default-evaluation --function bin.lightgbm_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/catboost_/black-friday/default-evaluation --function bin.catboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/xgboost_/black-friday/default-evaluation --function bin.xgboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/lightgbm_/weather-small/default-evaluation --function bin.lightgbm_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/catboost_/weather-small/default-evaluation --function bin.catboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/xgboost_/weather-small/default-evaluation --function bin.xgboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/lightgbm_/covtype/default-evaluation --function bin.lightgbm_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/catboost_/covtype/default-evaluation --function bin.catboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/xgboost_/covtype/default-evaluation --function bin.xgboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/lightgbm_/microsoft/default-evaluation --function bin.lightgbm_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/catboost_/microsoft/default-evaluation --function bin.catboost_.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/xgboost_/microsoft/default-evaluation --function bin.xgboost_.main

CUDA_VISIBLE_DEVICES=0 python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/house/default-evaluation --function bin.tabr.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/house/default-evaluation --function bin.tabr.main
CUDA_VISIBLE_DEVICES=0 python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/adult/default-evaluation --function bin.tabr.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/adult/default-evaluation --function bin.tabr.main
CUDA_VISIBLE_DEVICES=0 python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/otto/default-evaluation --function bin.tabr.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/otto/default-evaluation --function bin.tabr.main
CUDA_VISIBLE_DEVICES=0 python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/higgs-small/default-evaluation --function bin.tabr.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/higgs-small/default-evaluation --function bin.tabr.main
CUDA_VISIBLE_DEVICES=0 python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/black-friday/default-evaluation --function bin.tabr.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/black-friday/default-evaluation --function bin.tabr.main
CUDA_VISIBLE_DEVICES=0 python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/weather-small/default-evaluation --function bin.tabr.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/weather-small/default-evaluation --function bin.tabr.main
CUDA_VISIBLE_DEVICES=0 python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/covtype/default-evaluation --function bin.tabr.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/covtype/default-evaluation --function bin.tabr.main
CUDA_VISIBLE_DEVICES=0 python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/microsoft/default-evaluation --function bin.tabr.main
python bin/go.py /data/Blob_WestJP/xumeng/projects/TabR/exp_1015/tabr/microsoft/default-evaluation --function bin.tabr.main