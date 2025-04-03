/home/cxz/anaconda3/envs/myenv/bin/python fix_RMSE.py \
    --seed 42 \
    --dataset_name exchange_rate \
    --missing_rate 0.1 \
    --missing_type D-MNAR \
    --completeness_rate 0.1 \
    --imputation_method forward