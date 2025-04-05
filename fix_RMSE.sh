for imputation_method in forward mean knn xgboost IIM
do
    for missing_type in MCAR MAR F-MNAR D-MNAR
    do
        for missing_rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        do
            for completeness_rate in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
            do
                /home/cxz/anaconda3/envs/myenv/bin/python fix_RMSE.py \
                    --seed 42 \
                    --dataset_name exchange_rate \
                    --missing_rate $missing_rate \
                    --missing_type $missing_type \
                    --completeness_rate $completeness_rate \
                    --imputation_method $imputation_method
            done
        done
    done
done