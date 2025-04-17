dataset_name='exchange_rate'
forecast_model='TimesNet'

for completeness_rate in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
do
    for missing_type in MCAR MAR F-MNAR D-MNAR
    do
        for missing_rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        do
            for imputation_method in forward xgboost mean knn IIM
            do
                if [ -e "./results/${dataset_name}_${missing_rate}_${imputation_method}_${missing_type}_${completeness_rate}_${forecast_model}.json" ]; then
                    echo "Results already exist for ${dataset_name}_${missing_rate}_${imputation_method}_${missing_type}_${completeness_rate}_${forecast_model}.json"
                    continue
                fi

                /home/cxz/anaconda3/envs/myenv/bin/python main.py \
                    --seed 42 \
                    --dataset_name $dataset_name \
                    --missing_rate $missing_rate \
                    --missing_type $missing_type \
                    --completeness_rate $completeness_rate \
                    --imputation_method $imputation_method

                cd Time-Series-Library

                /home/cxz/anaconda3/envs/TSLib-env/bin/python run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path ../ \
                    --data_path imputed_data.csv \
                    --model_id Exchange_96_96 \
                    --model $forecast_model \
                    --data custom \
                    --features M \
                    --seq_len 96 \
                    --label_len 48 \
                    --pred_len 96 \
                    --e_layers 2 \
                    --d_layers 1 \
                    --factor 3 \
                    --enc_in 8 \
                    --dec_in 8 \
                    --c_out 8 \
                    --d_model 64 \
                    --d_ff 64 \
                    --top_k 5 \
                    --des 'Exp' \
                    --itr 1

                rm -rf results
                rm -rf checkpoints
                rm -rf test_results

                cd ..

                /home/cxz/anaconda3/envs/myenv/bin/python catch_results.py TimesNet

                rm imputed_data.csv
                rm config.npy
                rm metrics.npy
                rm Time-Series-Library/result_long_term_forecast.txt
            done
        done
    done
done