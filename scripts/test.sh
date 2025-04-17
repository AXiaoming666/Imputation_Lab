/home/cxz/anaconda3/envs/myenv/bin/python main.py \
    --seed 42 \
    --dataset_name exchange_rate \
    --missing_rate 0.8 \
    --missing_type F-MNAR \
    --completeness_rate 0.1 \
    --imputation_method knn

cd Time-Series-Library

/home/cxz/anaconda3/envs/TSLib-env/bin/python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../ \
  --data_path imputed_data.csv \
  --model_id Exchange_96_96 \
  --model TimesNet \
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

rm config.npy
rm metrics.npy
rm Time-Series-Library/result_long_term_forecast.txt