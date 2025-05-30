export CUDA_VISIBLE_DEVICES=0

model_name=DLinear
# d state 2
python run.py \
    --is_training 1 \
    --root_path ./dataset/CryptoTx/ \
    --data_path bitcoin_transactions.csv \
    --data crypto \
    --model_id bitcoin_96_12 \
    --model $model_name \
    --features M \
    --target transactions_bitcoin \
    --seq_len 96 \
    --pred_len 12 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --d_model 256 \
    --d_state 2\
    --d_ff 256 \
    --freq d \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --patience 5 \
    --gpu 0

python run.py \
    --is_training 1 \
    --root_path ./dataset/CryptoTx/ \
    --data_path bitcoin_transactions.csv \
    --data crypto \
    --model_id bitcoin_96_24 \
    --model $model_name \
    --features M \
    --target transactions_bitcoin \
    --seq_len 96 \
    --pred_len 24 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --d_model 256 \
    --d_state 2\
    --d_ff 256 \
    --freq d \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --patience 5 \
    --gpu 0

python run.py \
    --is_training 1 \
    --root_path ./dataset/CryptoTx/ \
    --data_path bitcoin_transactions.csv \
    --data crypto \
    --model_id bitcoin_96_24 \
    --model $model_name \
    --features M \
    --target transactions_bitcoin \
    --seq_len 96 \
    --pred_len 48 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --d_model 256 \
    --d_state 2\
    --d_ff 256 \
    --freq d \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --patience 5 \
    --gpu 0

python run.py \
    --is_training 1 \
    --root_path ./dataset/CryptoTx/ \
    --data_path bitcoin_transactions.csv \
    --data crypto \
    --model_id bitcoin_96_24 \
    --model $model_name \
    --features M \
    --target transactions_bitcoin \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --des 'Exp' \
    --d_model 256 \
    --d_state 2\
    --d_ff 256 \
    --freq d \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --patience 5 \
    --gpu 0