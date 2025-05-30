export CUDA_VISIBLE_DEVICES=0

model_name=TimeMachine
# d state 2
python run.py \
    --is_training 1 \
    --root_path ./dataset/CryptoTx/ \
    --data_path bitcoin_ethereum_transactions.csv \
    --data crypto \
    --model_id CryptoTx_96_12 \
    --model $model_name \
    --features M \
    --target transactions_ethereum \
    --seq_len 96 \
    --pred_len 12 \
    --enc_in 2 \
    --dec_in 2 \
    --c_out 2 \
    --des 'Exp' \
    --d_model 256 \
    --d_state 2\
    --d_ff 256 \
    --freq d \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --patience 5 \
    --gpu 0 \
    --revin 1 \
    --ch_ind 1 \
    --residual 1 \

# python run.py \
#     --is_training 1 \
#     --root_path ./dataset/CryptoTx/ \
#     --data_path bitcoin_ethereum_transactions.csv \
#     --data crypto \
#     --model_id CryptoTx_96_24 \
#     --model $model_name \
#     --features M \
#     --target transactions_bitcoin \
#     --seq_len 96 \
#     --pred_len 24 \
#     --enc_in 2 \
#     --dec_in 2 \
#     --c_out 2 \
#     --des 'Exp' \
#     --d_model 256 \
#     --d_state 2\
#     --d_ff 256 \
#     --freq d \
#     --batch_size 32 \
#     --learning_rate 1e-4 \
#     --patience 5 \
#     --gpu 0

# python run.py \
#     --is_training 1 \
#     --root_path ./dataset/CryptoTx/ \
#     --data_path bitcoin_ethereum_transactions.csv \
#     --data crypto \
#     --model_id CryptoTx_96_48 \
#     --model $model_name \
#     --features M \
#     --target transactions_bitcoin \
#     --seq_len 96 \
#     --pred_len 48 \
#     --enc_in 2 \
#     --dec_in 2 \
#     --c_out 2 \
#     --des 'Exp' \
#     --d_model 256 \
#     --d_state 2\
#     --d_ff 256 \
#     --freq d \
#     --batch_size 32 \
#     --learning_rate 1e-4 \
#     --patience 5 \
#     --gpu 0

# python run.py \
#     --is_training 1 \
#     --root_path ./dataset/CryptoTx/ \
#     --data_path bitcoin_ethereum_transactions.csv \
#     --data crypto \
#     --model_id CryptoTx_96_96 \
#     --model $model_name \
#     --features M \
#     --target transactions_bitcoin \
#     --seq_len 96 \
#     --pred_len 96 \
#     --enc_in 2 \
#     --dec_in 2 \
#     --c_out 2 \
#     --des 'Exp' \
#     --d_model 256 \
#     --d_state 2\
#     --d_ff 256 \
#     --freq d \
#     --batch_size 32 \
#     --learning_rate 1e-4 \
#     --patience 5 \
#     --gpu 0