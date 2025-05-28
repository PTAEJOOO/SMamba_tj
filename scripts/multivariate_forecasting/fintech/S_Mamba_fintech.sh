export CUDA_VISIBLE_DEVICES=0

model_name=S_Mamba
# d state 2
python run.py \
    --is_training 1 \
    --root_path ./dataset/CryptoTx/ \
    --data_path bitcoin_ethereum_transactions.csv \
    --data crypto \
    --model_id CryptoTx_96_96 \
    --model S_Mamba \
    --features M \
    --target transactions_bitcoin \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 2 \
    --dec_in 2 \
    --c_out 2 \
    --freq d \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --gpu 0

# python -u run.py \
#   --is_training 1 \
#   --root_path ./datasets/CryptoTx/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --freq d \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_state 2\
#   --d_ff 256 \
#   --itr 1 \
#   --learning_rate 0.00007

