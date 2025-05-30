export CUDA_VISIBLE_DEVICES=0

model_name="S_Mamba TimeMachine DLinear RLinear LSTM iTransformer Autoformer" 
pred_lens="12 24 48 96 192 336 720"

# d state 2
for mo in $model_name;do
    for pl in $pred_lens;do
        python run.py \
            --is_training 1 \
            --root_path ./dataset/CryptoTx/ \
            --data_path bitcoin_ethereum_transactions.csv \
            --data crypto \
            --model_id CryptoTx_96_12 \
            --model $mo \
            --features M \
            --target transactions_ethereum \
            --seq_len 96 --pred_len $pl \
            --enc_in 2 --dec_in 2 --c_out 2 \
            --des 'Exp' \
            --d_model 256 --d_state 2 --d_ff 256 \
            --freq d \
            --batch_size 32 --learning_rate 1e-4 \
            --patience 10 --gpu 0 \
            --revin 1 \
            --ch_ind 1 \
            --residual 1
    done
done
