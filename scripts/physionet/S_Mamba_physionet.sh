export CUDA_VISIBLE_DEVICES=0

model_name=S_Mamba
# d state 2
python -u run_imts.py \
  --is_training 1 \
  --model_id physionet2012_36_12 \
  --model $model_name \
  --data physionet2012 \
  --features M \
  --seq_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --d_ff 256 \
  --itr 1