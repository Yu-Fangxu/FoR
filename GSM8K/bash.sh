step_number=4
seed=123
num_data=50
CUDA_VISIBLE_DEVICES=7 python main.py \
    --step $step_number \
    --n_samples 4 \
    --epochs 0 \
    --ll-weight 1.5 \
    --pretrained_model "meta-llama/Meta-Llama-3-8B" \
    --reward_temp_end 2 \
    --pf_temp_start 4 \
    --p_buffer_start 0.25 \
    --p_buffer_end 0.5 \
    --epsilon_start 0.5 \
    --epsilon_end 0.1 \
    --world_model "meta-llama/Meta-Llama-3-8B" \
    --seed $seed \
    --num_data $num_data \
    --mode "test"
