step_number=$1
seed=$2
CUDA_VISIBLE_DEVICES=0 python main.py \
    --step $step_number \
    --n_samples 4 \
    --epochs 10 \
    --ll-weight 0 \
    --pretrained_model "meta-llama/Meta-Llama-3-8B" \
    --reward_temp_end 2 \
    --pf_temp_start 4 \
    --p_buffer_start 0.25 \
    --p_buffer_end 0.5 \
    --epsilon_start 0.0 \
    --epsilon_end 0.0 \
    --world_model "meta-llama/Meta-Llama-3-8B" \
    --seed $seed \
    --mode "train" > "./logs/step_${step_number}/DAG_seed_${seed}.txt" 2>&1 &
