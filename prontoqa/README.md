## Task Introduction
PrOntoQA is a logical reasoning task. A policy LLM needs to give correct fact that connects the previous state in the reasoning chain. The sentences in the examples are syntactically simple and amenable to semantic parsing.

## Command

To run the Llama-3 model, you will need to request access at Huggingface and set up account login:
```
huggingface-cli login --token [Your Hugging face token]
```

```
CUDA_VISIBLE_DEVICES=<GPU-index> nohup python -u main.py --sum_avg "sum" --test_csv "cache/test_success.csv" --valid_csv "cache/validation_success.csv" --check_val_every_n_epoch 3 --n_samples 4 --epochs 40 --ll-weight $ll_weight --use-buffer 0.5 --logZ_init 5 --pretrained_model "meta-llama/Meta-Llama-3-8B-Instruct" > "./logs/llama3_logs.txt" 2>&1 &
```
