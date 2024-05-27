#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import argparse
import random
import sys
sys.path.insert(0, './GPT2ForwardBackward')

from nltk.corpus import stopwords
from opt_util import *
from util import *

stop_words = set(stopwords.words('english'))

os.environ["OPENAI_API_KEY"] = "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"
def options():
    parser = argparse.ArgumentParser()
    ## setting
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pretrained_model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--world_model", type=str, default="meta-llama/Meta-Llama-3-8B")
    #parser.add_argument("--world-model-base", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--straight-through", action="store_true")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--step", type=int, default=2)

    ## experiment
    parser.add_argument("--input-file", type=str,
                        default="./data/lexical/commongen_data/test.multi.constraint.json")
    parser.add_argument("--load-checkpoint-path", type=str, default=None, help="The trained gflownet")
    parser.add_argument("--test-only", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--use-4bit", type=bool, default=True)
    parser.add_argument("--use-lora", type=bool, default=True)
    parser.add_argument("--accumulate_grad_batches", type=int, default=25)
    parser.add_argument("--buffer_size", type=int, default=50)
    parser.add_argument("--use-buffer-prob", type=float, default=0.5)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--ll-weight", type=float, default=0.9)
    parser.add_argument("--num_train", type=int, default=None)
    parser.add_argument("--sum_avg", type=str, default="sum")

    parser.add_argument("--test_csv", type=str, default="cache/test_success_text_llama3_iid_avg.csv")
    parser.add_argument("--valid_csv", type=str, default="cache/test_success_text_llama3_iid_avg.csv")

    #parser.add_argument("--mode", type=str, default='blocksworld',
                        #choices=['blocksworld', 'alfworld'])
    ## model
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=15, help="maximum length of optimized logits.")
    parser.add_argument("--max-length", type=int, default=20, help="maximum length of complete sentence.")
    parser.add_argument("--logZ_init", type=int, default=0, help="initialization of logZ")
    parser.add_argument("--lr-nll-portion", type=float, default=1)
    # temperature
    parser.add_argument("--reward_temp_start", type=float, default=1.0,
                        help="temperature of reward")
    parser.add_argument("--reward_temp_end", type=float, default=2.0,
                        help="temperature of reward")
    parser.add_argument("--epsilon_start", type=float, default=0.4,
                        help="epsilon greedy")
    parser.add_argument("--epsilon_end", type=float, default=0.01,
                        help="epsilon greedy")
    parser.add_argument("--pf_temp_start", type=float, default=2.0,
                        help="temperature of reward")
    parser.add_argument("--pf_temp_end", type=float, default=1.0,
                        help="temperature of reward")
    parser.add_argument("--pf_temp_prob", type=float, default=0.5,
                        help="probability of using tempered policy")
    # lr
    parser.add_argument("--lr", type=float, default=5e-6, help="learning rate in the backward pass.")
    parser.add_argument("--logZ_lr", type=float, default=1e-1, help="learning rate in the backward pass.")
    # iterations
    parser.add_argument("--num-iters", type=int, default=10000)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=3)
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    args = options()
    device = "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if args.seed != -1:
        seed_everything(args.seed)
    # Load pretrained model with lora
    model, tokenizer = load_model(args, device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from blocksworld import blocksworld_planning
    blocksworld_planning(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
