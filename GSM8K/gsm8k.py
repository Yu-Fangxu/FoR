import pandas as pd
import os

# from nltk.corpus import stopwords

from util import *
from bw_utils import *
# stop_words = set(stopwords.words('english'))
import pytorch_lightning as pl
from replay_buffer import ReplayBuffer
import os
import yaml
import sys
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
# from utils import *
import torch
from llama import *
import warnings
from peft import PeftModel, PeftConfig

from lightning_module_selection import *
from lightning_data import *

warnings.filterwarnings("ignore")

def load_model(pretrained_model, lora_model_path):
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=pretrained_model,
                                                trust_remote_code=True,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)

    # Load the LoRA weights into the base model
    model = PeftModel.from_pretrained(model, lora_model_path)

    return model

def gsm8k_planning(model, tokenizer, device, args, model_back=None):
    # TODO: load_data here
    rbuffer = ReplayBuffer(buffer_size=args.buffer_size) # initialize a replay buffer
    ##### config file #####
    logZ = torch.nn.Parameter(torch.tensor([args.logZ_init], dtype=torch.float))
    # train_data = json.load(open(f"/home/fangxu/GFlowPlan/data/blocksworld/step_{args.step}.json", 'r'))
    data = PromptDataModule(
        args=args,
        tokenizer=tokenizer,
        train_size=0.4,
        limit_prompts=None,
    )

    trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
        precision="16-true",
        max_epochs=args.epochs,
        check_val_every_n_epoch=10,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        profiler="simple",
        enable_checkpointing=False
        # strategy="deepspeed_stage_2"
        )
    train_probes = data.train_data
    val_probes = data.val_data
    if args.mode == "train":
        with trainer.init_module():
            task = GSM8KGFNTask(
                args, 
                model,
                logZ, 
                tokenizer,
                rbuffer,
                train_data=train_probes,
                val_data=val_probes)

        # # Here we adopt deepspeed to accelerate the training
        trainer.fit(model=task, datamodule=data)
        model.save_pretrained("/home/fangxu/GFlowPlan/ckpt/6-step")
        print("PEFT saved...")
        # trainer.test(model=task, datamodule=data)
        trainer.test(ckpt_path="last", dataloaders=data.test_dataloader())
        transition_path = f"/home/fangxu/GFlowPlan/transitions/{args.step}/transition.pkl"
        with open(transition_path, 'wb') as f:
            pickle.dump(task.transitions, f)
    else:
        model = load_model("meta-llama/Meta-Llama-3-8B", "./ckpt/6-step")

        with trainer.init_module():
            task = GSM8KGFNTask(
                args, 
                model,
                logZ, 
                tokenizer,
                rbuffer,
                train_data=train_probes,
                val_data=val_probes)
        trainer.test(task, dataloaders=data)  
