from util import *
from cube_utils import *
import pytorch_lightning as pl
from replay_buffer import ReplayBuffer
import os
from util import *
import torch
import warnings

from lightning_module_selection import *
from lightning_data import *


warnings.filterwarnings("ignore")

def blocksworld_planning(model, tokenizer, device, args, model_back=None):
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
            task = CubeGFNTask(
                args, 
                model,
                logZ, 
                tokenizer,
                rbuffer,
                train_data=train_probes,
                val_data=val_probes)

        # # Here we adopt deepspeed to accelerate the training
        trainer.fit(model=task, datamodule=data)
        print("PEFT saved...")
        trainer.test(ckpt_path="last", dataloaders=data.test_dataloader())
    else:
        model = AutoModelForCausalLM.from_pretrained("/home/fangxu/GFlowPlan/ckpt/6-step")
        with trainer.init_module():
            task = BlocksWorldGFNTask(
                args, 
                model,
                logZ, 
                tokenizer,
                rbuffer,
                train_data=train_probes,
                val_data=val_probes)
        trainer.test(task, dataloaders=data.test_dataloader())  
