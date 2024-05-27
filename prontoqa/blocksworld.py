import pandas as pd
import os

from nltk.corpus import stopwords

from util import *
from bw_utils import *
stop_words = set(stopwords.words('english'))
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from replay_buffer import ReplayBuffer
import os
import yaml
import sys
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
#from Executor import Executor
from util import *
from pathlib import Path
import torch
from typing import Tuple
import json
import time
import warnings

from lightning_module import *
from lightning_data import *


warnings.filterwarnings("ignore")

def blocksworld_planning(model, tokenizer, device, args, model_back=None):
    # TODO: load_data here
    rbuffer = ReplayBuffer(buffer_size=args.buffer_size) # initialize a replay buffer
    ##### config file #####
    logZ = torch.nn.Parameter(torch.tensor([0], dtype=torch.float))
    data = PromptDataModule(
        args=args,
        tokenizer=tokenizer,
        train_size=0.1,
        limit_prompts=None,
    )

    train_probes = data.train_data
    val_probes = data.val_data
    test_probes = data.test_data


    task = BlocksWorldGFNTask(
        args=args, 
        model=model,
        logZ=logZ, 
        tokenizer=tokenizer,
        replay_buffer=rbuffer,
        train_data=train_probes,
        val_data=val_probes,
        test_data=test_probes)

    checkpoint_callback = ModelCheckpoint(
            # dirpath=checkpoints_path, # <--- specify this on the trainer itself for version control
            filename="ckpt_{epoch:02d}_",
            every_n_epochs=1,
            save_top_k=-1,  # <--- this is important!
        )
    # # Here we adopt deepspeed to accelerate the training
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],     
        accelerator="gpu",
        devices=1,
        precision="16-true",
        #precision=16,
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_checkpointing=True,
        accumulate_grad_batches=10,
    )
    trainer.fit(model=task, datamodule=data) 
    model.save_pretrained("./weight")
    trainer.test(model=task, datamodule=data)
    trainer.test(ckpt_path="./last", dataloaders=data.test_dataloader())

    # for i, d in enumerate(train_data):
    #     # TODO: Train on this data
        
    #     cur_instance = train_data[i]
    #     problem = get_problem(cur_instance[0], domain_pddl)
    #     gt_plan_text = cur_instance[1]
    #     INIT, GOAL, PLAN = instance_to_text_blocksworld(problem, True, gt_plan_text, data)
    #     print(INIT)
    #     print(GOAL)
    #     print("plan:", gt_plan_text)
    #     print(PLAN)
    #     # query = prompts["baseline_action"]
    #     query = "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nAfter being given an initial state and an action, give the new state after performing the action.\n"
    #     # gt_plan = compute_plan(domain_pddl, cur_instance)
    #     # query += fill_template(*instance_to_text_blocksworld(problem, False, data)) + "\n"
    #     # print(query)
    #     break
    #     # It may transform to multi-processes to accelerate
    #     trajs, feasibility = generate_trajectories(initial_state = f'I have that, {INIT}.',
    #                                   goal = f'My goal is to have that {GOAL}.',
    #                                   prompts = prompts,
    #                                   model = model,
    #                                   tokenizer = tokenizer,
    #                                   max_steps = 2,
    #                                   temperature=1,
    #                                   eos_token_id=tokenizer.encode('\n', add_special_tokens=False)[0])
        