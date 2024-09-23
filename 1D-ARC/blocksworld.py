
import pytorch_lightning as pl
from replay_buffer import ReplayBuffer
import torch
import warnings
from lightning_module_selection import *
from lightning_data import *


warnings.filterwarnings("ignore")

def arc_planning(model, tokenizer, device, args, model_back=None):

    rbuffer = ReplayBuffer(buffer_size=args.buffer_size)  # initialize a replay buffer

    logZ = torch.nn.Parameter(torch.tensor([args.logZ_init], dtype=torch.float))

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
        enable_checkpointing=False,
    )
    train_probes = data.train_data
    val_probes = data.val_data
    with trainer.init_module():
        task = ARCGFNTask(
            args,
            model,
            logZ,
            tokenizer,
            rbuffer,
            train_data=train_probes,
            val_data=val_probes,
        )

    trainer.fit(model=task, datamodule=data)
    # Ensure the directory exists before writing the file
    trainer.test(ckpt_path="last", dataloaders=data.test_dataloader())
    print("Testing ended")
