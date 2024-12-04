from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings
import sys
import re
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
import json
from tarski.io import PDDLReader

def get_gsm8k_dataset(split):
    with open(f'./data/{split}.jsonl') as f:
        examples = [json.loads(l) for l in f]

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples

class PromptDataModule(LightningDataModule):
    def __init__(
        self,
        args,
        tokenizer,
        train_size=0.2,
        limit_prompts=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        self.tokenizer = tokenizer
        self.args = args
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        # all_data = []
        train = []
        test = []
        test_data = get_gsm8k_dataset("test")
        train_data = get_gsm8k_dataset("train")
        for d in test_data:
            question = d['question']
            answer = d['answer']
            answer = re.search('#### .*?([ $.0-9,\\-]+)', answer)
            answer = '' if answer is None else answer[1].replace(',', '').replace(' ', '').replace('$', '')

            GOAL = answer
            INIT = question
            test.append([INIT, GOAL])
        for d in train_data:
            question = d['question']
            answer = d['answer']
            answer = re.search('#### .*?([ $.0-9,\\-]+)', answer)
            answer = '' if answer is None else answer[1].replace(',', '').replace(' ', '').replace('$', '')

            GOAL = answer
            INIT = question
            train.append([INIT, GOAL])
        self.train_data = PromptDataPipe(train[-self.args.num_data:])
        self.val_data = PromptDataPipe(train[-10:])
        self.test_data = PromptDataPipe(test[:])

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1)

class PromptDataPipe(MapDataPipe):
    def __init__(self, problems) -> None:
        super().__init__()
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index):

        return self.problems[index]
