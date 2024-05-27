from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings
import sys
#from Executor import Executor
from util import *
from bw_utils import *
#sys.path.append("gpt-plan-benchmark/gpt_plan_test")
#warnings.filterwarnings("ignore", ".*does not have many workers.*")
import yaml
import json

# init the red block is clear, the blue block is clear, the hand is empty, the blue block is on top of the yellow block, the yellow block is on top of the white block, the white block is on top of the orange block, the red block is on the table and the orange block is on the table
#goal the red block is on top of the blue block and the blue block is on top of the yellow block
# plan ['pick up the red block', 'stack the red block on top of the blue block']

class PromptDataModule(LightningDataModule):
    def __init__(
        self,
        args,
        tokenizer,
        train_size=0.8,
        limit_prompts=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")

        self.base_prompt = json.load(open("data/prompt/next_step_examples.json", 'r'))["input"]

        #self.base_prompt = "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nAfter being given an initial state and an action, give the new state after performing the action.\n"
        self.tokenizer = tokenizer
        self.args = args
        self.train_data = []
        self.val_data = []
        self.test_data = []

    def setup(self, stage):
        train_data = []
        all_data = []
        all_data = json.load(open("data/345hop_random_true.json", 'r'))
        ood = ["3", "31", "131071", "real", "number", "imaginary", "numbers"]
        indis = ["bony", "insect", "cold-blooded", "animal"]
        for key in all_data:
            if key.startswith("example"):
                #print(train_data[key])
                ACTIONS = all_data[key]["test_example"]["question"]
                QUERY = all_data[key]["test_example"]["query"]
                PLAN = all_data[key]["test_example"]["chain_of_thought"]
                GT = all_data[key]["test_example"]["answer"]
            if any([o in ACTIONS for o in ood]):
                if len(self.test_data) < 120:
                    self.val_data.append([ACTIONS, QUERY, PLAN, GT])
                    
            if any([o in ACTIONS for o in indis]):
                #if len(self.train_data) < 50:
                train_data.append([ACTIONS, QUERY, PLAN, GT]) 
                
                if len(self.val_data) < 120:
                    self.val_data.append([ACTIONS, QUERY, PLAN, GT])

        cot_path = "sft/data.txt"

        train_answer = []
        valid_answer = []
        with open(cot_path, "r") as f:
            for line in f.read().splitlines():
                answer = line[line.index("answer='")+len("answer='"):line.index("';")]
                #print(answer)
                if "correct=True" in line:
                    train_answer.append(answer)
                else:
                    valid_answer.append(answer)

        for ans in train_answer:
            actions = ans.split("\\n")
            i = len(actions) - 1
            for train_ex in train_data:
                start_id = 0
                while i >= 0 and train_ex[2][-1-start_id*2] == actions[i]:
                    start_id += 1
                    i -= 1
                if i == -1 and len(self.train_data) < 20:
                    self.train_data.append(train_ex) 
                    break

            #problem = get_problem(d[0], self.domain_pddl)
            #gt_plan_text = d[1]
            #INIT, GOAL, PLAN = instance_to_text_blocksworld(problem, True, gt_plan_text, self.data)
            # all_data.append([INIT, GOAL, PLAN])
            # initial_state = f"I have that, {INIT}."

            # state = self.base_prompt + self.prompts["goal_prefix"] + GOAL.strip() + "\n" + self.prompts["state_prefix"].format(0) + " " + initial_state.strip() + "\n"


      #print("init", INIT)
        #print("goal", GOAL)
        #print("plan", PLAN)
        if self.hparams.limit_prompts is not None:
            all_data = all_data[: self.hparams.limit_prompts]
        #print("all", all_data)
        #num_train = int(len(all_data) * self.hparams.train_size)
        #if self.args.num_train:
           # num_train = self.args.num_train

        #num_train = 1
        #self.train_data = PromptDataPipe(all_data[21:71])
        #self.val_data = PromptDataPipe(all_data[:21])
       # self.test_data = PromptDataPipe(all_data[71:121])
        print("number of training data\n",len(self.train_data))
        print("number of valiation data\n",len(self.val_data))
        print("number of test data\n",len(self.test_data))

        #self.val_data = PromptDataPipe(all_data[1:2])

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