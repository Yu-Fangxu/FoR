
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings
import sys
from Task import *
# from Executor import Executor
from bw_utils import *

warnings.filterwarnings("ignore", ".*does not have many workers.*")
import yaml
import json

def load_json_data(folder):
    json_files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json')]
    data = {}
    for js in json_files:
        with open(os.path.join(folder, js)) as json_file:
            data[js] = json.load(json_file)
    return data

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

        folder = "/home/fangxu/ARC-Challenge/1D-ARC"
        myjson_train = load_json_data(folder)

        train_data, test_data = [], []
        
        filters = ["1d_move_1p_34.json", "1d_move_1p_39.json", "1d_move_1p_1.json", "1d_move_1p_5.json", "1d_move_1p_30.json", "1d_fill_25.json",
           "1d_fill_40.json", "1d_fill_5.json", "1d_fill_14.json", "1d_fill_43.json", "1d_flip_40.json", "1d_flip_47.json", "1d_flip_412.json", "1d_flip_38.json", "1d_flip_24.json"]
        
        for index, (file_name, grids) in enumerate(myjson_train.items()):
            taskId = file_name.split(".")[0].strip()
            # task = train_tasks["29ec7d0e"]
            # taskId = "29ec7d0e"
            originalT = Task.Task(grids, taskId, submission=False)
            # dataset.append([originalT, grids])

            # Handle files with multiple test inputs
            # Combine test inputs and outputs in alternating manner
            # combined_tests = [{'input': test_inputs[0]['input'], 'output': test_inputs[0]['output']}]
            if file_name in filters:
                train_data.append([str(originalT), str(grids)])
            else:
                test_data.append([str(originalT), str(grids)])
        self.train_data = PromptDataPipe(train_data)
        self.val_data = PromptDataPipe(train_data)
        self.test_data = PromptDataPipe(test_data)

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


import pandas as pd
from transformers import AutoTokenizer


def main():
    # Example arguments for initializing the data module
    class Args:
        step = 1  # Assuming step refers to a particular version of data
        limit_prompts = None  # Limit prompts for testing if needed

    # Initialize tokenizer (replace with the actual tokenizer used in the project)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Instantiate the PromptDataModule with sample arguments
    data_module = PromptDataModule(
        args=Args(),
        tokenizer=tokenizer,
        train_size=0.2,
        limit_prompts=5,  # Limit for testing
    )


if __name__ == "__main__":
    main()


# from pytorch_lightning.utilities.types import EVAL_DATALOADERS
# from torch.utils.data import DataLoader
# from torchdata.datapipes.map import MapDataPipe
# from pytorch_lightning import LightningDataModule
# import warnings
# import sys

# # from Executor import Executor
# from utils import *
# from bw_utils import *

# sys.path.append("gpt-plan-benchmark/gpt_plan_test")
# warnings.filterwarnings("ignore", ".*does not have many workers.*")
# import yaml
# import json
# from tarski.io import PDDLReader


# def get_problem(instance, domain):
#     reader = PDDLReader(raise_on_error=True)
#     reader.parse_domain(domain)
#     return reader.parse_instance(instance)


# class PromptDataModule(LightningDataModule):
#     def __init__(
#         self,
#         args,
#         tokenizer,
#         train_size=0.2,
#         limit_prompts=None,
#     ):
#         super().__init__()
#         self.save_hyperparameters(ignore="tokenizer")
#         self.tokenizer = tokenizer
#         self.args = args
#         self.train_data = None
#         self.val_data = None

#     def setup_arc(self):
#         train_csv = "./data/train_3_final_15.csv"
#         complete_train_data = pd.read_csv(train_csv)
#         train_tasks =  ["1d_move_1p", "1d_fill", "1d_flip"]
#         train_data_by_task = complete_train_data[
#             complete_train_data["task"].isin(train_tasks)
#         ]
#         no_of_examples_per_task = 5
#         train_data_arc = train_data_by_task.groupby("task").head(
#             no_of_examples_per_task
#         )
#         print(train_data_arc,"jalend")
#         test_tasks = train_tasks
#         test_csv = "./data/test_3_final_135.csv"
#         complete_test_data = pd.read_csv(test_csv)
#         test_data_by_task = complete_test_data[
#             complete_test_data["task"].isin(test_tasks)
#         ]
#         test_data_arc = test_data_by_task.groupby("task").head(no_of_examples_per_task)
#         train_data_arc = train_data_arc.reset_index(drop=True)
#         test_data_arc = test_data_arc.reset_index(drop=True)

#         def convert_row_to_dict(row):
#             return {
#                 "input": row["input"],  # Convert string to list
#                 "output": row["output"],  # Convert string to list
#             }
#             return {
#                 "input": eval(row["input"]),  # Convert string to list
#                 "output": eval(row["output"]),  # Convert string to list
#             }

#         # Convert train data
#         train_list = [
#             convert_row_to_dict(row) for idx, row in train_data_arc.iterrows()
#         ]

#         # Convert test data
#         test_list = [convert_row_to_dict(row) for idx, row in test_data_arc.iterrows()]

#         # Create final data dictionary
#         data = {
#             "test": test_list,
#             "train": train_list,
#             "uuid": "some_unique_identifier",  # Replace with actual UUID logic if necessary
#         }
#         # {'input': [[0, 6, 0]], 'output': [[0, 0, 0]]}
#         return data

#     def setup(self, stage):
#         all_data = []
#         # print(all_data[0])
#         all_data_arc = self.setup_arc()
#         all_data = []
#         for sample in all_data_arc["train"]:
#             INIT, GOAL, PLAN = sample["input"], sample["output"], "select a function"
#             all_data.append([INIT, GOAL, PLAN])
#         for sample in all_data_arc["test"]:
#             INIT, GOAL, PLAN = sample["input"], sample["output"], "select a function"
#             all_data.append([INIT, GOAL, PLAN])
#         # print(all_data[0])
#         if self.hparams.limit_prompts is not None:
#             all_data = all_data[: self.hparams.limit_prompts]
#         self.train_data = PromptDataPipe(all_data[:15])
#         self.val_data = PromptDataPipe(all_data[:15])
#         self.test_data = PromptDataPipe(all_data[15:])

#     def train_dataloader(self):
#         return DataLoader(self.train_data, shuffle=True, batch_size=1)

#     def val_dataloader(self):
#         return DataLoader(self.val_data, batch_size=1)

#     def test_dataloader(self):
#         return DataLoader(self.test_data, batch_size=1)


# class PromptDataPipe(MapDataPipe):
#     def __init__(self, problems) -> None:
#         super().__init__()
#         self.problems = problems

#     def __len__(self):
#         return len(self.problems)

#     def __getitem__(self, index):

#         return self.problems[index]


# import pandas as pd
# from transformers import AutoTokenizer


# def main():
#     # Example arguments for initializing the data module
#     class Args:
#         step = 1  # Assuming step refers to a particular version of data
#         limit_prompts = None  # Limit prompts for testing if needed

#     # Initialize tokenizer (replace with the actual tokenizer used in the project)
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#     # Instantiate the PromptDataModule with sample arguments
#     data_module = PromptDataModule(
#         args=Args(),
#         tokenizer=tokenizer,
#         train_size=0.2,
#         limit_prompts=5,  # Limit for testing
#     )

#     # Call setup_arc to load and process ARC data
#     data_sample = data_module.setup_arc()

#     # Inspect the train and test data after setup_arc
#     # print("Train Data ARC (Sample):")
#     # print(data_sample["train"][0])  # Print a few rows of train data


# if __name__ == "__main__":
#     main()