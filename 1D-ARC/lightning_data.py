
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings
from Task import *
import os
# from Executor import Executor
warnings.filterwarnings("ignore", ".*does not have many workers.*")
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
           "1d_fill_40.json", "1d_fill_5.json", "1d_fill_14.json", "1d_fill_43.json", "1d_flip_40.json", "1d_flip_47.json", "1d_flip_12.json", "1d_flip_38.json", "1d_flip_24.json"]
        
        for index, (file_name, grids) in enumerate(myjson_train.items()):
            taskId = file_name.split(".")[0].strip()
            # originalT = Task(grids, taskId, submission=False)
            if file_name in filters:
                train_data.append([grids, taskId])
            else:
                test_data.append([grids, taskId])
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