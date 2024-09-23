from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings
from cube_utils import *
import pandas as pd

warnings.filterwarnings("ignore", ".*does not have many workers.*")

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
        
        # 提取moves
        self.val_data = None
        self.train_data = None

    def setup(self, stage):
        all_data = []
        train_data_list = []
        test_data_list = []
        train_data = pd.read_csv('./data/cube_train.csv')
        # self.val_data = pd.read_csv('./data/cube_test.csv')
        test_data = pd.read_csv('./data/cube_test.csv')
        train_data['init_state'] = train_data.apply(lambda row: ' '.join(map(str, row[['state_{}'.format(i) for i in range(1, 25)]])), axis=1)
        test_data['init_state'] = test_data.apply(lambda row: ' '.join(map(str, row[['state_{}'.format(i) for i in range(1, 25)]])), axis=1)
        for index, row in train_data.iterrows():
            INIT = ' '.join(map(str, row[['state_{}'.format(i) for i in range(1, 25)]]))
            PLAN = row['moves']
            train_data_list.append([INIT, PLAN])
        
        for index, row in test_data.iterrows():

            INIT = ' '.join(map(str, row[['state_{}'.format(i) for i in range(1, 25)]]))
            PLAN = row['moves']
            test_data_list.append([INIT, PLAN])

        self.train_data = PromptDataPipe(train_data_list[:15])
        self.val_data = PromptDataPipe(train_data_list[:15])
        self.test_data = PromptDataPipe(test_data_list[:])

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
