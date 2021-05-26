import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
import pandas as pd
import dataclasses
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Any
from sklearn.model_selection import KFold
import random
import os
import numpy as np
from pytorch_lightning import Trainer
import mlflow.pytorch

class CommonLitDataset(Dataset):
    def __init__(self, df, tokenizer, transforms=None):
        self.df = df.reset_index()
        self.augmentations = transforms
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        text = row["excerpt"]

        text = self.tokenizer(text, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        input_ids = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]

        target = torch.tensor(row["target"], dtype=torch.float)
        return input_ids, attention_mask, target

@dataclasses.dataclass
class Config:
    experiment_name: str
    debug: bool = False
    fold: int = 0

    nlp_model_name: str = "bert-base-uncased"
    linear_dim: int = 128
    dropout: float = 0
    dropout_stack: float = 0.2
    batch_size: int = 16

    lr_bert: float = 1e-5
    lr_fc: float = 1e-4
    scheduler_params = {"num_warmup_steps": 16*100,
                        "num_training_steps": 16*3000}
    if debug:
        epochs: int = 2
    else:
        epochs: int = 15

    optimizer: Any = AdamW
    weight_decay = 0.1


class CommonLitModule(LightningModule):
    def __init__(self,
                 config: Config,
                 seed: int = 19900222):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.bert = AutoModel.from_pretrained(self.config.nlp_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.nlp_model_name)
        self.dropout_bert_stack = nn.Dropout(self.config.dropout_stack)
        self.seed = seed
        pl.seed_everything(seed)

        # network config
        self.linear = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.config.linear_dim),
            nn.Dropout(self.config.dropout),
            # nn.PReLU(),
            nn.Linear(self.config.linear_dim, 1)
        )

        self.df_train = None
        self.df_val = None

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)[2]
        x = torch.stack([self.dropout_bert_stack(x) for x in x[-4:]]).mean(dim=0)
        x = torch.sum(
            x * attention_mask.unsqueeze(-1), dim=1, keepdim=False
        )
        x = x / torch.sum(attention_mask, dim=-1, keepdim=True)

        x = self.linear(x)

        return x

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, target = batch
        output = self(input_ids, attention_mask)
        loss = torch.sqrt(F.mse_loss(output.flatten(), target.flatten()))
        self.log("rmse_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, target = batch
        output = self(input_ids, attention_mask)
        loss = torch.sqrt(F.mse_loss(output.flatten(), target.flatten()))
        self.log('rmse_val_loss', loss, prog_bar=True)
        return loss

    def setup(self, stage=None):
        df = pd.read_csv("input/commonlitreadabilityprize/train.csv")
        if config.debug:
            df = df.iloc[:100]
        self.df_val = df.iloc[self.config.fold::5]
        self.df_train = df[~df.index.isin(self.df_val.index)]

    def configure_optimizers(self):
        optimizer = self.config.optimizer(params=[{"params": self.bert.parameters(), "lr": config.lr_bert},
                                                  {"params": self.linear.parameters(), "lr": config.lr_fc}],
                                          weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, **self.config.scheduler_params)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = CommonLitDataset(df=self.df_train,
                                   tokenizer=self.tokenizer)
        return DataLoader(dataset, batch_size=self.config.batch_size)

    def val_dataloader(self):
        dataset = CommonLitDataset(df=self.df_val,
                                   tokenizer=self.tokenizer)
        return DataLoader(dataset, batch_size=self.config.batch_size)


def main(config: Config):
    model = CommonLitModule(config=config)
    trainer = Trainer(gpus=1, max_epochs=config.epochs, default_root_dir=f"../output/{os.path.basename(__file__)[:-3]}")
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        for key, value in config.__dict__.items():
            mlflow.log_param(key, value)
        trainer.fit(model)


if __name__ == "__main__":
    for nlp_model_name in [
        "roberta-large",
    ]:
        for lr_bert in [3e-5, 5e-5, 1e-4]:
            for lr_fc in [1e-4, 5e-4, 1e-3]:
                for dropout in [0, 0.2]:
                    for weight_decay in [0.01, 0.1]:
                        try:
                            experiment_name = f"exp001_{nlp_model_name}_lrbert{lr_bert}_lrfc{lr_fc}_dropout{dropout}"
                            config = Config(experiment_name=experiment_name)
                            config.lr_bert = lr_bert
                            config.lr_fc = lr_fc
                            config.dropout = dropout
                            config.nlp_model_name = nlp_model_name
                            config.weight_decay = weight_decay
                            main(config)
                        except Exception as e:
                            print(e)

