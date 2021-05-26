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
import os
import numpy as np
from pytorch_lightning import Trainer
import mlflow.pytorch
from datetime import datetime as dt
from pytorch_lightning.callbacks import ModelCheckpoint

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
    dropout: float = 0.5
    dropout_stack: float = 0.2
    batch_size: int = 16

    lr_bert: float = 2e-5
    lr_fc: float = 1e-4
    scheduler_params = {"num_warmup_steps": 16*100,
                        "num_training_steps": 16*2500}
    if debug:
        epochs: int = 2
    else:
        epochs: int = 15

    optimizer: Any = AdamW
    weight_decay = 0.1


class CommonLitModule(LightningModule):
    def __init__(self,
                 config: Config,
                 output_dir: str,
                 seed: int = 19900222):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.output_dir = output_dir
        self.bert = AutoModel.from_pretrained(self.config.nlp_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.nlp_model_name)
        self.dropout_bert_stack = nn.Dropout(self.config.dropout_stack)
        self.seed = seed
        pl.seed_everything(seed)

        # network config
        self.linear = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.config.linear_dim),
            nn.Dropout(self.config.dropout),
            nn.PReLU(),
            nn.Linear(self.config.linear_dim, 1)
        )

        self.df_train: pd.DataFrame
        self.df_val: pd.DataFrame
        self.dataset_train: Dataset
        self.dataset_val: Dataset

        self.best_rmse = np.inf

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
        scheduler = self.lr_schedulers()
        scheduler.step()

        input_ids, attention_mask, target = batch
        output = self.forward(input_ids, attention_mask)
        loss = torch.sqrt(F.mse_loss(output.flatten(), target.flatten()))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, target = batch
        output = self.forward(input_ids, attention_mask)
        loss = torch.sqrt(F.mse_loss(output.flatten(), target.flatten()))
        self.log('val_loss', loss, prog_bar=True)
        return output.cpu().detach().numpy().flatten(), target.cpu().detach().numpy().flatten()

    def validation_epoch_end(self, val_step_outputs):
        if len(val_step_outputs) > 5:
            pred = []
            target = []
            for output in val_step_outputs:
                pred.extend(output[0].tolist())
                target.extend(output[1].tolist())

            pred = np.array(pred)
            target = np.array(target)

            df_val = self.df_val[["id"]]
            df_val["pred"] = pred
            df_val["target"] = target

            df_val.to_csv(f"{self.output_dir}/val_epoch{self.current_epoch}.csv", index=False)

            rmse = np.sqrt(1 / len(pred) * ((target - pred)**2).sum())
            self.log("rmse", rmse, prog_bar=True)

            if self.best_rmse > rmse:
                self.log("best_rmse", rmse, prog_bar=True)
                self.best_rmse = rmse

    def setup(self, stage=None):
        df = pd.read_csv("input/commonlitreadabilityprize/train.csv")
        if config.debug:
            df = df.iloc[:100]

        val_idx = np.arange(self.config.fold, len(df), 5)
        train_idx = [i for i in range(len(df)) if i not in val_idx]
        self.df_val = df.iloc[val_idx].reset_index(drop=True)
        self.df_train = df.iloc[train_idx].reset_index(drop=True)
        self.dataset_train = CommonLitDataset(df=self.df_train,
                                              tokenizer=self.tokenizer)
        self.dataset_val = CommonLitDataset(df=self.df_val,
                                            tokenizer=self.tokenizer)

    def configure_optimizers(self):
        optimizer = self.config.optimizer(params=[{"params": self.bert.parameters(), "lr": config.lr_bert},
                                                  {"params": self.linear.parameters(), "lr": config.lr_fc}],
                                          # self.parameters(),
                                          weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, **self.config.scheduler_params)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.config.batch_size)


def main(config: Config):
    output_dir = f"output/{os.path.basename(__file__)[:-3]}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_dir,
        filename='best',
        save_top_k=1,
        mode='min',
    )

    model = CommonLitModule(config=config,
                            output_dir=output_dir)
    trainer = Trainer(gpus=1,
                      max_epochs=config.epochs,
                      progress_bar_refresh_rate=1,
                      default_root_dir=output_dir,
                      callbacks=[checkpoint_callback])
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        for key, value in config.__dict__.items():
            mlflow.log_param(key, value)
        trainer.fit(model)

if __name__ == "__main__":
    for nlp_model_name in [
        "roberta-large",
        # "bert-base-cased"
    ]:
        for lr_bert in [3e-5]:
            for lr_fc in [1e-5]:
                for dropout in [0]:
                    for weight_decay in [0.1]:
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

