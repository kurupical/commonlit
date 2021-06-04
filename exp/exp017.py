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
try:
    import mlflow.pytorch
except Exception as e:
    print("error: mlflow is not found")
from datetime import datetime as dt
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import List, Tuple
import gc
import pickle
from collections import OrderedDict
from gensim.models import KeyedVectors

class CommonLitDataset(Dataset):
    def __init__(self, df, tokenizer, transforms=None):
        self.df = df.reset_index()
        self.augmentations = transforms
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        def replace_stop_words(x):
            x = x.replace(".", " ")
            x = x.replace(",", " ")
            x = x.replace("!", " ")
            x = x.replace("?", " ")
            x = x.replace("\n", " ")
            x = x.replace(")", " ")
            x = x.replace("(", " ")
            x = x.replace('"', ' ')
            x = x.replace("'", " ")
            x = x.replace(";", " ")
            x = x.replace("  ", " ")
            x = x.replace("  ", " ")
            x = x.replace("  ", " ")
            return x

        row = self.df.iloc[index]

        text = row["excerpt"]
        text = replace_stop_words(text)

        text = self.tokenizer(text, padding="max_length", max_length=300, truncation=True, return_tensors="pt")
        input_ids = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]

        target = torch.tensor(row["target"], dtype=torch.float)
        return input_ids, attention_mask, target


@dataclasses.dataclass
class Config:
    experiment_name: str
    debug: bool = False
    fold: int = 0

    lr: float = 1e-4
    num_warmup_steps: int = 16*100
    num_training_steps: int = 16*2500
    if debug:
        epochs: int = 2
    else:
        epochs: int = 15

    activation: Any = nn.GELU
    optimizer: Any = AdamW
    weight_decay: float = 0.01

    rnn_module: nn.Module = nn.LSTM
    rnn_hidden_size = 256
    rnn_module_num: int = 1
    rnn_module_dropout: float = 0
    rnn_module_activation: Any = None
    rnn_module_shrink_ratio: float = 1

    augmantation_range: Tuple[float, float] = (-2, -0)
    lr_bert_decay: float = 0.98


class LSTMModule(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        hidden_out = int(hidden_size * config.rnn_module_shrink_ratio)
        self.rnn_module = self.config.rnn_module(hidden_size, hidden_out)
        self.layer_norm = nn.LayerNorm(hidden_out)
        self.rnn_module_activation = self.config.rnn_module_activation
        self.dropout = nn.Dropout(self.config.rnn_module_dropout)

    def forward(self, x):
        x = self.rnn_module(x)[0]
        x = self.layer_norm(x)
        x = self.dropout(x)
        if not self.rnn_module_activation is None:
            x = self.rnn_module_activation(x)
        return x

class CommonLitModule(LightningModule):
    def __init__(self,
                 config: Config,
                 output_dir: str,
                 seed: int = 19900222):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.output_dir = output_dir
        self.seed = seed
        pl.seed_everything(seed)
        self.lstm = self.make_lstm_module()

        # network config
        hidden_size = int(self.bert.config.hidden_size * (config.rnn_module_shrink_ratio**self.config.rnn_module_num))
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, self.config.linear_dim),
            nn.Dropout(self.config.dropout),
            config.activation(),
            nn.Linear(self.config.linear_dim, 1)
        )

        self.df_train: pd.DataFrame
        self.df_val: pd.DataFrame
        self.dataset_train: Dataset
        self.dataset_val: Dataset

        self.best_rmse = np.inf

    def make_lstm_module(self):
        ret = []
        hidden_size = self.config.rnn_hidden_size

        for i in range(self.config.rnn_module_num):
            ret.append((f"lstm_module_{i}", LSTMModule(config=self.config, hidden_size=hidden_size)))
            hidden_size = int(hidden_size * config.rnn_module_shrink_ratio)
        return nn.Sequential(OrderedDict(ret))

    def forward(self, x):
        x = self.lstm(x).mean(dim=1)
        x = self.linear(x)

        return x

    def training_step(self, batch, batch_idx):
        scheduler = self.lr_schedulers()
        scheduler.step()

        x, target = batch
        output = self.forward(x)
        loss = torch.sqrt(F.mse_loss(output.flatten(), target.flatten()))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        output = self.forward(x)
        loss = torch.sqrt(F.mse_loss(output.flatten(), target.flatten()))
        self.log('val_loss', loss, prog_bar=True)
        return output.cpu().detach().numpy().flatten(), target.cpu().detach().numpy().flatten()

    def validation_epoch_end(self, val_step_outputs):
        pred = []
        target = []
        for output in val_step_outputs:
            pred.extend(output[0].tolist())
            target.extend(output[1].tolist())

        pred = np.array(pred)
        target = np.array(target)

        if len(pred) != len(self.df_val):
            return

        df_val = self.df_val[["id"]]
        df_val["pred"] = pred
        df_val["target"] = target

        df_val.to_csv(f"{self.output_dir}/val_fold{self.config.fold}_step{self.global_step}.csv", index=False)

        rmse = np.sqrt(1 / len(pred) * ((target - pred)**2).sum())
        self.log(f"rmse_fold{config.fold}", rmse, prog_bar=True)

        if self.best_rmse > rmse:
            self.log(f"best_rmse_fold{config.fold}", rmse, prog_bar=True)
            self.best_rmse = rmse
            df_val.to_csv(f"{self.output_dir}/val_fold{self.config.fold}_best.csv", index=False)

    def setup(self, stage=None):
        df = pd.read_csv("input/commonlitreadabilityprize/train_folds.csv")
        if config.debug:
            df = df.iloc[:100]

        self.df_train = df[df["kfold"] != config.fold].reset_index(drop=True)
        self.df_val = df[df["kfold"] == config.fold].reset_index(drop=True)

        self.df_train = pd.concat([self.df_train,
                                   self.df_train[(config.augmantation_range[0] < self.df_train["target"]) &
                                                 (self.df_train["target"] < config.augmantation_range[1])]])
        self.dataset_train = CommonLitDataset(df=self.df_train,
                                              tokenizer=self.tokenizer)
        self.dataset_val = CommonLitDataset(df=self.df_val,
                                            tokenizer=self.tokenizer)

    def configure_optimizers(self):
        def extract_params(named_parameters, lr, weight_decay, no_decay=False):
            ret = {}
            no_decay_ary = ['bias', 'gamma', 'beta']

            if no_decay:
                ret["params"] = [p for n, p in named_parameters if not any(nd in n for nd in no_decay_ary)]
                ret["weight_decay"] = 0
            else:
                ret["params"] = [p for n, p in named_parameters if any(nd in n for nd in no_decay_ary)]
                ret["weight_decay"] = weight_decay
            ret["lr"] = lr
            return ret

        def bert_params():
            params = []
            no_decay_ary = ['bias', 'gamma', 'beta']
            layers = self.bert.config.num_hidden_layers
            for i in range(layers):
                # models
                # parameters
                ret = {}
                ret["params"] = [p for n, p in self.bert.named_parameters()
                                 if f"encoder.layer.{i}." in n and not any(nd in n for nd in no_decay_ary)]
                ret["weight_decay"] = self.config.weight_decay
                ret["lr"] = self.config.lr_bert * (self.config.lr_bert_decay ** (layers - i + 1))
                params.append(ret)

                ret = {}
                ret["params"] = [p for n, p in self.bert.named_parameters()
                                 if f"bert.encoder.layer.{i}." in n and any(nd in n for nd in no_decay_ary)]
                ret["weight_decay"] = 0
                ret["lr"] = self.config.lr_bert * (self.config.lr_bert_decay ** (layers - i + 1))
                params.append(ret)
            return params


        params = []
        params.extend(bert_params())
        params.append(extract_params(self.linear.named_parameters(), lr=config.lr_fc, weight_decay=config.weight_decay, no_decay=False))
        params.append(extract_params(self.linear.named_parameters(), lr=config.lr_fc, weight_decay=0, no_decay=True))
        params.append(extract_params(self.lstm.named_parameters(), lr=config.lr_rnn, weight_decay=config.weight_decay, no_decay=False))
        params.append(extract_params(self.lstm.named_parameters(), lr=config.lr_rnn, weight_decay=0, no_decay=True))

        optimizer = self.config.optimizer(params)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=config.num_warmup_steps,
                                                    num_training_steps=config.num_training_steps)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.config.batch_size)


def main(config: Config,
         folds: List):
    output_dir = f"output/{os.path.basename(__file__)[:-3]}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir)
    rmse = 0
    with mlflow.start_run() as run:
        mlflow.pytorch.autolog(log_models=False)
        for key, value in config.__dict__.items():
            mlflow.log_param(key, value)
        with open(f"{output_dir}/config.pickle", "wb") as f:
            pickle.dump(config, f)
        for fold in folds:
            try:
                config.fold = fold
                checkpoint_callback = ModelCheckpoint(
                    monitor='val_loss',
                    dirpath=output_dir,
                    filename=f'best_fold{fold}',
                    save_top_k=1,
                    mode='min',
                )

                model = CommonLitModule(config=config,
                                        output_dir=output_dir)
                trainer = Trainer(gpus=1,
                                  # precision=16,
                                  # amp_level="02",
                                  max_epochs=config.epochs,
                                  val_check_interval=0.2,
                                  progress_bar_refresh_rate=1,
                                  default_root_dir=output_dir,
                                  callbacks=[checkpoint_callback])

                trainer.fit(model)
                rmse += model.best_rmse
                del trainer, model
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)
        mlflow.log_metric("rmse_mean", rmse / len(folds))

if __name__ == "__main__":

    for model in ["roberta-base"]:
        experiment_name = "LSTM Only"
        folds = [0, 1, 2, 3, 4]

        for lr in [5e-5, 3e-5, 1e-4]:
            config = Config(experiment_name=experiment_name)
            config.nlp_model_name = model
            config.lr = lr
            main(config, folds=folds)


