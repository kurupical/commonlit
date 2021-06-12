import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
import pandas as pd
import dataclasses
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
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


class CommonLitDataset(Dataset):
    def __init__(self, df, tokenizer, cfg, transforms=None):
        self.df = df.reset_index()
        self.augmentations = transforms
        self.cfg = cfg
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        text = row["excerpt"]

        text = self.tokenizer(text,
                              padding="max_length",
                              max_length=256,
                              truncation=True,
                              return_tensors="pt",
                              return_token_type_ids=True)
        input_ids = text["input_ids"][0].detach().cpu().numpy()
        input_ids_masked = [x if np.random.random() > self.cfg.mask_p else self.tokenizer.mask_token_id for x in input_ids]
        input_ids_masked = torch.LongTensor(input_ids_masked).to("cuda")
        attention_mask = text["attention_mask"][0]
        token_type_ids = text["token_type_ids"][0]

        target = torch.tensor(row["target"], dtype=torch.float)
        return input_ids_masked, attention_mask, token_type_ids, input_ids, target


@dataclasses.dataclass
class Config:
    experiment_name: str
    seed: int = 19900222
    debug: bool = False
    fold: int = 0

    nlp_model_name: str = "roberta-base"
    linear_dim: int = 128
    dropout: float = 0.2
    dropout_stack: float = 0.1
    batch_size: int = 16

    lr_bert: float = 3e-5
    lr_fc: float = 1e-3
    lr_rnn: float = 1e-3
    warmup_ratio: float = 0
    if debug:
        epochs: int = 2
    else:
        epochs: int = 8

    activation: Any = nn.GELU
    optimizer: Any = AdamW
    weight_decay: float = 0.1

    rnn_module: nn.Module = nn.LSTM
    rnn_module_num: int = 0
    rnn_module_dropout: float = 0
    rnn_module_activation: Any = None
    rnn_module_shrink_ratio: float = 1

    augmantation_range: Tuple[float, float] = (0, 0)
    lr_bert_decay: float = 0.99

    multi_dropout_ratio: float = 0.3
    multi_dropout_num: int = 5
    fine_tuned_path: str = None

    mask_p: float = 0

    perplexity_linear_dim: int = 64

class LSTMModule(nn.Module):
    def __init__(self, cfg, hidden_size):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        hidden_out = int(hidden_size * cfg.rnn_module_shrink_ratio)
        self.rnn_module = self.cfg.rnn_module(hidden_size, hidden_out)
        self.layer_norm = nn.LayerNorm(hidden_out)
        self.rnn_module_activation = self.cfg.rnn_module_activation
        self.dropout = nn.Dropout(self.cfg.rnn_module_dropout)

    def forward(self, x):
        x = self.rnn_module(x)[0]
        x = self.layer_norm(x)
        x = self.dropout(x)
        if not self.rnn_module_activation is None:
            x = self.rnn_module_activation(x)
        return x

def fix_key(state_dict):
    ret = {}
    for k, v in state_dict.items():
        k = k.replace("bert.", "").replace("roberta.", "")
        ret[k] = v
    return ret

class CommonLitModule(LightningModule):
    def __init__(self,
                 cfg: Config,
                 output_dir: str):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.output_dir = output_dir
        if self.cfg.fine_tuned_path is not None:
            self.bert = AutoModelForMaskedLM.from_pretrained(self.cfg.fine_tuned_path)
        else:
            self.bert = AutoModelForMaskedLM.from_pretrained(self.cfg.nlp_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.nlp_model_name)
        if "gpt" in self.cfg.nlp_model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dropout_bert_stack = nn.Dropout(self.cfg.dropout_stack)
        pl.seed_everything(self.cfg.seed)
        self.lstm = self.make_lstm_module()

        # network cfg
        hidden_size = int(self.bert.config.hidden_size * (self.cfg.rnn_module_shrink_ratio**self.cfg.rnn_module_num))

        self.linear_perp = nn.Sequential(
            nn.Linear(1, self.cfg.perplexity_linear_dim),
            nn.Dropout(self.cfg.dropout),
            self.cfg.activation()
        )
        self.linear1 = nn.Sequential(
            nn.Linear(hidden_size+self.cfg.perplexity_linear_dim, self.cfg.linear_dim),
            nn.Dropout(self.cfg.dropout),
            self.cfg.activation()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(self.cfg.linear_dim+self.cfg.perplexity_linear_dim, 1)
        )

        self.df_train: pd.DataFrame
        self.df_val: pd.DataFrame
        self.dataset_train: Dataset
        self.dataset_val: Dataset

        self.best_rmse = np.inf

    def make_lstm_module(self):
        ret = []
        hidden_size = self.bert.config.hidden_size

        for i in range(self.cfg.rnn_module_num):
            ret.append((f"lstm_module_{i}", LSTMModule(cfg=self.cfg, hidden_size=hidden_size)))
            hidden_size = int(hidden_size * self.cfg.rnn_module_shrink_ratio)
        return nn.Sequential(OrderedDict(ret))

    def forward(self, input_ids_masked, attention_mask, token_type_ids, input_ids):
        def f(x_in, perplexity):
            x_out = F.dropout(x_in, p=self.cfg.multi_dropout_ratio, training=True)
            x_out = self.linear1(torch.cat([x_out, perplexity], dim=1))
            x_out = self.linear2(torch.cat([x_out, perplexity], dim=1))
            return x_out

        # if "deberta" in self.cfg.nlp_model_name:
        #     x = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)[1]
        #     x = torch.stack([self.dropout_bert_stack(x) for x in x[-4:]]).mean(dim=0)
        #     x = torch.sum(
        #         x * attention_mask.unsqueeze(-1), dim=1, keepdim=False
        #     )
        #     x = x / torch.sum(attention_mask, dim=-1, keepdim=True)
        # elif "xlnet" in self.cfg.nlp_model_name:
        #     x = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0].mean(dim=1)
        # else:
        x = self.bert.roberta(input_ids=input_ids_masked, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        input_ids_pred = self.bert.lm_head(x[0])

        loss = torch.nn.functional.cross_entropy(input_ids_pred.view(-1, self.bert.config.vocab_size), input_ids.view(-1), reduction="none")
        perplexity = loss.view(len(input_ids), -1).mean(dim=1).view(-1, 1)

        x = torch.stack([self.dropout_bert_stack(x) for x in x[1][-4:]]).mean(dim=0)
        x = torch.sum(
            x * attention_mask.unsqueeze(-1), dim=1, keepdim=False
        )
        x = x / torch.sum(attention_mask, dim=-1, keepdim=True)

        perplexity = self.linear_perp(perplexity)
        x = torch.stack([f(x, perplexity) for _ in range(self.cfg.multi_dropout_num)]).mean(dim=0)

        return x

    def training_step(self, batch, batch_idx):
        scheduler = self.lr_schedulers()
        scheduler.step()

        input_ids_masked, attention_mask, token_type_ids, input_ids, target = batch
        output = self.forward(input_ids_masked, attention_mask, token_type_ids, input_ids)
        loss = F.mse_loss(output.flatten(), target.flatten())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids_masked, attention_mask, token_type_ids, input_ids, target = batch
        output = self.forward(input_ids_masked, attention_mask, token_type_ids, input_ids)
        loss = F.mse_loss(output.flatten(), target.flatten())
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

        df_val.to_csv(f"{self.output_dir}/val_fold{self.cfg.fold}_step{str(self.global_step).zfill(5)}.csv", index=False)

        rmse = np.sqrt(1 / len(pred) * ((target - pred)**2).sum())
        self.log(f"rmse_fold{self.cfg.fold}", rmse, prog_bar=True)

        if self.best_rmse > rmse:
            self.log(f"best_rmse_fold{self.cfg.fold}", rmse, prog_bar=True)
            self.log(f"best_step_fold{self.cfg.fold}", self.global_step)
            self.best_rmse = rmse
            df_val.to_csv(f"{self.output_dir}/val_fold{self.cfg.fold}_best.csv", index=False)

    def setup(self, stage=None):
        df = pd.read_csv("input/commonlitreadabilityprize/train_folds.csv")
        if self.cfg.debug:
            df = df.iloc[::30]

        self.df_train = df[df["kfold"] != self.cfg.fold].reset_index(drop=True)
        self.df_val = df[df["kfold"] == self.cfg.fold].reset_index(drop=True)

        self.df_train = pd.concat([self.df_train,
                                   self.df_train[(cfg.augmantation_range[0] < self.df_train["target"]) &
                                                 (self.df_train["target"] < cfg.augmantation_range[1])]])
        self.dataset_train = CommonLitDataset(df=self.df_train,
                                              tokenizer=self.tokenizer,
                                              cfg=self.cfg)
        self.dataset_val = CommonLitDataset(df=self.df_val,
                                            tokenizer=self.tokenizer,
                                            cfg=self.cfg)

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
                ret["weight_decay"] = self.cfg.weight_decay
                ret["lr"] = self.cfg.lr_bert * (self.cfg.lr_bert_decay ** (layers - i + 1))
                params.append(ret)

                ret = {}
                ret["params"] = [p for n, p in self.bert.named_parameters()
                                 if f"bert.encoder.layer.{i}." in n and any(nd in n for nd in no_decay_ary)]
                ret["weight_decay"] = 0
                ret["lr"] = self.cfg.lr_bert * (self.cfg.lr_bert_decay ** (layers - i + 1))
                params.append(ret)
            return params


        params = []
        params.extend(bert_params())
        params.append(extract_params(self.linear1.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
        params.append(extract_params(self.linear1.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
        params.append(extract_params(self.linear2.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
        params.append(extract_params(self.linear2.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
        params.append(extract_params(self.linear_perp.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
        params.append(extract_params(self.linear_perp.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
        params.append(extract_params(self.lstm.named_parameters(), lr=self.cfg.lr_rnn, weight_decay=self.cfg.weight_decay, no_decay=False))
        params.append(extract_params(self.lstm.named_parameters(), lr=self.cfg.lr_rnn, weight_decay=0, no_decay=True))

        optimizer = self.cfg.optimizer(params)
        num_warmup_steps = int(self.cfg.epochs * len(self.df_train) / self.cfg.batch_size * self.cfg.warmup_ratio)
        num_training_steps = int(self.cfg.epochs * len(self.df_train) / self.cfg.batch_size)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.cfg.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.cfg.batch_size)

def main(cfg: Config,
         folds: List):
    output_dir = f"output/{os.path.basename(__file__)[:-3]}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir)
    rmse = 0
    with mlflow.start_run() as run:
        mlflow.pytorch.autolog(log_models=False)
        for key, value in cfg.__dict__.items():
            mlflow.log_param(key, value)
        with open(f"{output_dir}/cfg.pickle", "wb") as f:
            pickle.dump(cfg, f)
        for fold in folds:
            try:
                cfg.fold = fold
                checkpoint_callback = ModelCheckpoint(
                    monitor='val_loss',
                    dirpath=output_dir,
                    filename=f'best_fold{fold}',
                    save_top_k=1,
                    mode='min',
                )

                model = CommonLitModule(cfg=cfg,
                                        output_dir=output_dir)
                trainer = Trainer(gpus=1,
                                  precision=16,
                                  # amp_level="02",
                                  max_epochs=cfg.epochs,
                                  benchmark=True,
                                  val_check_interval=0.05,
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
    experiment_name = "perplexity"
    folds = [0, 1, 2, 3, 4]
    """
    for lr_bert in [3e-5, 5e-5]:
        for perplexity_linear_dim in [8, 16, 32, 64, 128]:
            cfg = Config(experiment_name=experiment_name)
            cfg.nlp_model_name = "roberta-base"
            cfg.lr_bert = lr_bert
            cfg.perplexity_linear_dim = perplexity_linear_dim
            main(cfg, folds=folds)
    """
    for mask_p in [0.05, 0.1]:
        cfg = Config(experiment_name=experiment_name)
        cfg.mask_p = mask_p
        main(cfg, folds=folds)