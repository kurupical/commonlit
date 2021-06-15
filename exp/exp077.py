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
from torch.nn.utils import weight_norm
import copy
import timm

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
                              max_length=self.cfg.max_length,
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

class LSTMModule(nn.Module):
    def __init__(self, cfg, hidden_size):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        hidden_out = int(hidden_size * cfg.rnn_module_shrink_ratio)
        self.rnn_module = self.cfg.rnn_module(hidden_size,
                                              hidden_out,
                                              bidirectional=self.cfg.bidirectional,
                                              batch_first=True,
                                              dropout=self.cfg.rnn_module_dropout)
        if self.cfg.bidirectional:
            self.layer_norm = nn.LayerNorm(hidden_out*2)
            self.rnn_module_activation = self.cfg.rnn_module_activation
        else:
            self.layer_norm = nn.LayerNorm(hidden_out*2)
            self.rnn_module_activation = self.cfg.rnn_module_activation

    def forward(self, x):
        x = self.rnn_module(x)[0]
        x = self.layer_norm(x)
        if not self.rnn_module_activation is None:
            x = self.rnn_module_activation(x)
        return x

def fix_key(state_dict):
    ret = {}
    for k, v in state_dict.items():
        k = k.replace("bert.", "").replace("roberta.", "")
        ret[k] = v
    return ret



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=(kernel_size-1)*dilation,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=(kernel_size-1)*dilation,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, padding=(kernel_size-1)*dilation) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@dataclasses.dataclass
class Config:
    experiment_name: str
    seed: int = 19900222
    debug: bool = False
    fold: int = 0

    nlp_model_name: str = "roberta-base"
    linear_dim: int = 128
    linear_vocab_dim_1: int = 64
    linear_vocab_dim: int = 16
    linear_perplexity_dim: int = 64
    dropout: float = 0.2
    dropout_stack: float = 0.1
    dropout_output_hidden: float = 0.2
    dropout_attn: float = 0
    batch_size: int = 16

    lr_bert: float = 3e-5
    lr_fc: float = 1e-3
    lr_rnn: float = 1e-3
    lr_tcn: float = 1e-3
    lr_cnn: float = 1e-3
    warmup_ratio: float = 0.05
    training_steps_ratio: float = 1
    if debug:
        epochs: int = 2
    else:
        epochs: int = 6
        epochs_max: int = 8

    activation: Any = nn.GELU
    optimizer: Any = AdamW
    weight_decay: float = 0.1

    rnn_module: nn.Module = nn.LSTM
    rnn_module_num: int = 0
    rnn_module_dropout: float = 0
    rnn_module_activation: Any = None
    rnn_module_shrink_ratio: float = 0.25
    rnn_hidden_indice: Tuple[int] = (-1, 0)
    bidirectional: bool = True

    tcn_module_enable: bool = False
    tcn_module_num: int = 3
    tcn_module: nn.Module = TemporalConvNet
    tcn_module_kernel_size: int = 4
    tcn_module_dropout: float = 0.2

    linear_vocab_enable: bool = False
    augmantation_range: Tuple[float, float] = (0, 0)
    lr_bert_decay: float = 0.99

    multi_dropout_ratio: float = 0.3
    multi_dropout_num: int = 10
    fine_tuned_path: str = None

    # convnet
    cnn_model_name: str = "resnet18"
    cnn_pretrained: bool = False
    self_attention_enable: bool = False

    mask_p: float = 0
    max_length: int = 256

    hidden_stack_enable: bool = False
    prep_enable: bool = True

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
        self.tcn = self.cfg.tcn_module(num_inputs=self.bert.config.hidden_size,
                                       num_channels=[self.bert.config.hidden_size]*cfg.tcn_module_num,
                                       kernel_size=self.cfg.tcn_module_kernel_size,
                                       dropout=self.cfg.tcn_module_dropout)

        self.convnet = timm.create_model(self.cfg.cnn_model_name,
                                         pretrained=self.cfg.cnn_pretrained,
                                         num_classes=0)
        if "efficientnet" in self.cfg.cnn_model_name:
            self.convnet.conv_stem = nn.Conv2d(self.bert.config.num_hidden_layers*self.bert.config.num_attention_heads,
                                               32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        if "resnet" in self.cfg.cnn_model_name:
            self.convnet.conv1 = nn.Conv2d(self.bert.config.num_hidden_layers*self.bert.config.num_attention_heads,
                                           64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # network cfg
        hidden_size = 0
        if self.cfg.linear_vocab_enable:
            hidden_size += self.cfg.linear_vocab_dim * self.cfg.max_length
        if self.cfg.self_attention_enable:
            hidden_size += self.convnet.num_features
        if self.cfg.hidden_stack_enable:
            hidden_size += self.bert.config.hidden_size
        if self.cfg.rnn_module_num > 0:
            if self.cfg.bidirectional:
                hidden_size += int(self.bert.config.hidden_size * len(self.cfg.rnn_hidden_indice) * ((2*self.cfg.rnn_module_shrink_ratio)**self.cfg.rnn_module_num))
            else:
                hidden_size += int(self.bert.config.hidden_size * len(self.cfg.rnn_hidden_indice) * (self.cfg.rnn_module_shrink_ratio**self.cfg.rnn_module_num))
        if self.cfg.tcn_module_enable:
            hidden_size += self.bert.config.hidden_size

        self.linear_perp = nn.Sequential(
            nn.Linear(1, self.cfg.linear_perplexity_dim),
            # nn.BatchNorm1d(self.cfg.perplexity_linear_dim),
            nn.Dropout(self.cfg.dropout),
            self.cfg.activation()
        )
        self.linear_vocab = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.cfg.linear_vocab_dim_1),
            nn.Dropout(self.cfg.dropout),
            self.cfg.activation(),
            nn.Linear(self.cfg.linear_vocab_dim_1, self.cfg.linear_vocab_dim),
            nn.Dropout(self.cfg.dropout),
            self.cfg.activation()
        )
        if self.cfg.prep_enable:
            self.linear1 = nn.Sequential(
                nn.Linear(hidden_size + self.cfg.linear_perplexity_dim, self.cfg.linear_dim),
                nn.Dropout(self.cfg.dropout),
                self.cfg.activation()
            )
            self.linear2 = nn.Sequential(
                nn.Linear(self.cfg.linear_dim + self.cfg.linear_perplexity_dim, 1)
            )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(hidden_size, self.cfg.linear_dim),
                nn.Dropout(self.cfg.dropout),
                self.cfg.activation()
            )
            self.linear2 = nn.Sequential(
                nn.Linear(self.cfg.linear_dim, 1)
            )

        self.dropout = nn.Dropout(self.cfg.dropout_output_hidden)
        self.dropout_attn = nn.Dropout(self.cfg.dropout_attn)

        self.df_train: pd.DataFrame
        self.df_val: pd.DataFrame
        self.dataset_train: Dataset
        self.dataset_val: Dataset

        self.best_rmse = np.inf


    def make_lstm_module(self):
        ret = []
        hidden_size = self.bert.config.hidden_size * len(self.cfg.rnn_hidden_indice)

        for i in range(self.cfg.rnn_module_num):
            ret.append((f"lstm_module_{i}", LSTMModule(cfg=self.cfg, hidden_size=hidden_size)))
            if self.cfg.bidirectional:
                hidden_size = int(hidden_size * self.cfg.rnn_module_shrink_ratio * 2)
            else:
                hidden_size = int(hidden_size * self.cfg.rnn_module_shrink_ratio)
        return nn.Sequential(OrderedDict(ret))

    def forward(self, input_ids_masked, attention_mask, token_type_ids, input_ids):
        def f(x_in, perplexity=None):
            x_out = F.dropout(x_in, p=self.cfg.multi_dropout_ratio, training=True)
            if perplexity is not None:
                x_out = self.linear1(torch.cat([x_out, perplexity], dim=1))
                x_out = self.linear2(torch.cat([x_out, perplexity], dim=1))
            else:
                x_out = self.linear1(x_out)
                x_out = self.linear2(x_out)
            return x_out

        if "roberta" in self.cfg.nlp_model_name:
            x = self.bert.roberta(input_ids=input_ids_masked,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  output_attentions=True,
                                  output_hidden_states=True)
            if self.cfg.prep_enable:
                input_ids_pred = self.bert.lm_head(x[0])
        elif "bert" in self.cfg.nlp_model_name:
            x = self.bert.bert(input_ids=input_ids_masked,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               output_attentions=True,
                               output_hidden_states=True)
            if self.cfg.prep_enable:
                input_ids_pred = self.bert.cls(x[0])

        # x[0]: last hidden layer, x[1]: all hidden layer, x[2]: attention matrix
        if self.cfg.prep_enable:
            loss = torch.nn.functional.cross_entropy(input_ids_pred.view(-1, self.bert.config.vocab_size), input_ids.view(-1), reduction="none")
            perplexity = loss.view(len(input_ids), -1) * attention_mask
            perplexity = perplexity.sum(dim=1) / attention_mask.sum(dim=1)
            perplexity = perplexity.view(-1, 1)

        # base feature
        x_bert = []
        if self.cfg.linear_vocab_enable:
            x_bert.append(self.dropout(self.linear_vocab(x[0]).view(len(input_ids), -1)))
        if self.cfg.self_attention_enable:
            xx = torch.cat([self.dropout_attn(xx) for xx in x[2]], dim=1)
            x_bert.append(self.convnet(xx))
        if self.cfg.hidden_stack_enable:
            xx = torch.stack([self.dropout_bert_stack(xx) for xx in x[1][-4:]]).mean(dim=0)
            xx = torch.sum(
                xx * attention_mask.unsqueeze(-1), dim=1, keepdim=False
            )
            xx = xx / torch.sum(attention_mask, dim=-1, keepdim=True)
            x_bert.append(xx)

        # residual feature
        if self.cfg.rnn_module_num > 0:
            x_lstm = self.lstm(torch.cat([x[1][idx] for idx in self.cfg.rnn_hidden_indice], dim=2)).mean(dim=1)
            x_bert.append(x_lstm)
        if self.cfg.tcn_module_enable:
            x_tcn = self.tcn(self.dropout(x[0]).permute(0, 2, 1)).mean(dim=2)
            x_bert.append(x_tcn)

        x_bert = torch.cat(x_bert, dim=1)

        if self.cfg.prep_enable:
            perplexity = self.linear_perp(perplexity)
            x_out = torch.stack([f(x_bert, perplexity) for _ in range(self.cfg.multi_dropout_num)]).mean(dim=0)
        else:
            x_out = torch.stack([f(x_bert) for _ in range(self.cfg.multi_dropout_num)]).mean(dim=0)
        return x_out

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
        params.append(extract_params(self.linear_vocab.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
        params.append(extract_params(self.linear_vocab.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
        params.append(extract_params(self.lstm.named_parameters(), lr=self.cfg.lr_rnn, weight_decay=self.cfg.weight_decay, no_decay=False))
        params.append(extract_params(self.lstm.named_parameters(), lr=self.cfg.lr_rnn, weight_decay=0, no_decay=True))
        params.append(extract_params(self.tcn.named_parameters(), lr=self.cfg.lr_tcn, weight_decay=self.cfg.weight_decay, no_decay=False))
        params.append(extract_params(self.tcn.named_parameters(), lr=self.cfg.lr_tcn, weight_decay=0, no_decay=True))
        params.append(extract_params(self.convnet.named_parameters(), lr=self.cfg.lr_cnn, weight_decay=self.cfg.weight_decay, no_decay=False))
        params.append(extract_params(self.convnet.named_parameters(), lr=self.cfg.lr_cnn, weight_decay=0, no_decay=True))

        optimizer = self.cfg.optimizer(params)
        num_warmup_steps = int(self.cfg.epochs_max * len(self.df_train) / self.cfg.batch_size * self.cfg.warmup_ratio)
        num_training_steps = int(self.cfg.epochs_max * len(self.df_train) / self.cfg.batch_size) * self.cfg.training_steps_ratio

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

def main(cfg_original: Config,
         folds: List):
    output_dir = f"output/{os.path.basename(__file__)[:-3]}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir)
    rmse = 0
    with mlflow.start_run() as run:
        mlflow.pytorch.autolog(log_models=False)
        for key, value in cfg_original.__dict__.items():
            mlflow.log_param(key, value)
        with open(f"{output_dir}/cfg.pickle", "wb") as f:
            pickle.dump(cfg_original, f)
        for fold in folds:
            try:
                cfg = copy.copy(cfg_original)
                cfg.fold = fold
                checkpoint_callback = ModelCheckpoint(
                    monitor='val_loss',
                    dirpath=output_dir,
                    filename=f'best_fold{fold}',
                    save_top_k=1,
                    mode='min',
                    save_weights_only=True
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

def config_large(cfg: Config):
    cfg.nlp_model_name = "roberta-large"
    cfg.lr_bert = 1e-5
    cfg.lr_bert_decay = 1
    cfg.batch_size = 8
    cfg.weight_decay = 0.01
    return cfg

if __name__ == "__main__":
    experiment_name = "FIX LSTM"
    folds = [0, 1, 2, 3, 4]

    """
    # for rnn_hidden_indice in [(-1, -2), (-1, -2, -3, -4), (-1, 0)]:
    for rnn_hidden_indice in [(-1, -2, 0, 1)]:
        cfg = Config(experiment_name=experiment_name)
        cfg.rnn_module_num = 1
        cfg.rnn_hidden_indice = rnn_hidden_indice
        cfg.prep_enable = True
        cfg.hidden_stack_enable = True
        main(cfg, folds=folds)
    for rnn_module_num in [1]:
        for shrink in [0.5]:
            cfg = Config(experiment_name=experiment_name)
            cfg.rnn_module_num = rnn_module_num
            cfg.rnn_module_shrink_ratio = shrink
            cfg.prep_enable = True
            cfg.hidden_stack_enable = True
            main(cfg, folds=folds)
    """

    for rnn_module_num in [2, 3]:
        for shrink in [0.5, 0.25, 0.125]:
            cfg = Config(experiment_name=experiment_name)
            cfg.rnn_module_num = rnn_module_num
            cfg.rnn_module_shrink_ratio = shrink
            cfg.prep_enable = True
            cfg.hidden_stack_enable = True
            main(cfg, folds=folds)

