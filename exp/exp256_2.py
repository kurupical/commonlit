import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
import pandas as pd
import dataclasses
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, T5EncoderModel, T5Tokenizer
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
from pytorch_lightning.utilities import rank_zero_warn
from typing import Optional

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

        text_original = row["excerpt"]

        text = self.tokenizer(text_original,
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
        std = row["standard_error"]

        features = ((row[self.cfg.feature_columns].fillna(0).values - self.cfg.feature_mean) / self.cfg.feature_std)
        features = torch.tensor(features, dtype=torch.float)

        target = torch.tensor(row["target"], dtype=torch.float)
        return input_ids_masked, attention_mask, token_type_ids, input_ids, features, target, std

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
        else:
            self.layer_norm = nn.LayerNorm(hidden_out * 2)
        if self.cfg.rnn_module_activation is None:
            self.rnn_module_activation = None
        else:
            self.rnn_module_activation = self.cfg.rnn_module_activation()

    def forward(self, x):
        x = self.rnn_module(x)[0]
        x = self.layer_norm(x)
        if self.rnn_module_activation is not None:
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


class NoActivation(nn.Module):
    def __init__(self):
        super(NoActivation, self).__init__()

    def forward(self, x):
        return x

@dataclasses.dataclass
class Config:
    experiment_name: str
    seed: int = 10
    debug: bool = False
    fold: int = 0

    nlp_model_name: str = "roberta-base"
    linear_dim: int = 64
    linear_vocab_dim_1: int = 64
    linear_vocab_dim: int = 16
    linear_perplexity_dim: int = 64
    linear_final_dim: int = 256
    dropout: float = 0
    dropout_stack: float = 0
    dropout_output_hidden: float = 0
    dropout_attn: float = 0
    batch_size: int = 32

    lr_bert: float = 3e-5
    lr_fc: float = 5e-5
    lr_rnn: float = 1e-3
    lr_tcn: float = 1e-3
    lr_cnn: float = 1e-3
    warmup_ratio: float = 0.1
    training_steps_ratio: float = 1
    if debug:
        epochs: int = 2
        epochs_max: int = 8
    else:
        epochs: int = 6
        epochs_max: int = 6

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
    lr_bert_decay: float = 1

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
    prep_enable: bool = False
    kl_div_enable: bool = False

    # reinit
    reinit_pooler: bool = True
    reinit_layers: int = 4

    # pooler
    pooler_enable: bool = True

    word_axis: bool = False

    # conv1d
    conv1d_num: int = 1
    conv1d_stride: int = 2
    conv1d_kernel_size: int = 2

    attention_pool_enable: bool = False
    conv2d_hidden_channel: int = 32

    simple_structure: bool = False
    crossentropy: bool = False
    crossentropy_min: int = -8
    crossentropy_max: int = 4

    accumulate_grad_batches: int = 1
    gradient_clipping: int = 0.2

    dropout_bert: float = 0

    feature_enable: bool = False
    decoder_only: bool = True

    stochastic_weight_avg: bool = False
    val_check_interval: float = 0.05

    attention_head_enable: bool = False

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_targets):
        super().__init__()
        self.in_features = in_features

        self.hidden_layer = nn.Linear(in_features, hidden_dim)
        self.final_layer = nn.Linear(hidden_dim, num_targets)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.hidden_layer(features))
        score = self.final_layer(att)
        attention_weights = torch.softmax(score, dim=1)
        return attention_weights


class CommonLitModule(LightningModule):
    def __init__(self,
                 cfg: Config,
                 output_dir: str):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.output_dir = output_dir
        if self.cfg.prep_enable:
            if self.cfg.fine_tuned_path is not None:
                self.bert = AutoModelForMaskedLM.from_pretrained(self.cfg.fine_tuned_path)
            else:
                self.bert = AutoModelForMaskedLM.from_pretrained(self.cfg.nlp_model_name)
        else:
            if self.cfg.fine_tuned_path is not None:
                self.bert = AutoModel.from_pretrained(self.cfg.fine_tuned_path)
            else:
                if "t5" in self.cfg.nlp_model_name:
                    self.bert = T5EncoderModel.from_pretrained(self.cfg.nlp_model_name)
                else:
                    self.bert = AutoModel.from_pretrained(self.cfg.nlp_model_name)
        if "t5" in self.cfg.nlp_model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(self.cfg.nlp_model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.nlp_model_name)
        if "gpt" in self.cfg.nlp_model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # setting bert dropout
        for module in self.bert.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.cfg.dropout_bert
        self.dropout_bert_stack = nn.Dropout(self.cfg.dropout_stack)
        pl.seed_everything(self.cfg.seed)

        # network cfg
        hidden_size = 0
        if self.cfg.linear_vocab_enable:
            hidden_size += self.cfg.linear_final_dim
            if self.cfg.nlp_model_name == "funnel-transformer/large-base":
                input_size = self.bert.config.hidden_size / 2
            else:
                input_size = self.bert.config.hidden_size
            self.linear_vocab = nn.Sequential(
                nn.Linear(input_size, self.cfg.linear_vocab_dim_1),
                nn.Dropout(self.cfg.dropout),
                self.cfg.activation(),
                nn.Linear(self.cfg.linear_vocab_dim_1, self.cfg.linear_vocab_dim),
                nn.Dropout(self.cfg.dropout),
                self.cfg.activation()
            )
            self.linear_vocab_final = nn.Sequential(
                nn.Linear(self.cfg.linear_vocab_dim*self.cfg.max_length, self.cfg.linear_final_dim),
                # nn.BatchNorm1d(self.cfg.linear_final_dim),
                self.cfg.activation(),
                nn.Dropout(self.cfg.dropout)
            )
        if self.cfg.self_attention_enable:
            hidden_size += self.cfg.linear_final_dim
            if self.cfg.cnn_model_name == "SimpleConv2D":
                self.convnet = nn.Sequential(
                    nn.Conv2d(self.bert.config.num_hidden_layers * self.bert.config.num_attention_heads,
                              self.cfg.conv2d_hidden_channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.ReLU(),
                    nn.Conv2d(self.cfg.conv2d_hidden_channel,
                              1, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.ReLU(),
                    Lambda(lambda x: x.view(x.size(0), -1)),
                )
                self.convnet.num_features = self.cfg.max_length ** 2
            else:
                self.convnet = timm.create_model(self.cfg.cnn_model_name,
                                                 pretrained=self.cfg.cnn_pretrained,
                                                 num_classes=0)
                if "swin" in self.cfg.cnn_model_name:
                    self.convnet.patch_embed.proj = nn.Conv2d(
                        self.bert.config.num_hidden_layers * self.bert.config.num_attention_heads,
                        96, kernel_size=(4, 4), stride=(4, 4)
                    )
                if "vit" in self.cfg.cnn_model_name:
                    self.convnet.patch_embed.proj = nn.Conv2d(
                        self.bert.config.num_hidden_layers * self.bert.config.num_attention_heads,
                        768, kernel_size=(32, 32), stride=(32, 32)
                    )
                if "efficientnet" in self.cfg.cnn_model_name:
                    self.convnet.conv_stem = nn.Conv2d(
                        self.bert.config.num_hidden_layers * self.bert.config.num_attention_heads,
                        32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                if "resnet" in self.cfg.cnn_model_name:
                    self.convnet.conv1 = nn.Conv2d(
                        self.bert.config.num_hidden_layers * self.bert.config.num_attention_heads,
                        64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            self.linear_conv_final = nn.Sequential(
                nn.Linear(self.convnet.num_features, self.cfg.linear_final_dim),
                # nn.BatchNorm1d(self.cfg.linear_final_dim),
                self.cfg.activation(),
                nn.Dropout(self.cfg.dropout)
            )
        if self.cfg.hidden_stack_enable:
            hidden_size += self.cfg.linear_final_dim
            self.linear_hidden_final = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, self.cfg.linear_final_dim),
                # nn.BatchNorm1d(self.cfg.linear_final_dim),
                self.cfg.activation(),
                nn.Dropout(self.cfg.dropout)
            )
        if self.cfg.attention_head_enable:
            hidden_size += self.cfg.linear_final_dim
            self.attention_head = nn.Sequential(
                AttentionHead(self.bert.config.hidden_size, self.bert.config.hidden_size, 1),
            )
            self.linear_attention_head_final = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, self.cfg.linear_final_dim),
                # nn.BatchNorm1d(self.cfg.linear_final_dim),
                self.cfg.activation(),
                nn.Dropout(self.cfg.dropout)
            )
        if self.cfg.rnn_module_num > 0:
            hidden_size += self.cfg.linear_final_dim
            self.lstm = self.make_lstm_module()
            if self.cfg.bidirectional:
                if self.cfg.word_axis:
                    lstm_size = int(self.cfg.max_length * len(self.cfg.rnn_hidden_indice) * (
                                (2 * self.cfg.rnn_module_shrink_ratio) ** self.cfg.rnn_module_num))
                else:
                    lstm_size = int(self.bert.config.hidden_size * len(self.cfg.rnn_hidden_indice) * (
                                (2 * self.cfg.rnn_module_shrink_ratio) ** self.cfg.rnn_module_num))
            else:
                if self.cfg.word_axis:
                    lstm_size = int(self.cfg.max_length * len(self.cfg.rnn_hidden_indice) * (
                                self.cfg.rnn_module_shrink_ratio ** self.cfg.rnn_module_num))

                else:
                    lstm_size = int(self.bert.config.hidden_size * len(self.cfg.rnn_hidden_indice) * (
                                self.cfg.rnn_module_shrink_ratio ** self.cfg.rnn_module_num))

            self.linear_lstm_final = nn.Sequential(
                nn.Linear(lstm_size, self.cfg.linear_final_dim),
                # nn.BatchNorm1d(self.cfg.linear_final_dim),
                self.cfg.activation(),
                nn.Dropout(self.cfg.dropout)
            )
        if self.cfg.tcn_module_enable:
            hidden_size += self.cfg.linear_final_dim
            if self.cfg.word_axis:
                self.tcn = self.cfg.tcn_module(num_inputs=self.cfg.max_length,
                                               num_channels=[self.cfg.max_length] * cfg.tcn_module_num,
                                               kernel_size=self.cfg.tcn_module_kernel_size,
                                               dropout=self.cfg.tcn_module_dropout)
            else:
                self.tcn = self.cfg.tcn_module(num_inputs=self.bert.config.hidden_size,
                                               num_channels=[self.bert.config.hidden_size] * cfg.tcn_module_num,
                                               kernel_size=self.cfg.tcn_module_kernel_size,
                                               dropout=self.cfg.tcn_module_dropout)
            self.linear_tcn_final = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, self.cfg.linear_final_dim),
                # nn.BatchNorm1d(self.cfg.linear_final_dim),
                self.cfg.activation(),
                nn.Dropout(self.cfg.dropout)
            )
        if self.cfg.pooler_enable:
            hidden_size += self.bert.config.hidden_size
        if self.cfg.attention_pool_enable:
            hidden_size += self.cfg.linear_final_dim
            self.linear_attention_pool_final = nn.Sequential(
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.max_length ** 2, self.cfg.linear_final_dim),
                # nn.BatchNorm1d(self.cfg.linear_final_dim),
                self.cfg.activation(),
                nn.Dropout(self.cfg.dropout)
            )
        if self.cfg.feature_enable:
            hidden_size += self.cfg.linear_final_dim
            self.linear_feature = nn.Sequential(
                nn.Linear(17, self.cfg.linear_final_dim),
                self.cfg.activation(),
                nn.Dropout(self.cfg.dropout)
            )
        if not self.cfg.simple_structure:
            if self.cfg.prep_enable:
                self.linear_perp = nn.Sequential(
                    nn.Linear(1, self.cfg.linear_perplexity_dim),
                    # nn.BatchNorm1d(self.cfg.perplexity_linear_dim),
                    nn.Dropout(self.cfg.dropout),
                    self.cfg.activation()
                )
                self.linear1 = nn.Sequential(
                    nn.Linear(hidden_size + self.cfg.linear_perplexity_dim, self.cfg.linear_dim),
                    nn.Dropout(self.cfg.dropout),
                    self.cfg.activation()
                )
                self.linear2 = nn.Sequential(
                    nn.Linear(self.cfg.linear_dim + self.cfg.linear_perplexity_dim, 1)
                )
                self.linear1_std = nn.Sequential(
                    nn.Linear(hidden_size + self.cfg.linear_perplexity_dim, self.cfg.linear_dim),
                    nn.Dropout(self.cfg.dropout),
                    self.cfg.activation()
                )
                self.linear2_std = nn.Sequential(
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
                self.linear1_std = nn.Sequential(
                    nn.Linear(hidden_size, self.cfg.linear_dim),
                    nn.Dropout(self.cfg.dropout),
                    self.cfg.activation()
                )
                self.linear2_std = nn.Sequential(
                    nn.Linear(self.cfg.linear_dim, 1)
                )
        else:
            self.linear_simple = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(self.cfg.dropout_output_hidden)
        self.dropout_attn = nn.Dropout(self.cfg.dropout_attn)

        self.df_train: pd.DataFrame
        self.df_val: pd.DataFrame
        self.dataset_train: Dataset
        self.dataset_val: Dataset
        self.feature_columns: list

        self.best_rmse = np.inf
        self.reinit_bert()

    def reinit_bert(self):
        def get_model_type(x):
            if "distilbert" in x: return "distilbert"
            if "albert" in x: return "albert"
            if "roberta" in x: return "roberta"
            if "bert" in x: return "bert"

        # re-init pooler
        if self.cfg.reinit_pooler and not self.cfg.prep_enable:
            if "bert" in self.cfg.nlp_model_name or "roberta" in self.cfg.nlp_model_name or "luke" in self.cfg.nlp_model_name:
                self.bert.pooler.dense.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
                self.bert.pooler.dense.bias.data.zero_()
                for p in self.bert.pooler.parameters():
                    p.requires_grad = True
            elif "xlnet" in self.cfg.nlp_model_name:
                raise ValueError(f"{self.cfg.nlp_model_name} does not have a pooler at the end")
            else:
                raise NotImplementedError

        # re-init layers
        if self.cfg.reinit_layers > 0:
            if "albert" in self.cfg.nlp_model_name:
                raise ValueError("albertはreinitしちゃだめ")

            elif "bert" in self.cfg.nlp_model_name or "roberta" in self.cfg.nlp_model_name or "ernie" in self.cfg.nlp_model_name:
                if self.cfg.prep_enable:
                    if get_model_type(self.cfg.nlp_model_name) == "bert":
                        layers = self.bert.bert.encoder.layer[-self.cfg.reinit_layers:]
                    elif get_model_type(self.cfg.nlp_model_name) == "roberta":
                        layers = self.bert.roberta.encoder.layer[-self.cfg.reinit_layers:]
                else:
                    if get_model_type(self.cfg.nlp_model_name) == "distilbert":
                        layers = self.bert.transformer.layer[-self.cfg.reinit_layers:]
                    else:
                        layers = self.bert.encoder.layer[-self.cfg.reinit_layers:]
                for layer in layers:
                    for module in layer.modules():
                        if isinstance(module, nn.Linear):
                            # Slightly different from the TF version which uses truncated_normal for initialization
                            # cf https://github.com/pytorch/pytorch/pull/5617
                            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
                            if module.bias is not None:
                                module.bias.data.zero_()
                        elif isinstance(module, nn.Embedding):
                            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
                            if module.padding_idx is not None:
                                module.weight.data[module.padding_idx].zero_()
                        elif isinstance(module, nn.LayerNorm):
                            module.bias.data.zero_()
                            module.weight.data.fill_(1.0)

            elif "xlnet" in self.cfg.nlp_model_name:
                for layer in self.bert.layer[-self.cfg.reinit_layers:]:
                    for module in layer.modules():
                        self.bert._init_weights(module)

            elif "luke" in self.cfg.nlp_model_name:
                if self.cfg.prep_enable:
                    raise NotImplementedError
                else:
                    layers = self.bert.encoder.layer[-self.cfg.reinit_layers:]
                for layer in layers:
                    for module in layer.modules():
                        if isinstance(module, nn.Linear):
                            # Slightly different from the TF version which uses truncated_normal for initialization
                            # cf https://github.com/pytorch/pytorch/pull/5617
                            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
                            if module.bias is not None:
                                module.bias.data.zero_()
                        elif isinstance(module, nn.Embedding):
                            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
                            if module.padding_idx is not None:
                                module.weight.data[module.padding_idx].zero_()
                        elif isinstance(module, nn.LayerNorm):
                            module.bias.data.zero_()
                            module.weight.data.fill_(1.0)
            elif "funnel" in self.cfg.nlp_model_name:
                for layer in self.bert.encoder.blocks[2][-self.cfg.reinit_layers:]:
                    for module in layer.modules():
                        self.bert._init_weights(module)
            elif "bart" in self.cfg.nlp_model_name:
                for layer in self.bert.decoder.layers[-self.cfg.reinit_layers:]:
                    for module in layer.modules():
                        self.bert._init_weights(module)
                if not self.cfg.decoder_only:
                    for layer in self.bert.encoder.layers[-self.cfg.reinit_layers:]:
                        for module in layer.modules():
                            self.bert._init_weights(module)
            elif "electra" in self.cfg.nlp_model_name:
                for layer in self.bert.encoder.layer[-self.cfg.reinit_layers:]:
                    for module in layer.modules():
                        self.bert._init_weights(module)
            elif "gpt2" in self.cfg.nlp_model_name:
                for layer in self.bert.h[-self.cfg.reinit_layers:]:
                    for module in layer.modules():
                        self.bert._init_weights(module)

            elif "gpt-neo" in self.cfg.nlp_model_name:
                for layer in self.bert.h[-self.cfg.reinit_layers:]:
                    for module in layer.modules():
                        self.bert._init_weights(module)

            elif "t5" in self.cfg.nlp_model_name:
                for layer in self.bert.encoder.block[-self.cfg.reinit_layers:]:
                    for module in layer.modules():
                        self.bert._init_weights(module)
                if not self.cfg.decoder_only:
                    for layer in self.bert.decoder.block[-self.cfg.reinit_layers:]:
                        for module in layer.modules():
                            self.bert._init_weights(module)
            elif "mpnet" in self.cfg.nlp_model_name:
                for layer in self.bert.encoder.layer[-self.cfg.reinit_layers:]:
                    for module in layer.modules():
                        self.bert._init_weights(module)
        """
        for layer in [self.linear1, self.linear2, self.linear1_std, self.linear2_std, self.linear_perp, self.linear_vocab,
                      self.linear_tcn_final, self.linear_lstm_final, self.linear_hidden_final,
                      self.linear_conv_final, self.linear_vocab_final]:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
        """

    def make_lstm_module(self):
        ret = []
        if self.cfg.word_axis:
            hidden_size = self.cfg.max_length * len(self.cfg.rnn_hidden_indice)
        else:
            hidden_size = self.bert.config.hidden_size * len(self.cfg.rnn_hidden_indice)

        for i in range(self.cfg.rnn_module_num):
            ret.append((f"lstm_module_{i}", LSTMModule(cfg=self.cfg, hidden_size=hidden_size)))
            if self.cfg.bidirectional:
                hidden_size = int(hidden_size * self.cfg.rnn_module_shrink_ratio * 2)
            else:
                hidden_size = int(hidden_size * self.cfg.rnn_module_shrink_ratio)
        return nn.Sequential(OrderedDict(ret))

    def forward(self, input_ids_masked, attention_mask, token_type_ids, input_ids, features):
        def f(x_in, perplexity=None):
            x_in = F.dropout(x_in, p=self.cfg.multi_dropout_ratio, training=True)
            if perplexity is not None:
                x_out_mean = self.linear1(torch.cat([x_in, perplexity], dim=1))
                x_out_mean = self.linear2(torch.cat([x_out_mean, perplexity], dim=1))
            else:
                x_out_mean = self.linear1(x_in)
                x_out_mean = self.linear2(x_out_mean)
            return x_out_mean

        def g(x_in, perplexity=None):
            x_in = F.dropout(x_in, p=self.cfg.multi_dropout_ratio, training=True)
            if perplexity is not None:
                x_out_std = self.linear1_std(torch.cat([x_in, perplexity], dim=1))
                x_out_std = self.linear2_std(torch.cat([x_out_std, perplexity], dim=1))
            else:
                x_out_std = self.linear1(x_in)
                x_out_std = self.linear2(x_out_std)
            x_out_std = torch.exp(x_out_std) ** 0.5
            return x_out_std

        if not self.cfg.prep_enable:
            x = self.bert(input_ids=input_ids_masked,
                          attention_mask=attention_mask,
                          output_attentions=True,
                          output_hidden_states=True)
            if "deberta" in self.cfg.nlp_model_name:
                x = [x[0], x[1], x[2]]
            elif "xlnet" in self.cfg.nlp_model_name:
                if len(x) == 4:
                    x = [x[0], x[2], x[3], x[1]]
                else:
                    x = [x[0], x[1], x[2]]
            elif "albert" in self.cfg.nlp_model_name:
                x = [x[0], x[1]]
            elif "distilbert" in self.cfg.nlp_model_name:
                x = [x[0], x[1], x[2]]
            elif "funnel" in self.cfg.nlp_model_name:
                x = [x[0], x[1], x[2]]
            elif "bart" in self.cfg.nlp_model_name:
                x = [x[0], x[2], x[3], None]  # x[2]: decoder hidden_states, x[3]: cross attention
            elif "electra" in self.cfg.nlp_model_name:
                x = [x[0], x[1], x[2]]
            elif "t5" in self.cfg.nlp_model_name:
                x = [x[0], x[1], x[2]]
            elif "mpnet" in self.cfg.nlp_model_name:
                x = [x[0], x[1], x[2]]
            else:
                x = [x[0], x[2], x[3], x[1]]
        elif "funnel" in self.cfg.nlp_model_name:
            x = self.bert.funnel(input_ids=input_ids_masked,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 output_attentions=True,
                                 output_hidden_states=True)
            if self.cfg.prep_enable:
                input_ids_pred = self.bert.lm_head(x[0])
        elif "albert" in self.cfg.nlp_model_name:
            x = self.bert.albert(input_ids=input_ids_masked,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 output_attentions=True,
                                 output_hidden_states=True)
            if self.cfg.prep_enable:
                input_ids_pred = self.bert.predictions(x[0])
        elif "deberta" in self.cfg.nlp_model_name:
            x = self.bert.deberta(input_ids=input_ids_masked,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  output_attentions=True,
                                  output_hidden_states=True)
            if self.cfg.prep_enable:
                input_ids_pred = self.bert.cls(x[0])
        elif "roberta" in self.cfg.nlp_model_name and "bigbird" not in self.cfg.nlp_model_name:
            x = self.bert.roberta(input_ids=input_ids_masked,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  output_attentions=True,
                                  output_hidden_states=True)
            if self.cfg.prep_enable:
                input_ids_pred = self.bert.lm_head(x[0])

        elif "bert" in self.cfg.nlp_model_name or "bigbird" in self.cfg.nlp_model_name:
            x = self.bert.bert(input_ids=input_ids_masked,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               output_attentions=True,
                               output_hidden_states=True)
            if self.cfg.prep_enable:
                input_ids_pred = self.bert.cls(x[0])
        else:
            x = self.bert(input_ids=input_ids_masked,
                          attention_mask=attention_mask,
                          output_attentions=True,
                          output_hidden_states=True)
            if "luke-base" in self.cfg.nlp_model_name or "luke-large" in self.cfg.nlp_model_name:
                x = [x[0], x[2], x[3]]

        # x[0]: last hidden layer, x[1]: all hidden layer, x[2]: attention matrix
        if self.cfg.prep_enable:
            loss = torch.nn.functional.cross_entropy(input_ids_pred.view(-1, self.bert.config.vocab_size), input_ids.view(-1), reduction="none")
            perplexity = loss.view(len(input_ids), -1) * attention_mask
            perplexity = perplexity.sum(dim=1) / attention_mask.sum(dim=1)
            perplexity = perplexity.view(-1, 1)

        # base feature
        x_bert = []
        if self.cfg.pooler_enable:
            x_bert.append(x[3])
        if self.cfg.linear_vocab_enable:
            xx = self.dropout(self.linear_vocab(x[0]).view(len(input_ids), -1))
            x_bert.append(self.linear_vocab_final(xx))
        if self.cfg.self_attention_enable:
            xx = torch.cat([self.dropout_attn(xx) for xx in x[2]], dim=1)
            xx = self.convnet(xx)
            xx = self.linear_conv_final(xx)
            x_bert.append(xx)
        if self.cfg.hidden_stack_enable:
            if "albert" in self.cfg.nlp_model_name:
                xx = self.linear_hidden_final(x[0].mean(dim=1))
                x_bert.append(xx)
            else:
                if "funnel" in self.cfg.nlp_model_name:
                    xx = torch.stack([self.dropout_bert_stack(xx) for xx in x[1][-3:]]).mean(dim=[0, 2])
                else:
                    xx = torch.stack([self.dropout_bert_stack(xx) for xx in x[1][-4:]]).mean(dim=0)
                    xx = torch.sum(
                        xx * attention_mask.unsqueeze(-1), dim=1, keepdim=False
                    )
                    xx = xx / torch.sum(attention_mask, dim=-1, keepdim=True)
                xx = self.linear_hidden_final(xx)
                x_bert.append(xx)

        # residual feature
        if self.cfg.rnn_module_num > 0:
            if self.cfg.word_axis:
                x_lstm = self.lstm(torch.cat([x[1][idx] for idx in self.cfg.rnn_hidden_indice], dim=1).transpose(2, 1)).mean(dim=1)
            else:
                x_lstm = self.lstm(torch.cat([x[1][idx] for idx in self.cfg.rnn_hidden_indice], dim=2)).mean(dim=1)
            x_lstm = self.linear_lstm_final(x_lstm)
            x_bert.append(x_lstm)
        if self.cfg.attention_head_enable:
            weights_attn = self.attention_head(x[0])
            x_attn = torch.sum(weights_attn * x[0], dim=1)
            x_attn = self.linear_attention_head_final(x_attn)
            x_bert.append(x_attn)
        if self.cfg.tcn_module_enable:
            if self.cfg.word_axis:
                x_tcn = self.tcn(self.dropout(x[0])).mean(dim=1)
            else:
                x_tcn = self.tcn(self.dropout(x[0]).permute(0, 2, 1)).mean(dim=2)
            x_tcn = self.linear_tcn_final(x_tcn)
            x_bert.append(x_tcn)
        if self.cfg.attention_pool_enable:
            xx = torch.cat([xx for xx in x[2]], dim=1).mean(dim=1).reshape(len(input_ids), -1)
            xx = self.linear_attention_pool_final(xx)
            x_bert.append(xx)
        if self.cfg.feature_enable:
            xx = self.linear_feature(features)
            x_bert.append(xx)

        x_bert = torch.cat(x_bert, dim=1)

        if self.cfg.prep_enable:
            perplexity = self.linear_perp(perplexity)
            x_out_mean = torch.stack([f(x_bert, perplexity) for _ in range(self.cfg.multi_dropout_num)]).mean(dim=0)
            x_out_std = torch.stack([g(x_bert, perplexity) for _ in range(self.cfg.multi_dropout_num)]).mean(dim=0)
        elif self.cfg.simple_structure:
            x_out_mean = self.linear_simple(x_bert)
            x_out_std = None
        else:
            x_out_mean = torch.stack([f(x_bert) for _ in range(self.cfg.multi_dropout_num)]).mean(dim=0)
            x_out_std = torch.stack([g(x_bert) for _ in range(self.cfg.multi_dropout_num)]).mean(dim=0)
        return x_out_mean, x_out_std

    def training_step(self, batch, batch_idx):
        scheduler = self.lr_schedulers()
        scheduler.step()

        input_ids_masked, attention_mask, token_type_ids, input_ids, features, target, std = batch
        output_mean, output_std = self.forward(input_ids_masked, attention_mask, token_type_ids, input_ids, features)

        if self.cfg.kl_div_enable:
            dist_pred = torch.distributions.Normal(output_mean.flatten(), output_std.flatten())
            dist_target = torch.distributions.Normal(target, std)
            loss = torch.distributions.kl_divergence(dist_pred, dist_target).mean()
        else:
            if self.cfg.crossentropy:
                target = (target - self.cfg.crossentropy_min) / (self.cfg.crossentropy_max - self.cfg.crossentropy_min)
                loss = F.binary_cross_entropy_with_logits(output_mean.flatten(), target.flatten())
            else:
                loss = F.mse_loss(output_mean.flatten(), target.flatten())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids_masked, attention_mask, token_type_ids, input_ids, features, target, std = batch
        output_mean, output_std = self.forward(input_ids_masked, attention_mask, token_type_ids, input_ids, features)

        if self.cfg.kl_div_enable:
            dist_pred = torch.distributions.Normal(output_mean.flatten(), output_std.flatten())
            dist_target = torch.distributions.Normal(target, std)
            loss = torch.distributions.kl_divergence(dist_pred, dist_target).mean()
        else:
            if self.cfg.crossentropy:
                target = (target - self.cfg.crossentropy_min) / (self.cfg.crossentropy_max - self.cfg.crossentropy_min)
                loss = F.binary_cross_entropy_with_logits(output_mean.flatten(), target.flatten())
            else:
                loss = F.mse_loss(output_mean.flatten(), target.flatten())
        self.log('val_loss', loss, prog_bar=True)
        return output_mean.cpu().detach().numpy().flatten(), target.cpu().detach().numpy().flatten()

    def validation_epoch_end(self, val_step_outputs):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        pred = []
        target = []
        for output in val_step_outputs:
            pred.extend(output[0].tolist())
            target.extend(output[1].tolist())

        pred = np.array(pred)
        target = np.array(target)

        if self.cfg.crossentropy:
            pred = (self.cfg.crossentropy_max - self.cfg.crossentropy_min) * sigmoid(np.array(pred)) + self.cfg.crossentropy_min
            target = (self.cfg.crossentropy_max - self.cfg.crossentropy_min) * np.array(target) + self.cfg.crossentropy_min

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
        def feature_engineering(df):
            def total_words(x):
                return len(x.split(" "))

            def total_unique_words(x):
                return len(np.unique(x.split(" ")))

            def total_charactors(x):
                x = x.replace(" ", "")
                return len(x)

            def total_sentence(x):
                x = x.replace("!", "[end]").replace("?", "[end]").replace(".", "[end]")
                return len(x.split("[end]"))

            df_ret = df[["id", "excerpt", "target", "standard_error", "kfold"]]
            excerpt = df["excerpt"].values
            df_ret["total_words"] = [total_words(x) for x in excerpt]
            df_ret["total_unique_words"] = [total_unique_words(x) for x in excerpt]
            df_ret["total_characters"] = [total_charactors(x) for x in excerpt]
            df_ret["total_sentence"] = [total_sentence(x) for x in excerpt]

            df_ret["div_sentence_characters"] = df_ret["total_sentence"] / df_ret["total_characters"]
            df_ret["div_sentence_words"] = df_ret["total_sentence"] / df_ret["total_words"]
            df_ret["div_characters_words"] = df_ret["total_characters"] / df_ret["total_words"]
            df_ret["div_words_unique_words"] = df_ret["total_words"] / df_ret["total_unique_words"]

            for i, word in enumerate(["!", "?", "(", ")", "'", '"', ";", ".", ","]):
                df_ret[f"count_word_special_{i}"] = [x.count(word) for x in excerpt]

            return df_ret.fillna(0)

        df = pd.read_csv("input/commonlitreadabilityprize/train_folds.csv")
        if self.cfg.debug:
            df = df.iloc[::30]

        df = feature_engineering(df)
        self.cfg.feature_columns = [x for x in df.columns if x not in ["id", "excerpt", "target", "kfold", "standard_error"]]
        self.cfg.feature_mean = df[self.cfg.feature_columns].mean().values
        self.cfg.feature_std = df[self.cfg.feature_columns].std().values
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
            no_decay_ary = ["bias", "LayerNorm.weight", "pooler"]

            if no_decay:
                ret["params"] = [p for n, p in named_parameters if not any(nd in n for nd in no_decay_ary)]
                ret["weight_decay"] = 0
            else:
                ret["params"] = [p for n, p in named_parameters if any(nd in n for nd in no_decay_ary) and "pooler" not in n]
                ret["weight_decay"] = weight_decay
            ret["lr"] = lr
            return ret

        params = []
        if self.cfg.prep_enable:
            if "funnel" in self.cfg.nlp_model_name:
                params.append({"params": self.bert.lm_head.parameters(), "weight_decay": self.cfg.weight_decay, "lr": self.cfg.lr_bert})
            elif "albert" in self.cfg.nlp_model_name:
                params.append({"params": self.bert.predictions.parameters(), "weight_decay": self.cfg.weight_decay, "lr": self.cfg.lr_bert})
            elif "deberta" in self.cfg.nlp_model_name:
                params.append({"params": self.bert.cls.parameters(), "weight_decay": self.cfg.weight_decay, "lr": self.cfg.lr_bert})
            elif "roberta" in self.cfg.nlp_model_name and "bigbird" not in self.cfg.nlp_model_name:
                params.append({"params": self.bert.lm_head.parameters(), "weight_decay": self.cfg.weight_decay, "lr": self.cfg.lr_bert})
            elif "bert" in self.cfg.nlp_model_name or "bigbird" in self.cfg.nlp_model_name:
                params.append({"params": self.bert.cls.parameters(), "weight_decay": self.cfg.weight_decay, "lr": self.cfg.lr_bert})
            else:
                raise ValueError("mask用のparameterありません")
        params.append(extract_params(self.bert.named_parameters(), lr=self.cfg.lr_bert, weight_decay=self.cfg.weight_decay, no_decay=False))
        params.append(extract_params(self.bert.named_parameters(), lr=self.cfg.lr_bert, weight_decay=0, no_decay=True))

        if self.cfg.linear_vocab_enable:
            params.append(extract_params(self.linear_vocab.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear_vocab.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
            params.append(extract_params(self.linear_vocab_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear_vocab_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
        if self.cfg.self_attention_enable:
            params.append(extract_params(self.linear_conv_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear_conv_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
            params.append(extract_params(self.convnet.named_parameters(), lr=self.cfg.lr_cnn, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.convnet.named_parameters(), lr=self.cfg.lr_cnn, weight_decay=0, no_decay=True))
        if self.cfg.attention_pool_enable:
            params.append(extract_params(self.linear_attention_pool_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear_attention_pool_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
        if self.cfg.tcn_module_enable:
            params.append(extract_params(self.linear_tcn_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear_tcn_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
            params.append(extract_params(self.tcn.named_parameters(), lr=self.cfg.lr_tcn, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.tcn.named_parameters(), lr=self.cfg.lr_tcn, weight_decay=0, no_decay=True))
        if self.cfg.rnn_module_num > 0:
            params.append(extract_params(self.linear_lstm_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear_lstm_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
            params.append(extract_params(self.lstm.named_parameters(), lr=self.cfg.lr_rnn, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.lstm.named_parameters(), lr=self.cfg.lr_rnn, weight_decay=0, no_decay=True))
        if self.cfg.hidden_stack_enable:
            params.append(extract_params(self.linear_hidden_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear_hidden_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
        if self.cfg.simple_structure:
            params.append(extract_params(self.linear_simple.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear_simple.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
        else:
            params.append(extract_params(self.linear1.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear1.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
            params.append(extract_params(self.linear2.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear2.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
            params.append(extract_params(self.linear1_std.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear1_std.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
            params.append(extract_params(self.linear2_std.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear2_std.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
        if self.cfg.attention_head_enable:
            params.append(extract_params(self.attention_head.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.attention_head.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
            params.append(extract_params(self.linear_attention_head_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear_attention_head_final.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
        if self.cfg.pooler_enable:
            params.append(extract_params(self.bert.pooler.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.bert.pooler.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))
        if self.cfg.feature_enable:
            params.append(extract_params(self.linear_feature.named_parameters(), lr=self.cfg.lr_fc, weight_decay=self.cfg.weight_decay, no_decay=False))
            params.append(extract_params(self.linear_feature.named_parameters(), lr=self.cfg.lr_fc, weight_decay=0, no_decay=True))

        optimizer = self.cfg.optimizer(params)
        num_warmup_steps = int(self.cfg.epochs_max * len(self.df_train) / self.cfg.batch_size * self.cfg.warmup_ratio)
        num_training_steps = int(self.cfg.epochs_max * len(self.df_train) / self.cfg.batch_size) * self.cfg.training_steps_ratio

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.cfg.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.cfg.batch_size)

class CustomModelCheckpoint(ModelCheckpoint):
    def check_monitor_top_k(self, trainer: 'pl.Trainer', current: Optional[torch.Tensor] = None) -> bool:
        if current is None:
            return False

        if current.cpu().numpy() > 0.55:
            should_update_best_and_save = False
        else:
            if self.save_top_k == -1:
                return True
            less_than_k_models = len(self.best_k_models) < self.save_top_k
            if less_than_k_models:
                return True

            if not isinstance(current, torch.Tensor):
                rank_zero_warn(
                    f"{current} is supposed to be a `torch.Tensor`. Saving checkpoint may not work correctly."
                    f" HINT: check the value of {self.monitor} in your validation loop",
                    RuntimeWarning,
                )
                current = torch.tensor(current)

            monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
            should_update_best_and_save = monitor_op(current, self.best_k_models[self.kth_best_model_path])

            # If using multiple devices, make sure all processes are unanimous on the decision.
            should_update_best_and_save = trainer.training_type_plugin.reduce_boolean_decision(should_update_best_and_save)

        return should_update_best_and_save

def main(cfg_original: Config,
         folds: List):
    output_dir = f"output/{os.path.basename(__file__)[:-3]}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir)
    rmse = 0
    with mlflow.start_run() as run:
        mlflow.pytorch.autolog(log_models=False)
        for key, value in cfg_original.__dict__.items():
            mlflow.log_param(key, value)
        for fold in folds:
            try:
                cfg = copy.copy(cfg_original)
                if "deberta" in cfg.nlp_model_name:
                    cfg.pooler_enable = False
                    cfg.reinit_pooler = False
                    cfg.hidden_stack_enable = True
                cfg.fold = fold
                checkpoint_callback = CustomModelCheckpoint(
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
                                  auto_scale_batch_size=True,
                                  # amp_level="02",
                                  max_epochs=cfg.epochs,
                                  benchmark=True,
                                  val_check_interval=cfg.val_check_interval,
                                  progress_bar_refresh_rate=1,
                                  default_root_dir=output_dir,
                                  gradient_clip_val=cfg.gradient_clipping,
                                  accumulate_grad_batches=cfg.accumulate_grad_batches,
                                  stochastic_weight_avg=cfg.stochastic_weight_avg,
                                  callbacks=[checkpoint_callback])

                trainer.fit(model)
                with open(f"{output_dir}/cfg.pickle", "wb") as f:
                    pickle.dump(cfg, f)

                # 足切り1
                if fold == 0 and model.best_rmse > 0.485:
                    break
                rmse += model.best_rmse

                if cfg.nlp_model_name in ["roberta-large",
                                          "studio-ousia/luke-large",
                                          "google/electra-large-discriminator"]:
                    if fold == 0 and rmse > 0.465:
                        break
                    if fold == 1 and rmse / 2 > 0.463:
                        break
                    if fold == 2 and rmse / 3 > 0.4775:
                        break

                if cfg.nlp_model_name in ["funnel-transformer/large",
                                          "funnel-transformer/large-base"]:
                    if fold == 0 and rmse > 0.47:
                        break
                if cfg.nlp_model_name in ["microsoft/deberta-large"]:
                    if fold == 0 and rmse > 0.475:
                        break
                    if fold == 1 and rmse / 2 > 0.4675:
                        break
                    if fold == 2 and rmse / 3 > 0.484:
                        break
                if cfg.nlp_model_name in ["nghuyong/ernie-2.0-large-en"]:
                    if fold == 0 and rmse > 0.458:
                        break
                    if fold == 1 and rmse / 2 > 0.457:
                        break

                del trainer, model, checkpoint_callback
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)

        mlflow.log_metric("rmse_mean", rmse / len(folds))


def config_large(cfg: Config, nlp_model_name: str):
    cfg.nlp_model_name = nlp_model_name
    cfg.lr_bert = 2e-5
    cfg.lr_bert_decay = 1
    cfg.batch_size = 8
    cfg.weight_decay = 0.01
    return cfg


if __name__ == "__main__":
    experiment_name = "bart tune"
    folds = [0, 1, 2, 3, 4]

    def common_config(cfg) -> Config:
        cfg.pooler_enable = False
        cfg.reinit_pooler = False
        cfg.hidden_stack_enable = True
        cfg.self_attention_enable = False
        cfg.feature_enable = True
        cfg.tcn_module_enable = False
        cfg.linear_vocab_enable = True
        cfg.seed = 19920224
        cfg.rnn_module_num = 1
        cfg.simple_structure = False
        cfg.batch_size = 12
        cfg.epochs = 4
        cfg.epochs_max = 4
        cfg.warmup_ratio = 0.05
        if cfg.nlp_model_name == "gpt2-medium":
            cfg.reinit_layers = 6
        if cfg.nlp_model_name == "gpt2":
            cfg.batch_size = 32
        if cfg.nlp_model_name == "funnel-transformer/large-base":
            cfg.rnn_hidden_indice = (-1, -2)
        if cfg.nlp_model_name == "funnel-transformer/large":
            cfg.epochs = 6
            cfg.epochs_max = 6
        if cfg.nlp_model_name == "t5-large":
            cfg.tcn_module_enable = False
            cfg.linear_vocab_enable = False
            cfg.rnn_module_num = 0
            cfg.batch_size = 18
        if cfg.nlp_model_name == "microsoft/mpnet-base":
            cfg.tcn_module_enable = False
            cfg.linear_vocab_enable = False
            cfg.rnn_module_num = 0
            cfg.batch_size = 32
        if cfg.nlp_model_name == "google/electra-large-discriminator":
            cfg.lr_bert = 4e-5
        if cfg.nlp_model_name == "google/electra-large-discriminator":
            cfg.lr_bert = 4e-5
        if cfg.nlp_model_name == "studio-ousia/luke-large":
            cfg.reinit_layers = 5
            cfg.rnn_hidden_indice = (-1, -2, 0, 1)
        if cfg.nlp_model_name == "funnel-transformer/xlarge-base":
            cfg.epochs = 6
            cfg.epochs_max = 6
            cfg.linear_vocab_enable = False
            cfg.rnn_module_num = 0
            cfg.feature_enable = False
        if cfg.nlp_model_name == "nghuyong/ernie-2.0-large-en":
            cfg.reinit_layers = 2
        return cfg

    for nlp_model_name in ["nghuyong/ernie-2.0-large-en"]:
        for lr_bert in [2e-5, 3e-5, 4e-5]:
            for reinit_layers in [1, 3, 5]:
                cfg = Config(experiment_name=experiment_name)
                cfg.nlp_model_name = nlp_model_name
                cfg = common_config(cfg)
                cfg.reinit_layers = reinit_layers
                cfg.lr_bert = lr_bert
                main(cfg, folds=folds)

        cfg = Config(experiment_name=experiment_name)
        cfg.nlp_model_name = nlp_model_name
        cfg = common_config(cfg)
        cfg.rnn_hidden_indice = (-1, -2, 0, 1)
        main(cfg, folds=folds)
