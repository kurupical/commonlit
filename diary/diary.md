# 2021/5/27
今日から頑張る。<br>
Shopeeは行き当たりばったりな実験も多かったので、計画的に実験をしていきたい。<br>
また、後々楽になる投資はめんどくさがらずに先行投資したい。<br>

今までできたこと
* 実験パイプラインの作成　←　できてる、エライ
近々やるべきことは、

* 5-fold cvを一列で見て平均とれるように
* どのような実験をするかの決定。　(近々頑張る)
* 予測パイプラインの作成　(今週末頑張る)
* 予実を見れるツールの作成。　(今週末頑張る)
  -> mlflowでできんかな？
  
マイルストーンとしては、来週末に1サブしたい。。

今日は、

## exp004
* 5-fold cvを簡単に見れるようにした
* float16
* いろんな条件を振る

* 条件振った結果
  * baseline: 0.523(2fold)
  
  * model
    * bert-base-uncased: 0.584
    * roberta-base: 0.553
  * lr_bert:
    * 1e-5: 0.525
    * 2e-5: 0.519　★
    * 5e-5: 0.557
  * dropout_stack
    * 0: 0.524
    * 0.1: 0.524
    * 0.2: 0.523
    * 0.5: 0.513　★
  * dropout:
    * 0: 0.523
    * 0.1: 0.514　★
    * 0.2: 0.518
    * 0.5: 0.535
  * weight_decay:
    * 0: 0.515
    * 0.01: 0.513　★
    * 0.1: 0.525
  * linear_dim:
    * 64: 0.536
    * 128: 0.523　★
    * 256: 0.525
    * 512: 0.524
  * num_warmup_steps:
    * 16*100: 0.523　★
    * 16*300: 0.543
  
# 2021/5/28
ネタが思いつかないので、exp004のベストスコアをベースにもう1回パラメータ振る！(exp005)
## exp005


# 2021/5/29
便利になることには最初に投資したい。

## exp006
* 予実をplotlyでmlflowに保存する?
* UIで見る!
  * text
  * text_length
* debertaやる

# 2021/5/30
* dropout を有効にして何回も予測させるのはありかもしれない

# 2021/5/31
[TODO] foldを公開ノートブックと合わせる<br> ok
[TODO] Datasetの出力をバラバラにする(stdで) <br> X
[TODO] weight_decay をちゃんとやる<br> X
[TODO] 英単語むずかしさの統計量 OK

## exp008
* LSTM系を1個or2個かますのを試してみたい
-> rnn_module_dropout=0が効きそう(fold0: rmse 0.497)

## EDA(001_baysian_mean_target)
!, ?, なども特徴として効く

## exp009
* !, ?などをちゃんと1つの単語として扱うように

## exp010
* loss: rmse -> mse

## exp011
* weight_decay をちゃんとやる

# 2021/6/1
## exp012
multidropout -> CV -5pt

## exp013
真ん中のデータだけ水増ししてみた

## exp014
lrを変える
-> BEST: 0.505(decay=0.99, bert=3e-5)

## exp015
GPT2をためす

## survey
解法をいろいろ見る -> Jigsow をsurvey.mdに残した

# 2021/6/2
ここでfinetune_mlmを試してみる
https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling

## exp016
finetune_mlm (roberta)


# 2020/6/3
* root mean squared errorだからといって、lossをRMSEにする必要があるのか？→ない。結局mseを最小化する方がいいし、バッチサイズが小さいとRMSEは有効にならんような…
* nn.LinearのactivationにReLU入れてたけど、これ最終層のマイナス方面予測に悪影響与えるんでは。ということで削除…
  

## exp020
分布

## exp021
exp018 + loss=mse
-> exp021 (fold0: 0.482)

## exp022
exp018 - tokenizer

# 2020/6/4

## exp024
exp023 + activation入れない

## exp025
exp023 + 1層dense
epochs = 6: 0.484だが、epochs = 8: 0.478

スケジューラーをいじったらもうちょっといい感じになるんでは？

## exp026
exp023 + 3層dense

## exp027
exp023 + No scheduler

## exp028
exp023 + StepLR(gamma=0.5)

## exp029
exp023 + SAM

# 2021/6/5
今日はGoogle Questの解法をサーベイする


## pretrain: lr2.5e-5シリーズも作ってみる
## exp030:
exp023 + LSTMいれる, BiDirectional

## exp031:
exp023 + pretrained

## exp032:
exp030 + mean(dim=1) を最後に入れてみる -> 却下 (安定しなさすぎ)

## exp033:
exp023 + cleaning text

## exp034:
exp023 + all CLS token

## exp035:
exp023 + pooler-mean-min-max concat

## exp036:
exp023 + all hidden token - mean(exp034-like)

## exp037:
exp023 + accumurate_grad_batch

## exp038:
exp023 + small batchsize

## exp039:
exp023 + epochs = 7(exp023はepoch8だったが、いつのまにか7になっていた。。)

## exp040:
いろんなモデルためす(fine_tuned_model=None)

## exp041:
exp023 + BERT lrの調整をやめる

## exp042
exp040 + tune betas

# 2020/6/6

## readbility
https://arxiv.org/pdf/1907.11779.pdf

## exp043
exp040 + finetuned

## exp040_2
exp040 + warmup_ratio = 0.1

## exp044
robertaはfinetune無しで精度が出たので、こいつをいろいろいじくりまわす

## exp045
いろいろなモデル②(bert-base-uncased, )

## exp046
perplexity を入れる①

## exp047
perplexity　を入れる②(linearで拡張)

# 2020/6/7
## exp048
perplexityで実験振る

## exp049
epochs = 4, get_cosine_schedule_with_warmup

## exp050
exp047 + BN(perplexity)

## exp051
exp047 + lstm復活(cnnでもOK)

## exp052
exp051 + residual structure 
  => CV: 0.481 (tcn_module_kernel=3)

## exp053
exp051 + tcn tuning

## exp054
max_length = 48 (そんな文章情報いらんのでは)

## exp055
48単語以降をmask -> だめ

## exp056
vocab-wise denseをしてみる
→スコア安定していい感じ

# 2020/6/9
## exp057
exp053+exp056の機能マージ

## exp058
attention-cnn!
fold=0だけに絞って、まずはよさげなパラメータのあたり付け

cnn_lr [1e-4, 3e-4]

## exp059
bertのattentionが出力できないか見てみる

-> resnet18 / pretrained=Falseが強いか?(fold0: CV0.483)

## exp060
パラメタチューニング(attentionで)

# 2021/6/10
## exp062
roberta-largeでいろいろ見てみる

## exp063
linear_vocabとattentionを組み合わせたい

# 2021/6/11
## exp064
perpをmeanしない

## exp065
perpを, mask部分0にしてmean + roberta-largeでいろんなモデル作った

# 2021/6/12
## exp067
perpも当てるモデルを作ってみる(roberta-baseでcv0.480出たパラメータを再現して)

## exp068
attentionをあきらめたくない

## exp069
attention は、層ごとに平均とる(16, 12, 256, 256 -> 16, 256, 256)

## exp070
attention　をbertに

## exp071
BERTをガチ目にチューニングすることにしよう

# 2021/6/13
## memo
* https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159245
* attnをグラフとして扱ってみる
  相関係数?
  * https://www.jstage.jst.go.jp/article/neuropsychology/34/3/34_17038/_pdf/-char/ja

## exp074
exp073 + simple 1DCNN-①：last 1 ~ 4

## exp075
exp073 + simple 1D-CNN②：start + last

## exp076
exp073 + concat mean+max pooling

## exp077
exp073 + FIX LSTM(batch first)

## exp078
exp077 + start_end attn

## exp080
attn 28*28 -> いいかもしれない、けど計算時間が長すぎる

## exp081
simple 2d_CNN

# 2021/6/15
## exp085
いろいろなモデル

## exp086
roberta-large

# 2021/6/17
## exp089
deberta tuning

# 2021/6/18
## exp090 
bert-base tune
linear_dim=1024がアンサンブルに一番効いてる
→roberta-base, largeもこれで作ってみる?

単純な"精度"だと、dropoutをすべて0.5にしたのが一番よかった

## exp091
linear_dim=1024
bert-base-uncased, roberta-base, large

# 2021/6/20
## exp096
attention: ゼロ埋め->Resize(256,256)

## exp097
1dconv for stack

## exp098
1dconv*2 for stack (ぼつ)

## exp099
+std

## exp100
exp099 + 最近のモデル

# 2021/6/21
これを読む　→　https://arxiv.org/pdf/2006.05987.pdf

本家コード
https://github.com/asappresearch/revisit-bert-finetuning

記事にできるようにまとめるか。。。

## exp104
randomly initializeをやってみる
hidden_bert only

# 2021/6/22
## exp107
pooler_enable=True

# 2021/6/23
## exp111
CNN/LSTM(word方向に)

## exp112
attention_pool_enable(only)
final_dimをいくつかチューニングし、fold=0であたり付けする

## exp113
conv2D(kernel_size=(1, 1))を試してみる

# 2021/6/26
## exp119: deberta-large + re-initialize
1: 0.492
2: 0.488 (ensembleにはこっちのほうが効く)
3: 0.49
4: 0.485

## exp121: gradient clipping (epoch=0 only)
0(baseilne): 0.494
1: 0.487
2: 0.479
3: 0.494

# 2021/6/27
## exp129
finetune luke-large
* gradient_clipping=0.2
  * init=2 0.483
  * init=3 0.489
  * init=4 0.485
  * init=5 0.485
* gradient_clipping=0.5
  * init=2 0.487
  * init=3 0.487
  * init=4 0.483
  * init=5 0.485

# 2021/6/28

## exp131
* deberta-base でexp060を再現。gradient_clipping.

## exp132
* luke-largeのいろんな種類を試す(luke-largeのベストパラメータ)

## exp133
* deberta-largeでいろいろ

## SUB
60+82+84+91+110+119+125 (CV: 0.4566 / LB: 0.459)
+ exp047: CV 0.4554 / LB: 0.461
+ exp124(bert-base-cased): CV: 0.4549 / LB: 0.46
- exp110/119 CV: 0.4569 / LB: 0.459(微減)

# 2021/7/2
## exp139の結果記録
### reinit_layers + gradient_clipping
[0, 0]: 0.489
[0, 0.2]: 0.488
[1, 0]: 0.482
[1, 0.2]: 0.484
[2, 0]: 0.485
[2, 0.2]: 0.494

### dropout_bert
0: 0.478
0.05: 0.48
0.15: 0.495
0.2: 0.518
0.5: 0.7

ここから dropout_bert = 0, reinit_layers = 1, gradient_clipping = 0.2

batch_size = 16, lr_bert = 3e-5 -> CV: 0.480
batch_size = 16, lr_bert = 5e-5 -> CV: 0.482
batch_size = 32, lr_bert = 3e-5 -> CV: 0.479
batch_size = 32, lr_bert = 5e-5 -> CV: 0.479

batch_size = 8, lr_fc = 1e-5 -> CV: 0.484
batch_size = 8, lr_fc = 1e-4 -> CV: 0.481
batch_size = 8, lr_fc = 1e-3 -> CV: 0.481
batch_size = 16, lr_fc = 1e-5 -> CV: 0.48
batch_size = 16, lr_fc = 1e-4 -> CV: 0.48
batch_size = 16, lr_fc = 1e-3 -> CV: 0.48
batch_size = 32, lr_fc = 1e-5 -> CV: 0.48
batch_size = 32, lr_fc = 1e-4 -> CV: 0.48
batch_size = 32, lr_fc = 1e-3 -> CV: 0.479

(batch_size = 32)
lr_bert_decay = 1, lr_bert = 1e-5 -> CV: 0.494
lr_bert_decay = 1, lr_bert = 3e-5 -> CV: 0.478
lr_bert_decay = 1, lr_bert = 5e-5 -> CV: 0.479
lr_bert_decay = 0.95, lr_bert = 1e-5 -> CV: 0.504
lr_bert_decay = 0.95, lr_bert = 3e-5 -> CV: 0.482
lr_bert_decay = 0.95, lr_bert = 5e-5 -> CV: 0.479
lr_bert_decay = 0.9, lr_bert = 1e-5 -> CV: 0.513
lr_bert_decay = 0.9, lr_bert = 3e-5 -> CV: 0.49

# 2021/7/4

## exp142: hidden_stack
dropout_stack = 0 -> CV: 0.48
dropout_stack = 0.2 -> CV: 0.479
dropout_stack = 0.5 -> CV: 0.483

## exp143: lstm
linear_final_dim = 64 -> CV: 0.481
linear_final_dim = 128 -> CV: 0.479
linear_final_dim = 256 -> CV: 0.479
linear_final_dim = 512 -> CV: 0.476

## exp144: vocab
linear_vocab_dim = 8 -> CV: 0.477
linear_vocab_dim = 16 -> CV: 0.478

linear_final_dim = 32 -> CV: 0.478
linear_final_dim = 128 -> CV: 0.478
linear_final_dim = 256 -> CV: 0.479
linear_final_dim = 512 -> CV: 0.477

## exp149: fine-tune 2
reinit_layers = 1
  * gradient_clipping = 1 CV: 0.478
  * gradient_clipping = 2 CV: 0.482
reinit_layers = 2
  * gradient_clipping = 0.2 CV: 0.482
  * gradient_clipping = 0.5 CV: 0.481
  * gradient_clipping = 1.0 CV: 0.479
reinit_layers = 3
  * gradient_clipping = 0.2 CV: 0.477
  * gradient_clipping = 0.5 CV: 0.476
  * gradient_clipping = 1.0 CV: 0.476
reinit_layers = 4
  * gradient_clipping = 0.2 CV: 0.479
  * gradient_clipping = 0.5 CV: 0.475
  * gradient_clipping = 1.0 CV: 0.475

## exp147: self_attn_pooler
ごみ

## exp150: large batch_size
bs = 64 / lr_bert = 3e-5 CV: 0.481
bs = 64 / lr_bert = 5e-5 CV: 0.482
bs = 128 / lr_bert = 3e-5 CV: 0.502
bs = 128 / lr_bert = 5e-5 CV: 0.503

## exp151: iroiro
crossentropy (-6, 4) CV: 0.479
crossentropy (-5, 3) CV: 0.477
kl_div_enable=True CV: 0.491
dropout
* 0.0 CV: 0.48
* 0.1 CV: 0.478
* 0.2 CV: 0.48
* 0.5 CV: 0.478

## exp152: roberta-large tune(only fold-0)
lr_bert 2e-5
* bs 8: CV: 0.491
* bs 16: CV: 0.478

lr_bert 3e-5
* bs 8: CV: 0.475 
* bs 16: CV: 0.492

lr_bert 5e-5
* bs 8: CV: 0.515
* bs 16: CV: 0.492

## exp153: roberta-large / large batch_size
* lr_bert 1e-5 CV: 0.499
* lr_bert 2e-5 CV: 0.487
* lr_bert 3e-5 CV: 0.488
* lr_bert 4e-5 CV: 0.49
* lr_bert 5e-5 CV: 0.493

## exp154: feature eng(いっぱい)
* CV: 0.490

## exp160: No activation
* CV: 0.480

## exp155: feature eng(17)
* linear_final_dim 32: 0.478
* linear_final_dim 64: 0.475
* linear_final_dim 128: 0.471
* linear_final_dim 256: 0.476
* linear_final_dim 512: 0.478

## exp156: roberta-large (reinit-layers)
* reinit_layers = 3 CV: 0.489
* reinit_layers = 4 CV: 0.485
* reinit_layers = 5 CV: 0.486
* reinit_layers = 6 CV: 0.487
* reinit_layers = 7 CV: 0.488
* reinit_layers = 8 CV: 0.491


## exp156: model-tune
* bert-base-cased
  * reinit_layers = 1 CV: 0.509
  * reinit_layers = 2 CV: 0.502
  * reinit_layers = 3 CV: 0.502
  * reinit_layers = 4 CV: 0.505
  * reinit_layers = 5 CV: 0.511
  * reinit_layers = 6 CV: 0.512
* luke-base
  * reinit_layers = 1 CV: 0.48
  * reinit_layers = 2 CV: 0.479
  * reinit_layers = 3 CV: 0.478
  * reinit_layers = 4 CV: 0.477
  * reinit_layers = 5 CV: 0.48
  * reinit_layers = 6 CV: 0.48
  
  
## exp162: roberta-large tuning
* gradient_clip
  * 0.1 : 0.484
  * 0.2 : 0.483
  * 0.5 : 0.486
  
* lr_fc
  * 1e-4: 0.487
  * 3e-4: 0.487
  * 5e-4: 0.485
  * 1e-5: 0.484
  * 3e-5: 0.483
  * 5e-5: 0.482

* warmup_ratio
  * 0: 0.487
  * 0.05: 0.486
  * 1: 0.509 <- なにやってんねん

* feature_enable
  * True 0.479
  

## exp164: 
roberta-large tuning

* bert_lr in [2e-5, 4e-5, 5e-5]
* dropout_bert
* gradient_clip=0.2 / lr_fc = 0.2 / feature_enable=True

## exp165
* xlnet_base_cased
  * reinit_layer
    * 0: 0.499
    * 1: 0.497
    * 2: 0.499
    * 3: 0.502
    * 4: 0.507

* distilbert-base-cased
  * reinit_layer(only 0)
    * 0: 0.503
    * 1: 0.514
    * 2: 0.516
    * 3: 0.512
    * 4: 0.540
  
* albert-v2-based
  * base -> 0.516
  
  * lr_bert (only 0)
    * 1e-5: 0.500
    * 2e-5: 0.507
    * 3e-5: 0.503
    * 4e-5: 0.501
    * 5e-5: 0.502
  
  * lr_fc
    * 1e-4: 0.502
    * 3e-4: 0.499
    * 5e-4: 0.511
    * 1e-5: 0.512
    * 3e-5: 0.492
    * 5e-5: 0.499
  
## exp166
* bert-largeをfold0だけでいろいろ
  * reinit_layer
    * 0: 0.491
    * 1: 0.498
    * 2: 0.495
    * 3: 0.490
    * 4: 0.494
  * lr_bert
    * 1e-5: 0.498
    * 2e-5: 0.495
    * 3e-5: 0.494
    * 4e-5: 0.496
    * 5e-5: 0.5
  * gradient_clip
    * 0.1: 0.492
    * 0.2: 0.485
    * 0.5: 0.494
  

https://discuss.pytorch.org/t/how-to-avoid-memory-leak-when-training-multiple-models-sequentially/100315/5

# 2021/7/15
## exp180


## exp180_2
dropout = 0