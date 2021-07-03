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
lr_bert_decay = 0.9, lr_bert = 5e-5 -> CV: 0.4


