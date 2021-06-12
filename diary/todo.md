# アンサンブルの実験しやすいようにする

# book corpasだけでfinetune-lmしてみる
    -> データでかすぎるのでやめ
# SGD

# Remove 4layers-mean and lstm

# ranking 学習

# BERT, ALBERT, XLNetはpretrainさせてからもう一度

# warmup_scheduler, weight_decayをいじってfinetuneする

# book-corpus, training_step=10000 * 5epoch
    bert-base, albert-base, roberta-base

# 単語の予測のしにくさを特徴に入れる?(perplexity)

=> LGBM
=> BERTなど　※not pretrainedなモデルに限る
https://arxiv.org/pdf/1907.11779.pdf

# カスタムスケジューラーを作る。val_rmseが低くなったらlrを下げて、そうじゃなかったら上げる、てきなやつｗ
