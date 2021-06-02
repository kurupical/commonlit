# Jigsaw Unintended Bias in Toxicity Classification
## 3rd: 
https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471

* 単純なLSTMとのアンサンブルも大事?

* Finetune BERT https://arxiv.org/pdf/1905.05583.pdf
  4. Methodology
  * 4-1. Fine-Tuning strategies
    * どの層が一番ターゲットのタスクに向いている？
    * どのようにLRを決めたらいいか?
  * 4-2. Further Pre-training
    * 分布と違うんだから学習しなおし必要だよね
    * これが一番強い
  * Multi-Task Fine-Tuning
    * ?
  
## 4th 
https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100811
* blending 12-LSTM models
* 17 BERT
* 2 GPT-2
* CNN-headは、CV改善しなかったけどモデルのdiversity観点ではよかった

## 10th 
https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/101630
* further pretrained with competition data.
* BERT large cased & uncased
* GPT-2はスコア出なかったけどアンサンブルに効いた
* RNN
  * Glove 300d embedding
  * Crawl 300d embedding
  * Wordvec 100d embedding
  * char-level count vector and keyword-based features
  * 3 RNN layers + attention pooling + classifier
* BERTは0.98ずつ学習率を下げる
  * first layerの学習率は5e-5*0.98^12

## 14th
https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100821
* RNN
* LightGBM / Catboost (bert 768d feature)

