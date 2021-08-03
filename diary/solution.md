# Public 6th / Private 22nd Place solution

# Acknowledgement
First of all, thanks to competition organizers for hosting this competition.<br>
And I would like to thank you @torch for share a lot of great notebook and discussion. I couldn't have achieved this ranking without you, and I learned a lot method for fine-tune BERT models from you!<br>

# summary
I have three options(below), and in the end I chose the top two for final submission.
* LB best (9models, CV: 0.4449, LB: 0.444) 
* CV best exclude base-model (9models, CV: 0.4430, LB: 0.447)
* CV best (12models, CV: 0.4420, LB: 0.453)

The detail of LB best is below table:<br>

## LB-Best

|                   | CV(5fold) | weight | Public | Private | other        |
|-------------------|-----------|--------|--------|---------|--------------|
| luke-large        | 0.4795    | 0.0877 | 0.467  |         |              |
| roberta-large     | 0.4693    | 0.1035 | 0.464  |         |              |
| gpt2-medium       | 0.481     | 0.158  | 0.469  |         |              |
| funnel-large-base | 0.4706    | 0.1455 | 0.461  |         |              |
| funnel-large      | 0.4705    | 0.0267 | 0.46   |         |              |
| ernie-large       | 0.4712    | 0.093  | 0.463  |         |              |
| deberta-large     | 0.475     | 0.1693 | 0.464  |         |              |
| mpnet-base        | 0.4782    | 0.1415 | 0.466  |         |              |
| gpt2-large        | 0.4744    | 0.1121 | 0.468  |         | batch_size=4 |
| coef              | -         | 0.0235 | -      |         |              |


## CV-Best

|                  | CV(5fold) | weight | Public | Private | other        |
|------------------|-----------|--------|--------|---------|--------------|
| electra-large    | 0.4743    | 0.0829 | 0.474  |         |              |
| funnel-large     | 0.4752    | 0.1215 | 0.467  |         |              |
| deberta-large(1) | 0.4738    | 0.1516 | 0.47   |         |              |
| gpt2-medium      | 0.4775    | 0.1219 | 0.471  |         |              |
| luke-large       | 0.474     | 0.0937 | 0.468  |         |              |
| deberta-large(2) | 0.475     | 0.1179 | 0.464  |         |              |
| mpnet-base       | 0.4782    | 0.0999 | 0.466  |         |              |
| ernie-large      | 0.4681    | 0.0792 | 0.47   |         |              |
| gpt2-large       | 0.4744    | 0.1648 | ?      |         | batch_size=4 |
| coef             | -         | 0.009  | -      |         |              |


# configuration for almost all models
* epochs = 4
* optimizer: AdamW
* scheduler: linear_schedule_with_warmup(warmup: 5%)
* lr_bert: 3e-5
* batch_size: 12
* gradient clipping: 0.2~0.5
* reinitialize layers: last 2~6 layers  
* ensemble: Nelder-Mead
* custom head(finally concat all)
  * averaging last 4 hidden layer
  * LSTM head
  * vocabulary dense
  ```
  hidden_states: (batch_size, vocab_size, bert_hidden_size)
  linear_vocab = nn.Sequential(
      nn.Linear(bert_hidden_size, 128),
      nn.GELU(),
      nn.Linear(128, 64),
      nn.GELU()
  )
  linear_final = nn.Linear(vocab_size * 64, 128)
  out = linear_vocab(hidden_states).view(len(input_ids), -1)) # final shape: (batch_size, vocab_size * 64)
  out = linear_final(out) # out shape: (batch_size, 128)
  ```
* 17 hand-made features 
  * sentence count
  * average character count in documents

  
# worked for me
* model ensemble: I thought diversity is the most important thing in this competition.
  * At the beginning of the competition, I tested the effectiveness of the ensemble.
  * Up to the middle stage, I fixed the model to roberta-large and tried to improve the score. 
  * At the end, I applied the method to another models. I found that key parameters for this task are {learning_rate, N layers to re-initialize}, so I tuned those parameters for each models.
* re-initialization
  * This paper (https://arxiv.org/pdf/2006.05987.pdf) shows that fine-tuning with reinitialization last N layers works well.
  * Different models have different optimal N. Almost models set N=4~5, gpt2-models set N=6.
* LSTM head
  * Input BERT's first and last hidden layer into LSTM layer worked well.
  * I think first layer represent vocabulary difficulty and last layer represent sentence difficulty. Both are important for inference readbility.
* Remove dropout. Improve 0.01~0.02 CV.
* gradient clipping. (0.2 or 0.5 works well for me, improve about 0.005 CV)

# not worked for me
* Input attention matrix to 2D-CNN(like ResNet18 or simple 2DCNN)
  * I thought this could represent the complexity of sentences with relative pronouns.
* masked 5%~10% vocabulary.
* Minimize KLDiv loss to fit distribution.
* Scale target to 0~1 and minimize crossentropy loss
* "base" models excluding mpnet. I got 0.47x CV but Public LB: 0.48x ~ 0.49x.
* Stacking using LightGBM.
* another models.(result is below table. single CV is well but zero weight for ensemble)
* T5. Below notebook achieve 0.47 LB using T5, so I tried but failed.
  I got only 0.49x(fold 0 only) with learning_rate=1.5e-4
  https://www.kaggle.com/gilfernandes/commonlit-pytorch-t5-large-svm/comments
  
  
| model             | CV(5fold) | Public |
|-------------------|-----------|--------|
| xlnet-large-cased | 0.471     | 0.473  |
| bert-large-cased  | 0.4828    | 0.488  |


# GPU
* GeForce RTX3090 * 1.

# hyperparameter detail
* attatched csv(cv_best_parameters.csv, lb_best_parameters.csv)