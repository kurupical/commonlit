import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from datetime import datetime as dt
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import tqdm
import math

try:
    import mlflow
except Exception:
    print("import error: mlflow")

np.random.seed(0)
class BertSequenceVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'bert-base-cased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 256

    def vectorize(self, text: str) -> np.array:
        text = self.tokenizer(text,
                              max_length=self.max_len,
                              padding="max_length",
                              truncation=True,
                              return_tensors="pt")

        input_ids = text["input_ids"][0].reshape(1, -1).to("cuda")
        attention_mask = text["attention_mask"][0].reshape(1, -1).to("cuda")
        seq_out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)

        if torch.cuda.is_available():
            return seq_out[0].mean(dim=1).cpu().detach().numpy()  # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0].mean(dim=1).detach().numpy()


class BertPerplexityExtracter:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'bert-base-cased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 256

    def vectorize(self, text: str) -> np.array:
        text = self.tokenizer(text,
                              max_length=self.max_len,
                              padding="max_length",
                              truncation=True,
                              return_tensors="pt")

        input_ids = text["input_ids"][0]

        perplexities = []
        for i in range(5):
            input_ids_masked = [x if np.random.random() > 0.15 else self.tokenizer.mask_token_id for x in input_ids]
            input_ids_masked = torch.LongTensor(input_ids_masked).reshape(1, -1).to("cuda")
            attention_mask = text["attention_mask"][0].reshape(1, -1).to("cuda")
            seq_out = self.bert_model(input_ids=input_ids_masked, attention_mask=attention_mask)
            perplexities.append(math.exp(torch.mean(seq_out[0]).cpu().detach().numpy()))
        return perplexities


def replace_stop_words(x):
    x = x.replace(".", " . ")
    x = x.replace(",", " , ")
    x = x.replace("!", " ! ")
    x = x.replace("?", " ? ")
    x = x.replace("\n", " \n ")
    x = x.replace(")", " ) ")
    x = x.replace("(", " ( ")
    x = x.replace('"', ' " ')
    x = x.replace(";", " ; ")

    x = x.replace("  ", " ")
    x = x.replace("  ", " ")
    return x


def get_embeddings(bsv, x):
    return bsv.vectorize(x)

def get_perplexities(bpe, x):
    return bpe.vectorize(x)

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


def feature_engineering(df):
    df_ret = df[["id", "excerpt", "target", "kfold"]]
    excerpt = df["excerpt"].values
    excerpt = [replace_stop_words(x) for x in excerpt]

    df_ret["total_words"] = [total_words(x) for x in excerpt]
    df_ret["total_unique_words"] = [total_unique_words(x) for x in excerpt]
    df_ret["total_characters"] = [total_charactors(x) for x in excerpt]
    df_ret["total_sentence"] = [total_sentence(x) for x in excerpt]
    df_ret["total_isupper"] = [x.isupper() for x in excerpt]
    df_ret["total_islower"] = [x.islower() for x in excerpt]

    df_ret["div_sentence_characters"] = df_ret["total_sentence"] / df_ret["total_characters"]
    df_ret["div_sentence_words"] = df_ret["total_sentence"] / df_ret["total_words"]
    df_ret["div_characters_words"] = df_ret["total_characters"] / df_ret["total_words"]
    df_ret["div_words_unique_words"] = df_ret["total_words"] / df_ret["total_unique_words"]
    df_ret["div_isupper_words"] = df_ret["total_isupper"] / df_ret["total_words"]
    df_ret["div_islower_words"] = df_ret["total_islower"] / df_ret["total_words"]

    emb_path = "output/bert_embeddings.npy"
    if os.path.isfile(emb_path):
        bert_embeddings = np.load(emb_path)
    else:
        bsv = BertSequenceVectorizer()
        bert_embeddings = np.array([get_embeddings(bsv, x) for x in tqdm.tqdm(excerpt)]).squeeze(1)
        np.save(emb_path, bert_embeddings)
    for i in range(bert_embeddings.shape[1]):
        df_ret[f"bert_{i}"] = bert_embeddings[:, i]

    perp_path = "output/bert_perplexities.npy"
    if os.path.isfile(perp_path):
        bert_perplexities = np.load(perp_path)
    else:
        bpe = BertPerplexityExtracter()
        bert_perplexities = np.array([get_perplexities(bpe, x) for x in tqdm.tqdm(excerpt)])
        np.save(perp_path, bert_perplexities)
    df_ret[f"perplexities_mean"] = np.array(bert_perplexities).mean(axis=1)
    df_ret[f"perplexities_max"] = np.array(bert_perplexities).max(axis=1)
    df_ret[f"perplexities_min"] = np.array(bert_perplexities).min(axis=1)
    df_ret[f"perplexities_std"] = np.array(bert_perplexities).std(axis=1)

    for i, word in enumerate(["!", "?", "(", ")", "'", '"', ";", ".", ","]):
        df_ret[f"count_word_special_{i}"] = [x.count(word) for x in excerpt]

    difficult_words = [
        'its', 'which', 'being', 'such', 'may', 'those', 'power',
        'has', 'between', 'war', 'been', 'any', 'an', 'upon', 'present',
        'form', 'or', 'by', 'known', 'certain', 'possible', 'thus',
        'less', 'this', 'well', '1', 'however', 'given', 'order', 'either',
        'than', 'should', 'action', 'latter', 'per', 'In', 'therefore',
        'state', 'itself', 'Mr', 'against', 'same', 'most', 'Government',
        'result', 'great', 'without', 'these', 'provided', 'as',
        'material', 'light', 'iron', 'part', 'means', 'German', 'only',
        'These', 'nature', 'greater', 'general', '2', 'within', '5',
        'account', 'whose', 'life', 'character', 'obtained', 'distance',
        'taken', 'generally', 'since', 'now', 'purpose', 'question', 'due',
        'France', 'surface', 'former', 'natural', 'be', 'matter', 'above',
        'method', '000', 'are', 'manner', 'England', 'system', 'subject',
        'common', 'country', 'pressure', 'history', 'To', 'whole', 'human',
        'case', 'consists', 'necessary', 'yet', '4', 'especially',
        'results', 'below', 'perhaps', 'public', 'Germany', 'space', 'far',
        'temperature', 'length', 'force', 'object', 'theory', 'e',
        'formed', 'London', 'conditions', 'specific', 'sense', 'effect',
        'from', 'term', 'century', 'including', 'methods', 'process',
        'systems', 'political', '3', 'M', 'second', 'produced', 'is',
        'observed', 'developed', 'both', 'ordinary', 'machine', 'French',
        'nearly', 'quantity', 'The', 'current', '6', 'fixed', 'service',
        'acid', 'high', 'men', 'age', 'application', 'weight',
        'development', 'contains', 'entirely', 'during', 'hour', 'source',
        'electric', 'vessel', 'other', 'position', 'points', 'required',
        'thousand', 'becomes', 'construction', 'data', 'origin', 'each',
        'number', 'least', 'peace', 'experiments', 'persons', 'at', 'Thus',
        'used', 'arranged', 'regard', 'true', 'twenty', 'considered',
        'cases', 'army', 'level', 'uses', 'suitable', 'vast', 'direct',
        'fact', 'following', 'remained', 'charge', 'gas', 'purposes',
        'hydrogen', 'success', 'modern', 'our', 'direction', 'sufficient',
        'single', 'increased', 'end', 'amount', 'A', 'among',
        'interesting', 'carried', 'presence', 'designed', 'must',
        'difference', 'circumstances', 'steel', 'processes', 'first',
        'solution', 'ancient', 'become', 'clear', 'having', 'heat', 'On',
        'lower', 'influence', 'Europe', 'volume', 'English', 'require',
        'military', 'addition', 'peculiar', '50', 'produce', '100',
        'applied', 'Russia', 'works', 'policy', 'qualities', 'done',
        'apparatus', 'more', 'point', 'published', 'solved', 'chemical',
        'arrangement', 'increase', 'several', 'Paris', 'world', 'metal',
        'own', 'condition', 'equal', 'view', 'containing', '40',
        'composed', 'attempt', 'advanced', 'remarkable', 'employed',
        'diameter', 'support', 'strength', 'operations', 'shown', 'stars',
        'with', 'nations', 'physical', 'necessity', 'mass', 'forms',
        'established', 'relations', 'absolute', 'energy', 'exist', 'law',
        'powers', 'cent', 'European', 'pure', 'bodies', 'composition',
        'face', 'producing', '8', 'square'
    ]
    difficult_columns = []
    for word in difficult_words:
        col_name = f"count_word_{word}"
        df_ret[col_name] = [x.split(" ").count(word) for x in excerpt]
        difficult_columns.append(col_name)

    df_ret["difficult_count_sum"] = df_ret[difficult_columns].sum(axis=1)
    df_ret["difficult_count_mean"] = df_ret[difficult_columns].mean(axis=1)
    df_ret["difficult_count_max"] = df_ret[difficult_columns].max(axis=1)
    df_ret["difficult_count_min"] = df_ret[difficult_columns].min(axis=1)
    df_ret["difficult_count_std"] = df_ret[difficult_columns].std(axis=1)
    df_ret["difficult_count_ratio"] = df_ret["difficult_count_sum"] / df_ret["total_words"]

    easy_words = [
        'fun', 'neck', 'round', 'named', 'answered', 'pulled', 'Soon',
        'crying', 'straight', 'frightened', 'asleep', 'chair', 'walking',
        'person', 'bird', 'Come', 'seemed', 'pick', 'papa', 'watch',
        'cake', 'ride', 'fly', 'ate', 'safe', 'horse', 'dear', 'ways',
        'everything', 'winter', 'kind', 'will', 'there', 'inside',
        'garden', 'Suddenly', 'anything', 'noise', 'else', 'live',
        'saying', 'if', 'who', 'Some', 'felt', 'parents', 'can', 'were',
        'climbed', 'bit', 'dark', 'angry', 'Do', 'once', 'then',
        'At', 'bright', 'eating', 'better', 'shouted', "couldn't", 'Have',
        'window', 'cry', 'However', "It's", 'wait', 'woman', 'Mother',
        'brain', 'along', 'red', 'watching', 'stay', 'white', 'cold',
        'talking', 'over', 'Every', 'girls', 'tried', 'black', 'That',
        'would', 'caught', 'hear', "Don't", 'today', 'flew', 'road',
        'snow', 'was', 'decided', 'sit', 'baby', 'grass', 'lot',
        'beautiful', 'stop', 'friend', 'long', 'catch', 'playing', 'After',
        'ground', 'liked', 'learn', 'walk', 'Her', 'jumped', 'kept',
        'tired', 'money', 'herself', 'afraid', 'run', 'try', 'near', 'And',
        'loved', 'left', 'sure', 'trying', 'sister', 'many', 'getting',
        'his', 'eyes', 'Just', 'night', 'sun', 'sad', 'room', 'enough',
        'My', 'replied', 'fell', 'right', 'People', 'before', 'walked',
        'knew', 'never', 'when', 'am', 'why', 'pretty', "can't", 'bed',
        'keep', 'fast', 'nice', 'head', 'family', 'boys', 'child', 'gave',
        'much', 'hungry', 'opened', 'mamma', 'poor', 'so', 'Sometimes',
        'sat', 'gone', "didn't", 'birds', 'thing', 'stopped', 'them',
        'Once', 'trees', 'young', 'time', 'thinking', 'Why', 'happy',
        'warm', 'coming', 'had', 'We', 'sleep', 'let', 'animals', 'Yes',
        'dog', 'people', 'door', 'next', "I'm", 'about', 'looking', 'Well',
        'There', 'wanted', 'look', 'he', "don't", 'again', 'something',
        'way', 'lived', 'girl', 'hard', 'play', 'stood', 'Now', 'tree',
        'school', 'come', 'found', 'How', 'things', 'they', 'find',
        'friends', 'father', 'make', 'I', 'like', 'started', 'off', 'take',
        'heard', 'my', 'too', 'told', 'boy', 'morning', 'soon', 'going',
        'him', 'children', 'cried', 'good', 'What', 'began', 'around',
        'tell', 'old', 'food', 'very', 'Oh', 'house', 'what', 'up', 'ran',
        'want', 'out', 'could', 'down', 'big', 'know', 'because', 'help',
        'little', 'called', 'He', 'looked', 'took', 'how', 'You',
        'eat', 'got', 'They', 'did', 'put', 'think', 'But', 'back', 'just',
        'mother', 'thought', 'So', 'away', 'her', 'asked', 'your', 'day',
        'When', 'do', 'saw', 'came', 'see', 'home', 'me', 'Then', 'she',
        'go', 'you', 'get', 'She', 'said', 'went', 'One'
    ]
    easy_columns = []
    for word in easy_words:
        col_name = f"count_word_{word}"
        df_ret[f"count_word_{word}"] = [x.count(word) for x in excerpt]
        easy_columns.append(col_name)

    df_ret["easy_count_sum"] = df_ret[easy_columns].sum(axis=1)
    df_ret["easy_count_mean"] = df_ret[easy_columns].mean(axis=1)
    df_ret["easy_count_max"] = df_ret[easy_columns].max(axis=1)
    df_ret["easy_count_min"] = df_ret[easy_columns].min(axis=1)
    df_ret["easy_count_std"] = df_ret[easy_columns].std(axis=1)
    df_ret["easy_count_ratio"] = df_ret["easy_count_sum"] / df_ret["total_words"]

    df_ret["div_difficult_easy"] = df_ret["easy_count_sum"] / df_ret["difficult_count_sum"]

    return df_ret


def main(params):
    output_dir = f"output/{os.path.basename(__file__)[:-3]}/{dt.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir)

    df = pd.read_csv("input/commonlitreadabilityprize/train_folds.csv")
    df = feature_engineering(df)

    mlflow.start_run(experiment_id=1)
    for key, value in params.items():
        mlflow.log_param(key, value)

    rmses = []
    for fold in range(5):
        df_train = df[df["kfold"] != fold]
        df_val = df[df["kfold"] == fold]

        X_train = df_train.drop(["id", "excerpt", "target", "kfold"], axis=1)
        y_train = df_train["target"]
        X_val = df_val.drop(["id", "excerpt", "target", "kfold"], axis=1)
        y_val = df_val["target"]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val)

        model = lgb.train(params=params,
                          train_set=lgb_train,
                          valid_sets=[lgb_train, lgb_val],
                          early_stopping_rounds=300,
                          verbose_eval=500)

        y_pred = model.predict(X_val)
        # output oof
        df_oof = df_val[["id"]]
        df_oof["pred"] = y_pred
        df_oof["target"] = y_val
        df_oof.to_csv(f"{output_dir}/val_fold{fold}_best.csv", index=False)

        # output feature importance
        df_imp = pd.DataFrame()
        df_imp["col"] = X_val.columns
        df_imp["imp"] = model.feature_importance(importance_type="gain")
        df_imp["imp"] = df_imp["imp"] / df_imp["imp"].sum()
        df_imp.sort_values("imp", ascending=False).to_csv(f"{output_dir}/fold{fold}_importance.csv", index=False)

        rmse = np.sqrt(1 / len(y_pred) * ((y_val - y_pred)**2).sum())

        mlflow.log_metric(f"fold{fold}", rmse)
        rmses.append(rmse)

    mlflow.log_metric(f"fold_mean", np.array(rmses).mean())
    mlflow.end_run()

if __name__ == "__main__":
    params = {
        'objective': 'regression',
        'num_leaves': 16,
        'max_depth': -1,
        'learning_rate': 0.01,
        'boosting': 'gbdt',
        'bagging_fraction': 0.7,
        'feature_fraction': 0.1,
        'bagging_seed': 0,
        'reg_alpha': 5,  # 1.728910519108444,
        'reg_lambda': 15,
        'random_state': 0,
        "metrics": "rmse",
        'verbosity': -1,
        "n_estimators": 10000,
        "early_stopping_rounds": 100
    }
    main(params=params)


