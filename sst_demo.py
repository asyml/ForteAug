import numpy as np
import torch
import warnings
from typing import Iterable
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from forte.data.data_pack import DataPack
from ft.onto.base_ontology import Token
from fortex.aug.data.data_aug_iterator import DataAugIterator
from forte.data.readers import SST2Reader
from forte.pipeline import Pipeline
from ft.onto.base_ontology import ConstituentNode, Sentence, Token
from sklearn.linear_model import LinearRegression

warnings.simplefilter(action='ignore', category=Warning)

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")

pipeline = Pipeline()
reader = SST2Reader()
pipeline.set_reader(reader)
pipeline.initialize()
dataset_path = "/Users/lechuanwang/Downloads/ForteAug/sst"

data_packs: Iterable[DataPack] = pipeline.process_dataset(
    dataset_path, 5
)

# small dataset for demo
data_packs = iter(list(data_packs)[0:15])

phrase_to_id = {}
# Read the mapping from phrase to phrase-id.
with open(dataset_path + "/dictionary.txt", "r", encoding="utf8") as file:
    for line in file:
        phrase, id_ = line.split("|")
        phrase_to_id[phrase] = int(id_)

id_to_senti = {}
# Read the mapping from phrase-id to sentiment score.
with open(dataset_path + "/sentiment_labels.txt", "r", encoding="utf8") as file:
    for i, line in enumerate(file):
        if i == 0:
            continue
        id_, score = line.split("|")
        id_to_senti[int(id_)] = float(score)

configs1={
    "prob": 1.0,
    "model_to": "fortex.aug.utils.machine_translator.MarianMachineTranslator",
    "model_back": "fortex.aug.utils.machine_translator.MarianMachineTranslator",
    "src_language": "en",
    "tgt_language": "fr",
    "device": "cpu",
}

da_iter = DataAugIterator(data_packs)
da_iter.aug_with("back_trans", configs1)

# get all labels for all sentence.
# Since labels/sentiments are not stored in Sentence, we manually retrieve them.
def get_label(pack):
    res = []
    text = []
    sentences = list(pack.get(Sentence))
    for s in sentences:
        text.append(s.text)
        res.append(id_to_senti[phrase_to_id[s.text]])
    return text, res

def predict(input, label):
    input = [tokenizer.encode(x, add_special_tokens=True) for x in input]
    max_len = 0
    for i in input:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([x + [0]*(max_len-len(x)) for x in input])
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    features = last_hidden_states[0].numpy()

    lr_clf = LinearRegression()
    lr_clf.fit(features, label)

    print(lr_clf.score(features, label))


# Store labels of the original data. They will be used as labels for the
# augmented data for prediction.
labels = []
# demo for the API
# We use pre-trained Bert model to predict directly and compare the result of
# the original data and back translation augmented data.
for i, (augment_steps, pack) in enumerate(da_iter):
    if augment_steps == "original_data":
        text, label = get_label(pack)
        labels.append(label)
        print("original_data:")
        predict(text, label)
    elif augment_steps == "back_trans":
        # use index to locate the corresponding label of the original data.
        label = labels[i % len(labels)]
        text = []
        sentences = list(pack.get(Sentence))
        for s in sentences:
            text.append(s.text)
        print("back_trans:")
        predict(text, label)
