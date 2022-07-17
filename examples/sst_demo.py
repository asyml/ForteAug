import os
import warnings
import random
from typing import Iterable

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.optimization import AdamW

from forte.data.data_pack import DataPack
from forte.data.readers import SST2Reader
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, Token
from fortex.aug.data.data_aug_iterator import DataAugIterator

from data_sst import ClassificationDataProcessor
from utils import train
from models import BertModel

warnings.simplefilter(action="ignore", category=Warning)

pipeline = Pipeline()
reader = SST2Reader()
pipeline.set_reader(reader)
pipeline.initialize()
dataset_path = "/Users/lechuanwang/Downloads/ForteAug/sst"

data_packs: Iterable[DataPack] = pipeline.process_dataset(dataset_path, 5)

# small dataset for demo
data_packs = iter(list(data_packs)[0:5])

configs1 = {
    "prob": 1.0,
    "model_to": "fortex.aug.utils.machine_translator.MarianMachineTranslator",
    "model_back": "fortex.aug.utils.machine_translator.MarianMachineTranslator",
    "src_language": "en",
    "tgt_language": "fr",
    "device": "cpu",
}
configs2 = {
    "segment_type": "Sentence",
}


def data_pack_node_weighting(pack: DataPack, sentence: Sentence) -> float:
    entity_num = 0.0
    for _ in pack.get(Sentence, sentence):
        entity_num += 1
    return entity_num


# return a random node id in a datapack
def data_pack_random_node(
    pack: DataPack, sentence: Sentence, num_entity: int
) -> int:
    rand_idx = random.randint(0, num_entity - 1)
    for idx, entity in enumerate(pack.get(Sentence, sentence)):
        if rand_idx == idx:
            return entity.tid


# get all labels for all sentence.
# Since labels/sentiments are not stored in Sentence, we manually retrieve them.
def get_label(pack):
    l = []
    t = []
    sentences = list(pack.get(Sentence))
    for s in sentences:
        t.append(s.text)
        l.append(s.sentiment)
    return t, l


def predict(input, label):
    tokenizer = AutoTokenizer.from_pretrained(
        "textattack/bert-base-uncased-SST-2"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-SST-2"
    )
    input = [tokenizer.encode(x, add_special_tokens=True) for x in input]
    max_len = 0
    for i in input:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([x + [0] * (max_len - len(x)) for x in input])
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    features = last_hidden_states[0].numpy()

    lr_clf = LinearRegression()
    lr_clf.fit(features, label)

    print(lr_clf.score(features, label))


def model_train(
    texts,
    labels,
    target_dir="bertmodels",
    max_seq_len=50,
    epochs=3,
    batch_size=32,
    lr=2e-05,
    patience=1,
    max_grad_norm=10.0,
    if_save_model=True,
):
    model = BertModel(requires_grad=True)
    tokenizer = model.tokenizer
    train_data = ClassificationDataProcessor(
        tokenizer, texts, labels, max_seq_len=max_seq_len
    )
    train_loader = DataLoader(train_data, batch_size=batch_size)

    device = torch.device("cpu")
    model = model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.85, patience=0
    )

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    valid_aucs = []

    # -------------------- Training epochs -----------------------------------#

    print(
        "\n",
        20 * "=",
        "Training bert model on device: {}".format(device),
        20 * "=",
    )
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(
            model, train_loader, optimizer, epoch, max_grad_norm
        )
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(
            "-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".format(
                epoch_time, epoch_loss, (epoch_accuracy * 100)
            )
        )

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            if if_save_model:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "train_accuracy": train_accuracies,
                        "valid_losses": valid_losses,
                        "valid_accuracy": valid_accuracies,
                        "valid_auc": valid_aucs,
                    },
                    os.path.join(target_dir, "best.pth.tar"),
                )
                print("save model succesfully!\n")

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


da_iter = DataAugIterator(data_packs)
da_iter.aug_with("back_trans", configs1)
# da_iter.aug_with("mix_up", configs2, data_pack_node_weighting, data_pack_random_node)

# demo for the API
# We use pre-trained Bert model to predict directly and compare the result of
# the original data and back translation augmented data.
texts = []
labels = []
for i, (augment_steps, pack) in enumerate(da_iter):
    if augment_steps == "original_data":
        text, label = get_label(pack)
        texts.extend(text)
        labels.extend(label)
    elif augment_steps == "back_trans":
        text, label = get_label(pack)
        texts.extend(text)
        labels.extend(label)

model_train(texts, labels)

# elif augment_steps == "mix_up":
#     text = []
#     label = []
#     # two mixed data
#     for p in pack:
#         t, l = get_label(p)
#         text.extend(t)
#         label.extend(l)
#     print("mix_up:")
#     print(text)
#     print(label)
#     predict(text, label)
