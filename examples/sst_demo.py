import warnings
import random
from typing import Iterable, List, Tuple

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from forte.data.data_pack import DataPack
from forte.data.ontology.core import Entry
from ft.onto.base_ontology import Sentence
from fortex.aug.data.data_aug_iterator import DataAugIterator, IterPrep

from models import model_train

warnings.simplefilter(action="ignore", category=Warning)

dataset_path = "sst"
print('loading data...')
data_packs: Iterable[DataPack] = IterPrep('sst', dataset_path)
data_packs = iter(list(data_packs)[0:5])
print('finish loading')

configs1 = {
    "prob": 1.0,
    "model_to": "fortex.aug.utils.machine_translator.MarianMachineTranslator",
    "model_back": "fortex.aug.utils.machine_translator.MarianMachineTranslator",
    "src_language": "en",
    "tgt_language": "fr",
    "device": "cuda",
}

# configs2 = {
#     "segment_type": "Sentence",
# }

# def data_pack_node_weighting(pack: DataPack, sentence: Sentence) -> float:
#     entity_num = 0.0
#     for _ in pack.get(Sentence, sentence):
#         entity_num += 1
#     return entity_num


# return a random node id in a datapack
# def data_pack_random_node(
#     pack: DataPack, sentence: Sentence, num_entity: int
# ) -> int:
#     rand_idx = random.randint(0, num_entity - 1)
#     for idx, entity in enumerate(pack.get(Sentence, sentence)):
#         if rand_idx == idx:
#             return entity.tid


# get all labels for all sentence.
# Since labels/sentiments are not stored in Sentence, we manually retrieve them.
def get_label(pack: DataPack, context: Entry = Sentence) -> Tuple[List, List]:
    l = []
    t = []
    sentences = list(pack.get(context))
    for s in sentences:
        t.append(s.text)
        l.append(s.sentiment)
    return t, l

da_iter = DataAugIterator(data_packs, get_label, Sentence)
da_iter.aug_with("back_trans", configs1)
# da_iter.aug_with("mix_up", configs2, data_pack_node_weighting, data_pack_random_node)

# demo for the API
# We use pre-trained Bert model to predict directly and compare the result of
# the original data and back translation augmented data.
texts = []
labels = []
for i, (augment_steps, text, label) in enumerate(da_iter):
    if augment_steps == "original_data":
        texts.extend(text)
        labels.extend(label)
    elif augment_steps == "back_trans":
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
