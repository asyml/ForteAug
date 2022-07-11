#  Copyright 2020 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Provide data across multiple data packs during training. A data pack iterator
iterates over each single data example across multiple data packs. A data pack
data set represents the dataset of a bunch of data packs. A raw example
represents a single data point in the dataset. A feature collection represents
an extracted feature corresponding to an input data point.
"""
# pylint: skip-file
from typing import (
    Dict,
    Iterator,
    Type,
    Optional,
    List,
    Tuple,
    Union,
    Any,
    Callable,
)
import copy
import random
import numpy as np

from forte.utils.utils import get_class
from forte.common.configurable import Configurable
from forte.datasets.conll.conll_utils import get_tag
from forte.data.data_pack import DataPack
from forte.common.configuration import Config
from forte.utils import create_import_error_msg
from ft.onto.base_ontology import Token, Sentence, EntityMention
from fortex.aug.data.data_pack_weight_sampler import DataPackWeightSampler

try:
    import torch
except ImportError as e:
    raise ImportError(
        create_import_error_msg("torch", "extractor", "data pack dataset")
    ) from e

try:
    from texar.torch.data import IterDataSource, DatasetBase, Batch
except ImportError as e:
    raise ImportError(
        create_import_error_msg(
            "texar-pytorch", "extractor", "data pack dataset"
        )
    ) from e


def get_label_ids(label):
    for i, l in enumerate(label):
        if l == 1:
            return i
    return -1


class Instance(object):
    def __init__(self, input_ids, input_mask, valid, labels, label_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.valid = valid
        self.labels = labels
        self.label_mask = label_mask


class MixInstance(object):
    def __init__(
        self,
        input_ids_a,
        input_ids_b,
        input_mask_a,
        input_mask_b,
        valid_a,
        valid_b,
        mix_ratio,
        labels,
        mix_idx,
        label_mask,
    ):
        self.input_ids_a = input_ids_a
        self.input_ids_b = input_ids_b
        self.input_mask_a = input_mask_a
        self.input_mask_b = input_mask_b
        self.valid_a = valid_a
        self.valid_b = valid_b
        self.mix_ratio = mix_ratio
        self.labels = labels
        self.mix_idx = mix_idx
        self.label_mask = label_mask


class OriginalDataIterator:
    def __init__(self, iterator):
        self.original_data_iterator = iterator
        self._curr_data_pack = next(self.original_data_iterator)
        self._context_iter = self._curr_data_pack.get(Sentence)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            sent = next(self._context_iter)
            tag = get_tag(
                self._curr_data_pack, sent, Token, EntityMention, "ner_type"
            )
        except StopIteration:
            self._curr_data_pack = next(self.original_data_iterator)
            self._context_iter = self._curr_data_pack.get(Sentence)
            sent = next(self._context_iter)
            tag = get_tag(
                self._curr_data_pack, sent, Token, EntityMention, "ner_type"
            )
        d = DataPack()
        d.set_text(sent.text)
        setattr(d, "tag", tag)
        return d


class MixUpIterator(Configurable):
    """ """

    def __init__(
        self,
        pack_iterator: Iterator[
            DataPack
        ],  # data iterator to construct the segment pool
        data_pack_weighting_fn: Callable[
            [DataPack, Sentence], float
        ],  # weighting function for sampling
        segment_annotate_fn: Callable[
            [DataPack, Sentence, int], int
        ],  # function to pick a segment from a sampled instance
        configs: Union[Config, Dict[str, Any]],
        train_iterator,  # training data iterator
        eval_iterator=None,  # eval data iterator
        test_iterator=None,
    ):
        self.configs = self.make_configs(configs)
        self._data_request = {
            Token: {"fields": ["ner"]},
        }
        augment_entry = ""
        if self.configs["segment_type"] == "Entity":
            augment_entry = "ft.onto.base_ontology.EntityMention"
        elif self.configs["segment_type"] == "Token":
            augment_entry = "ft.onto.base_ontology.Token"
        elif self.configs["segment_type"] == "Sentence":
            augment_entry = "ft.onto.base_ontology.Sentence"
        self.augment_entry = get_class(augment_entry)
        self._segment_pool_sampler: DataPackWeightSampler = (
            DataPackWeightSampler(
                pack_iterator,
                data_pack_weighting_fn,
                self.configs["context_type"],
                self._data_request,
                self.configs["skip_k"],
            )
        )
        self.segment_pool = (
            self._segment_pool_sampler.generate_weighted_samples(
                self.configs["augment_size"]
            )
        )
        print(
            [
                sentence.text
                for sentence, pack, seg_num, tag in self.segment_pool
            ]
        )
        self.segment_annotate_fn = segment_annotate_fn
        self.train_iterator = train_iterator
        self.eval_iterator = eval_iterator
        self.test_iterator = test_iterator

        self.iter_mode_mixed = False
        self.train_mode = "TRAIN"

        self.mix_up_data_iterator = self.convert_segment_pool_to_pair_iterator()
        self.original_data_iterator = OriginalDataIterator(
            self.train_iterator()
        )

    def convert_segment_pool_to_pair_iterator(self):

        self._annotated_pack_pool = []

        for sentence, pack, seg_num, tag in self.segment_pool:
            seg_tid = self.segment_annotate_fn(pack, sentence, seg_num)
            annotated_pack = DataPack()
            annotated_pack.set_text(sentence.text)
            segment = pack.get_entry(seg_tid)
            type = getattr(segment, "ner_type")
            start, end = (
                segment.begin - sentence.begin,
                segment.end - sentence.begin,
            )
            entry = self.augment_entry(annotated_pack, start, end)
            setattr(entry, "ner_type", type)
            setattr(annotated_pack, "tag", tag)
            annotated_pack.add_entry(entry)
            self._annotated_pack_pool.append(annotated_pack)

        return iter(self._annotated_pack_pool)

    def switch_original_dataset(self, train_mode):
        self.train_mode = train_mode

        if self.train_mode == "TRAIN":
            self.original_data_iterator = OriginalDataIterator(
                self.train_iterator()
            )
        elif self.train_mode == "EVAL":
            self.original_data_iterator = OriginalDataIterator(
                self.eval_iterator()
            )
        elif self.train_mode == "TEST":
            self.original_data_iterator = OriginalDataIterator(
                self.test_iterator()
            )
        else:
            raise Exception

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_mode_mixed:
            try:
                data_a = next(self.mix_up_data_iterator)
                data_b = next(self.mix_up_data_iterator)
            except StopIteration as e:
                raise StopIteration from e
            return data_a, data_b
        else:
            return next(self.original_data_iterator)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "segment_type": "Entity",
            "context_type": Sentence,
            "skip_k": 0,
            "augment_size": 10,
            "max_len": 128,
        }


class MixupDataProcessor:
    @classmethod
    def _tokenize_sentence(
        cls, tokenizer, label_map, data_pack, tags, begin=-1, end=-1
    ):
        text_list = data_pack.text.split(" ")
        # label_list = example.label_a
        tokens = []
        labels = []
        valid = []
        label_mask = []
        (
            mix_start,
            mix_end,
            lab_mix_start,
            lab_mix_end,
            word_len,
            token_len,
            lab_len,
        ) = (0, 0, 0, 0, 0, 0, 0)
        tokens.append("[CLS]")
        valid.append(1)
        label_mask.append(1)
        labels.append(label_map["[CLS]"])
        token_len += 1
        lab_len += 1
        for i, word in enumerate(text_list):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            # print(word_len, begin, end)
            if word_len == begin:
                mix_start = token_len
                lab_mix_start = lab_len
            if word_len == end + 1:
                mix_end = token_len
                lab_mix_end = lab_len
            word_len += len(word) + 1
            token_len += len(token)
            for idx in range(len(token)):
                if idx == 0:
                    labels.append(label_map[tags[i]])
                    label_mask.append(1)
                    valid.append(1)
                    lab_len += 1
                else:
                    valid.append(0)
        tokens.append("[SEP]")
        valid.append(1)
        label_mask.append(1)
        labels.append(label_map["[SEP]"])
        return (
            tokens,
            valid,
            labels,
            label_mask,
            mix_start,
            mix_end,
            lab_mix_start,
            lab_mix_end,
        )

    def _adjust_segment_length(
        self, input_ids, mix_end, valid, labels, label_padding, adjust_len
    ):
        input_ids[mix_end:mix_end] = [0] * adjust_len
        valid[mix_end:mix_end] = [0] * adjust_len
        labels[mix_end:mix_end] = [label_padding] * adjust_len
        return input_ids, valid, labels, mix_end + adjust_len

    @classmethod
    def _adjust_seq_length(cls, sequence, max_len, padding=0, gen_mask=False):
        mask = None
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
            if gen_mask:
                mask = [1] * max_len
        else:
            if gen_mask:
                mask = [1] * len(sequence)
                mask += [0] * (max_len - len(sequence))
            sequence += [padding] * (max_len - len(sequence))

        assert len(sequence) == max_len
        if gen_mask:
            assert len(mask) == max_len
        return sequence, mask

    def get_original_data_batch(
        self,
        iterator: MixUpIterator,
        tokenizer,
        label_map,
        batch_size=1,
        max_len=128,
        mode="TRAIN",
    ):
        instances = []
        iterator.iter_mode_mixed = False
        iterator.switch_original_dataset(mode)
        for d in iterator:
            (
                tokens,
                valid,
                labels,
                label_mask,
                _,
                _,
                _,
                _,
            ) = self._tokenize_sentence(tokenizer, label_map, d, d.tag)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids, input_mask = self._adjust_seq_length(
                input_ids, max_len, gen_mask=True
            )
            valid, _ = self._adjust_seq_length(valid, max_len, gen_mask=False)
            labels, _ = self._adjust_seq_length(labels, max_len, label_map[""])
            label_mask, _ = self._adjust_seq_length(label_mask, max_len)
            labels = np.array(labels)
            instances.append(
                Instance(input_ids, input_mask, valid, labels, label_mask)
            )
            if len(instances) == batch_size:
                yield self._batch_fn(instances)
                instances = []

    def get_mixed_data_batch(
        self,
        iterator,
        tokenizer,
        mix_ratio_fn,
        label_map,
        batch_size=1,
        max_len=128,
    ):
        instances = []
        tags_a, tags_b = [], []
        begin_a, end_a, begin_b, end_b = 0, 0, 0, 0
        iterator.iter_mode_mixed = True
        for d_a, d_b in iterator:
            # print(d_a.text, d_b.text)
            for augment_entry in d_a.get(iterator.augment_entry):
                begin_a, end_a = augment_entry.begin, augment_entry.end
                tags_a = d_a.tag
            (
                tokens_a,
                valid_a,
                labels_a,
                label_mask_a,
                mix_start_a,
                mix_end_a,
                begin_a,
                end_a,
            ) = self._tokenize_sentence(
                tokenizer, label_map, d_a, tags_a, begin_a, end_a
            )
            tag_a = tags_a[begin_a - 1][2:]

            for augment_entry in d_b.get(iterator.augment_entry):
                begin_b, end_b = augment_entry.begin, augment_entry.end
                tags_b = d_b.tag
            (
                tokens_b,
                valid_b,
                labels_b,
                label_mask_b,
                mix_start_b,
                mix_end_b,
                begin_b,
                end_b,
            ) = self._tokenize_sentence(
                tokenizer, label_map, d_b, tags_b, begin_b, end_b
            )
            print(begin_b, end_b)
            print(tags_b)
            tag_b = tags_b[begin_b - 1][2:]

            input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
            input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

            len_a, len_b = mix_end_a - mix_start_a, mix_end_b - mix_start_b
            if len_b > len_a:
                (
                    input_ids_a,
                    valid_a,
                    labels_a,
                    mix_end_a,
                ) = self._adjust_segment_length(
                    input_ids_a,
                    mix_end_a,
                    valid_a,
                    labels_a,
                    label_map[""],
                    len_b - len_a,
                )
            else:
                (
                    input_ids_b,
                    valid_b,
                    labels_b,
                    mix_end_b,
                ) = self._adjust_segment_length(
                    input_ids_b,
                    mix_end_b,
                    valid_b,
                    labels_b,
                    label_map[""],
                    len_a - len_b,
                )

            input_ids_a, input_mask_a = self._adjust_seq_length(
                input_ids_a, max_len, gen_mask=True
            )
            input_ids_b, input_mask_b = self._adjust_seq_length(
                input_ids_b, max_len, gen_mask=True
            )
            valid_a, _ = self._adjust_seq_length(valid_a, max_len)
            valid_b, _ = self._adjust_seq_length(valid_b, max_len)

            mix_ratio = mix_ratio_fn(8)
            labels_a, _ = self._adjust_seq_length(
                labels_a, max_len, label_map[""]
            )
            labels_a = np.array(labels_a)
            labels_b = np.array(labels_b)
            mixed_label = self.mix_labels(
                labels_a,
                labels_b,
                mix_ratio,
                (begin_a, end_a, begin_b, end_b),
                tag_b,
                label_map,
            )
            instances.append(
                MixInstance(
                    input_ids_a,
                    input_ids_b,
                    input_mask_a,
                    input_mask_b,
                    valid_a,
                    valid_b,
                    mix_ratio_fn(8),
                    mixed_label,
                    (mix_start_a, mix_end_a, mix_start_b, mix_end_b),
                    label_mask_a,
                )
            )
            print(
                tags_a[begin_a - 1 : end_a - 1],
                tags_b[begin_b - 1 : end_b - 1],
                mixed_label[begin_a:end_a],
            )

            if len(instances) == batch_size:
                yield self._mixed_batch_fn(instances)
                instances = []

    @classmethod
    def _batch_fn(cls, instances: List[Instance]):
        input_ids_batch = torch.tensor(
            [ins.input_ids for ins in instances], dtype=torch.long
        )
        input_mask_batch = torch.tensor(
            [ins.input_mask for ins in instances], dtype=torch.long
        )
        valid_batch = torch.tensor(
            [ins.valid for ins in instances], dtype=torch.long
        )
        label_batch = torch.tensor(
            [ins.labels for ins in instances], dtype=torch.float
        )
        label_mask_batch = torch.tensor(
            [ins.label_mask for ins in instances], dtype=torch.long
        )
        return (
            input_ids_batch,
            input_mask_batch,
            valid_batch,
            label_batch,
            label_mask_batch,
        )

    @classmethod
    def _mixed_batch_fn(cls, instances: List[MixInstance]):
        input_ids_a_batch = torch.tensor(
            [ins.input_ids_a for ins in instances], dtype=torch.long
        )
        input_ids_b_batch = torch.tensor(
            [ins.input_ids_b for ins in instances], dtype=torch.long
        )
        input_mask_a_batch = torch.tensor(
            [ins.input_mask_a for ins in instances], dtype=torch.long
        )
        input_mask_b_batch = torch.tensor(
            [ins.input_mask_b for ins in instances], dtype=torch.long
        )
        valid_a_batch = torch.tensor(
            [ins.valid_a for ins in instances], dtype=torch.long
        )
        valid_b_batch = torch.tensor(
            [ins.valid_b for ins in instances], dtype=torch.long
        )
        mix_ratio_batch = torch.tensor(
            [ins.mix_ratio for ins in instances], dtype=torch.float
        )
        label_batch = torch.tensor(
            [ins.labels for ins in instances], dtype=torch.float
        )
        mixed_idxes = torch.tensor(
            [ins.mix_idx for ins in instances], dtype=torch.long
        )
        label_mask_batch = torch.tensor(
            [ins.label_mask for ins in instances], dtype=torch.long
        )
        return (
            input_ids_a_batch,
            input_ids_b_batch,
            input_mask_a_batch,
            input_mask_b_batch,
            valid_a_batch,
            valid_b_batch,
            mix_ratio_batch,
            label_batch,
            label_mask_batch,
            mixed_idxes,
        )

    @classmethod
    def mix_labels(cls, lab_a, lab_b, mix_ratio, mix_idx, tag_b, label_map):
        begin_a, end_a, begin_b, end_b = mix_idx
        mixed_seq = lab_a.copy()
        padding = np.array(label_map["I-" + tag_b])
        for idx in range(end_a - begin_a):
            idx_a = idx + begin_a
            idx_b = idx + begin_b
            if idx_a < len(lab_a):
                if idx_b < len(lab_b):
                    mixed_seq[idx_a] = lab_a[idx_a] * mix_ratio + lab_b[
                        idx_b
                    ] * (1 - mix_ratio)
                else:
                    mixed_seq[idx_a] = lab_a[idx_a] * mix_ratio + padding * (
                        1 - mix_ratio
                    )
        return mixed_seq

    @classmethod
    def mix_embeddings(cls, embedding_a, embedding_b, mix_ratios, mix_idxes):
        batch_size = embedding_a.shape[0]
        mixed_embeddings = embedding_a.clone()
        for bt in range(batch_size):
            if mix_ratios[bt] > 0:
                mix_ratio = mix_ratios[bt]
                assert (
                    mix_idxes[bt][1] - mix_idxes[bt][0]
                    == mix_idxes[bt][3] - mix_idxes[bt][2]
                )
                for idx in range(mix_idxes[bt][1] - mix_idxes[bt][0]):
                    idx_a = idx + mix_idxes[bt][0]
                    idx_b = idx + mix_idxes[bt][2]
                    if idx_a < len(embedding_a[bt]) and idx_b < len(
                        embedding_b[bt]
                    ):
                        mixed_embeddings[bt][idx_a] = embedding_a[bt][
                            idx_a
                        ] * mix_ratio + embedding_b[bt][idx_b] * (1 - mix_ratio)
        return embedding_a
