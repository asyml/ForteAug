#  Copyright 2022 The Forte Authors. All Rights Reserved.
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
from typing import Dict, Iterator, Type, Optional, Tuple, Callable, Any
import heapq as hq
import random

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from ft.onto.base_ontology import Token, Sentence, EntityMention
from forte.datasets.conll.conll_utils import get_tag
from forte.data.types import DataRequest
from forte.utils import create_import_error_msg

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

__all__ = [
    "DataPackWeightSampler",
]


class DataPackWeightSampler:
    def __init__(
        self,
        pack_iterator: Iterator[DataPack],
        data_pack_weighting_fn: Callable[[DataPack, Sentence], float],
        context_type: Type[Annotation],
        request: Optional[DataRequest] = None,
        skip_k: int = 0,
    ):
        self._get_data_args: Dict = {
            "context_type": context_type,
            "request": request,
            "skip_k": skip_k,
        }
        self._data_pack_iter: Iterator[DataPack] = pack_iterator
        self.data_pack_weighting_fn: Callable[
            [DataPack, Sentence], float
        ] = data_pack_weighting_fn
        self._curr_data_pack: Optional[DataPack] = None
        self._context_iter: Optional[Iterator[Sentence]] = None
        self._priority_queue: Optional[list] = None

    def generate_weighted_samples(self, reservoir_size: int):
        self._curr_data_pack = next(self._data_pack_iter)
        self._context_iter = self._curr_data_pack.get(Sentence)
        while self._curr_data_pack:
            try:
                # data = next(self._instance_iter)
                sent = next(self._context_iter)
                # tid = sent["tid"]
                tag = get_tag(
                    self._curr_data_pack, sent, Token, EntityMention, "ner_type"
                )
            except StopIteration:
                try:
                    self._curr_data_pack = next(self._data_pack_iter)
                except StopIteration:
                    break
                self._context_iter = self._curr_data_pack.get(Sentence)
                # tid = next(self._instance_iter)["tid"]
                sent = next(self._context_iter)
                tag = get_tag(
                    self._curr_data_pack, sent, Token, EntityMention, "ner_type"
                )
            _curr_data_pack_weight = self.data_pack_weighting_fn(
                self._curr_data_pack, sent
            )
            if _curr_data_pack_weight == 0:
                continue
            _curr_data_pack_key = random.random() ** (
                1 / _curr_data_pack_weight
            )
            if self._priority_queue is None:
                self._priority_queue = [
                    (_curr_data_pack_key, (sent, self._curr_data_pack, tag))
                ]
                hq.heapify(self._priority_queue)
            elif len(self._priority_queue) < reservoir_size:
                hq.heappush(
                    self._priority_queue,
                    (_curr_data_pack_key, (sent, self._curr_data_pack, tag)),
                )
            else:
                hq.heappop(self._priority_queue)
                hq.heappush(
                    self._priority_queue,
                    (_curr_data_pack_key, (sent, self._curr_data_pack, tag)),
                )
        return list(
            map(
                lambda x: (
                    x[1][0],
                    x[1][1],
                    self.data_pack_weighting_fn(x[1][1], x[1][0]),
                    x[1][2],
                ),
                self._priority_queue,
            )
        )
