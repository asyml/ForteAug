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
import random
from typing import (
    Dict,
    Iterator,
    Optional,
    Union,
    Any,
    Callable,
)

from forte.common.configurable import Configurable
from forte.data.data_pack import DataPack
from forte.common.configuration import Config

from ft.onto.base_ontology import Sentence, EntityMention

from fortex.aug.data.mix_up_dataset import (
    MixUpIterator,
)
from fortex.aug.algorithms.character_flip_op import (
    CharacterFlipOp,
)
from fortex.aug.algorithms.typo_replacement_op import (
    TypoReplacementOp,
)
from fortex.aug.algorithms.back_translation_op import BackTranslationOp

# Abstract class with weighting / random entity
# DEFAULT WEIGHTING SCHEMES
# Assign weights based on number of entities contained in a sentence
def data_pack_entity_weighting(pack: DataPack, sentence: Sentence) -> float:
    entity_num = 0.0
    for _ in pack.get(EntityMention, sentence):
        entity_num += 1
    return entity_num


# return a random entity id in a datapack
def data_pack_random_entity(
    pack: DataPack, sentence: Sentence, num_entity: int
) -> int:
    rand_idx = random.randint(0, num_entity - 1)
    for idx, entity in enumerate(pack.get(EntityMention, sentence)):
        if rand_idx == idx:
            return entity.tid


class DataAugIterator(Configurable):
    def __init__(self, pack_iterator: Iterator[DataPack]):
        self._data_pack_iter: Iterator[DataPack] = pack_iterator
        self._origin_data_pack = list(self._data_pack_iter).copy()
        self._augment_pool = self._origin_data_pack.copy()
        self._augmented_ops = ["original_data"] * len(self._origin_data_pack)
        self.ops = None

    def aug_with(
        self,
        ops: str,
        configs: Union[Config, Dict[str, Any]],
        data_pack_weighting_fn: Optional[
            Callable[[DataPack, Sentence], float]
        ] = None,
        segment_annotate_fn: Optional[
            Callable[[DataPack, Sentence, int], int]
        ] = None,
    ):
        """
        Use `_data_pack_iter` to store the iterator of pre-augmented data,
        restart the iterator of the augmented data after each augmentation.
        """
        self._augmented_ops.extend([ops] * len(self._origin_data_pack))

        if ops == "mix_up":
            # placeholder for mix up
            self._data_pack_iter = MixUpIterator(
                self._data_pack_iter,
                data_pack_weighting_fn,
                segment_annotate_fn,
                configs,
                # train iterator
                self._data_pack_iter,
            )
        else:
            # place holder for ops
            # configs example {"prob": 1.0, other_configs_specific_to_ops: xxx}
            if ops == "char_flip":
                ops = CharacterFlipOp(configs=configs)
            elif ops == "typo_replace":
                ops = TypoReplacementOp(configs=configs)
            elif ops == "back_trans":
                ops = BackTranslationOp(configs=configs)
            else:
                raise ValueError(f"does not support {ops} operations")
            for pack in self._origin_data_pack:
                self._augment_pool.append(ops.perform_augmentation(pack))
            self._data_pack_iter = iter(self._augment_pool)
        self.ops = iter(self._augmented_ops)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.ops), next(self._data_pack_iter)
