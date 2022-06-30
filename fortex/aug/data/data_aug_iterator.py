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
from typing import (
    Dict,
    Iterator,
    # Optional,
    Union,
    Any,
    # Callable,
)

from forte.common.configurable import Configurable
from forte.data.data_pack import DataPack
from forte.common.configuration import Config

# from ft.onto.base_ontology import Sentence

# from fortex.aug.data.mix_up_dataset import (
#     MixUpIterator,
# )
from fortex.aug.algorithms.character_flip_op import (
    CharacterFlipOp,
)
from fortex.aug.algorithms.typo_replacement_op import (
    TypoReplacementOp,
)


class DataAugIterator(Configurable):
    def __init__(self, pack_iterator: Iterator[DataPack]):
        self._data_pack_iter: Iterator[DataPack] = pack_iterator
        self._augmented_ops = []
        self._augment_pool = []

    def aug_with(
        self,
        ops: str,
        configs: Union[Config, Dict[str, Any]],
        # data_pack_weighting_fn: Optional[
        #     Callable[[DataPack, Sentence], float]
        # ] = None,
        # segment_annotate_fn: Optional[
        #     Callable[[DataPack, Sentence, int], int]
        # ] = None,
    ):
        """
        Use `_data_pack_iter` to store the iterator of pre-augmented data,
        restart the iterator of the augmented data after each augmentation.
        """
        self._augmented_ops.append(ops)

        if ops == "mix_up":
            # placeholder for mix up
            # self._data_pack_iter = MixUpIterator(
            #     self._data_pack_iter,
            #     data_pack_weighting_fn,
            #     segment_annotate_fn,
            #     configs=configs,
            # )
            pass
        else:
            # place holder for ops
            # configs example {"prob": 1.0, other_configs_specific_to_ops: xxx}
            if ops == "char_flip":
                ops = CharacterFlipOp(configs=configs)
            elif ops == "typo_replace":
                ops = TypoReplacementOp(configs=configs)
            else:
                raise ValueError(f"does not support {ops} operations")
            for pack in list(self._data_pack_iter):
                if pack not in self._augment_pool:
                    self._augment_pool.append(pack)
                self._augment_pool.append(ops.perform_augmentation(pack))
            self._data_pack_iter = iter(self._augment_pool)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._data_pack_iter)
