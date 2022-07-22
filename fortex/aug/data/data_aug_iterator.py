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
    Optional,
    Union,
    Any,
    Callable,
)
from itertools import chain

from forte.common.configurable import Configurable
from forte.data.data_pack import DataPack
from forte.common.configuration import Config
from forte.data.readers import SST2Reader
from forte.pipeline import Pipeline

from ft.onto.base_ontology import Sentence

from fortex.aug.data.mix_up_dataset import (
    MixUpIterator,
    # MixupDataProcessor
)
from fortex.aug.algorithms.character_flip_op import (
    CharacterFlipOp,
)
from fortex.aug.algorithms.typo_replacement_op import (
    TypoReplacementOp,
)
from fortex.aug.algorithms.back_translation_op import BackTranslationOp

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

def IterPrep(task: str, data_path: str):
    pipeline = Pipeline()
    if task == 'sst':
        reader = SST2Reader()
    else:
        raise ValueError('Does not support this task now.')
    pipeline.set_reader(reader)
    pipeline.initialize()
    return pipeline.process_dataset(data_path, 5)


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
        if ops == "mix_up":
            self._augmented_ops.extend(
                [ops]
                * len(
                    list(
                        MixUpIterator(
                            iter(self._origin_data_pack),
                            data_pack_weighting_fn,
                            segment_annotate_fn,
                            configs,
                            train_iterator=lambda: iter(self._origin_data_pack),
                        )
                    )
                )
            )
            # placeholder for mix up
            # self.data_processor = MixupDataProcessor()
            self._data_pack_iter = chain(
                iter(self._augment_pool),
                # self.data_processor.get_mixed_data_batch(
                #     MixUpIterator(
                #         iter(self._origin_data_pack),
                #         data_pack_weighting_fn,
                #         segment_annotate_fn,
                #         configs,
                #         train_iterator=lambda: iter(self._origin_data_pack),
                #     ),
                #     tokenizer,
                #     8,
                #     {0: 1, 1: 2, "[CLS]": 0, "[SEP]": 3},
                #     4,
                #     128,
                # ),
                MixUpIterator(
                    iter(self._origin_data_pack),
                    data_pack_weighting_fn,
                    segment_annotate_fn,
                    configs,
                    train_iterator=lambda: iter(self._origin_data_pack),
                ),
            )
        else:
            self._augmented_ops.extend([ops] * len(self._origin_data_pack))
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
