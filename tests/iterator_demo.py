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
import unittest
from forte.data.data_pack import DataPack
from ft.onto.base_ontology import Token
from fortex.aug.data.data_aug_iterator import DataAugIterator


class TestIterator(unittest.TestCase):
    def setUp(self):
        data_pack = DataPack()
        data_pack.set_text("auxiliary colleague apparent")
        token_1 = Token(data_pack, 0, 9)
        token_2 = Token(data_pack, 10, 19)
        token_3 = Token(data_pack, 20, 28)
        data_pack.add_entry(token_1)
        data_pack.add_entry(token_2)
        data_pack.add_entry(token_3)
        self.test = DataAugIterator(iter([data_pack]))

    def test_iterator(self):
        """
        Usgae:
            DataAugIterator.aug_with("ops1", configs)
            DataAugIterator.aug_with("ops2", configs)
            ....
        Iterate:
            next(DataAugIterator)

        Here, we do two ops. The first op gives two datapacks, the original one
        and the augmented one. The second op augments these two datapacks
        respectively, which gives four datapacks eventually.
        """
        random.seed(42)
        configs1 = {
            "prob": 0.3,
        }
        configs2 = {
            "prob": 1.0,
            "typo_generator": "uniform",
        }

        self.test.aug_with("typo_replace", configs2)
        self.assertEqual(len(self.test._augment_pool), 2)

        self.test.aug_with("char_flip", configs1)
        self.assertEqual(len(self.test._augment_pool), 4)

        origin_token = list(next(self.test).get("ft.onto.base_ontology.Token"))
        expected_tokens = [
            ["auxiliary"],
            ["colleague"],
            ["apparent"],
        ]
        for orig, exp in zip(origin_token, expected_tokens):
            self.assertIn(orig.text, exp)

        # typo aug
        augmented_tokens = list(
            next(self.test).get("ft.onto.base_ontology.Token")
        )
        expected_tokens = [
            ["auxilliary"],
            ["collegue"],
            ["aparent"],
        ]
        for aug, exp in zip(augmented_tokens, expected_tokens):
            self.assertIn(aug.text, exp)

        # char_flip aug
        augmented_tokens = list(
            next(self.test).get("ft.onto.base_ontology.Token")
        )
        expected_tokens = [
            ["au)(1|_iary"],
            ["co1l3a9ue"],
            ["app4rent"],
        ]
        for aug, exp in zip(augmented_tokens, expected_tokens):
            self.assertIn(aug.text, exp)

        # typo + char_flip aug
        augmented_tokens = list(
            next(self.test).get("ft.onto.base_ontology.Token")
        )
        expected_tokens = [
            ["auxil7iary"],
            ["col!3gu3"],
            ["apaI2ent"],
        ]
        for aug, exp in zip(augmented_tokens, expected_tokens):
            self.assertIn(aug.text, exp)

        self.assertEqual(
            self.test._augmented_ops, ["typo_replace", "char_flip"]
        )


if __name__ == "__main__":
    unittest.main()
