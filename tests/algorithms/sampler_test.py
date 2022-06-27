# Copyright 2020 The Forte Authors. All Rights Reserved.
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
Unit tests for distribution sampler.
"""
import unittest

from fortex.aug.utils.sampler import (
    UniformSampler,
    UnigramSampler,
)


class TestSampler(unittest.TestCase):
    def test_unigram_sampler(self):
        word_count = {"apple": 1, "banana": 2, "orange": 3}
        sampler = UnigramSampler(configs={"sampler_data": word_count})
        word = sampler.sample()
        self.assertIn(word, word_count)
        word_prob = {
            "apple": 0.4,
            "banana": 0.4,
            "orange": 0.2,
        }
        sampler = UnigramSampler(configs={"sampler_data": word_prob})
        word = sampler.sample()
        self.assertIn(word, word_prob)

    def test_uniform_sampler(self):
        word_list = ["apple", "banana", "orange"]
        sampler = UniformSampler(configs={"sampler_data": word_list})
        word = sampler.sample()
        self.assertIn(word, word_list)


if __name__ == "__main__":
    unittest.main()
