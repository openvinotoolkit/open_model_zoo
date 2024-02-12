"""
Copyright (c) 2018-2024 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pickle  # nosec B403  # disable import-pickle check
from pathlib import Path
import numpy as np

from ..representation import ClassificationAnnotation
from ..config import NumberField, StringField, PathField, BoolField
from ..utils import get_path
from .format_converter import BaseFormatConverter
from .format_converter import ConverterReturn


class DataIterator:
    def __init__(self, source,
                 uid_voc,
                 mid_voc,
                 cat_voc,
                 item_info,
                 reviews_info,
                 batch_size=128,
                 maxlen=100):

        self.source = open(source, 'r', encoding='UTF-8') # pylint: disable=R1732
        self.source_dicts = []
        for source_dict in [uid_voc, mid_voc, cat_voc]:
            with open(source_dict, 'rb') as source_content:
                # disable pickle check
                self.source_dicts.append(pickle.load(source_content, encoding='UTF-8'))  # nosec B301

        with open(item_info, "r", encoding='UTF-8') as f_meta:
            meta_map = {}
            for line in f_meta:
                arr = line.strip().split("\t")
                if arr[0] not in meta_map:
                    meta_map[arr[0]] = arr[1]
        self.meta_id_map = {}
        for key, val in meta_map.items():
            if key in self.source_dicts[1]:
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx

        with open(reviews_info, "r", encoding='UTF-8') as f_review:
            self.mid_list_for_random = []
            for line in f_review:
                arr = line.strip().split("\t")
                tmp_idx = 0
                if arr[1] in self.source_dicts[1]:
                    tmp_idx = self.source_dicts[1][arr[1]]
                self.mid_list_for_random.append(tmp_idx)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])

        self.shuffle = False

        self.source_buffer = []
        self.k = batch_size * 20

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for _ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

            # sort by  history behavior length
            his_length = np.array([len(s[4].split("")) for s in self.source_buffer])
            tidx = his_length.argsort()

            _sbuf = [self.source_buffer[i] for i in tidx]
            self.source_buffer = _sbuf

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
                mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
                cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0
                tmp = []
                for fea in ss[4].split(""):
                    m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                    tmp.append(m)
                mid_list = tmp

                tmp1 = []
                for fea in ss[5].split(""):
                    c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                    tmp1.append(c)
                cat_list = tmp1

                # read from source file and map to word index

                source.append([uid, mid, cat, mid_list, cat_list])
                target.append([float(ss[0]), 1-float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target


class AmazonProductData(BaseFormatConverter):

    __provider__ = 'amazon_product_data'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            "data_dir": PathField(optional=False, is_directory=True, check_exists=True,
                                  description="Dataset root"),
            "preprocessed_dir": PathField(optional=False, is_directory=True, check_exists=True,
                                          description="Preprocessed dataset location"),
            "separator": StringField(optional=True, default='#',
                                     description="Separator between input identifier and file identifier"),
            "test_data": StringField(optional=True, default='local_test_splitByUser',
                                     description="test data filename."),
            "batch": NumberField(optional=True, default=1, description="Batch size", value_type=int),
            "max_len": NumberField(optional=True, default=None, description="Maximum sequence length", value_type=int),
            "subsample_size": NumberField(
                optional=True, default=0, description="Number of sentences to process", value_type=int
            ),
            "uid_voc": StringField(optional=True, default='uid_voc.pkl', description="uid_voc filename"),
            "mid_voc": StringField(optional=True, default='mid_voc.pkl', description="mid_voc filename"),
            "cat_voc": StringField(optional=True, default='cat_voc.pkl', description="cat_voc filename"),
            "item_info": StringField(optional=True, default='item-info', description="item info filename"),
            "reviews_info": StringField(optional=True, default='reviews-info', description="reviews info filename"),
            "mid_his_batch": StringField(optional=True, default="Inputs/mid_his_batch_ph",
                                         description="mid_his_batch input identifier"),
            "cat_his_batch": StringField(optional=True, default="Inputs/cat_his_batch_ph",
                                         description="cat_his_batch input identifier"),
            "uid_batch": StringField(optional=True, default="Inputs/uid_batch_ph",
                                     description="uid_batch input identifier"),
            "mid_batch": StringField(optional=True, default="Inputs/mid_batch_ph",
                                     description="mid_batch input identifier"),
            "cat_batch": StringField(optional=True, default="Inputs/cat_batch_ph",
                                     description="cat_batch input identifier"),
            "mask": StringField(optional=True, default="Inputs/mask",
                                description="mask input identifier"),
            "seq_len": StringField(optional=True, default="Inputs / seq_len_ph",
                                   description="seq_len input identifier"),
            "skip_dump": BoolField(optional=True, default=True, description='Annotate without saving features')
        })

        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.test_data = self.get_value_from_config('test_data')
        self.separator = self.get_value_from_config('separator')
        self.preprocessed_dir = self.get_value_from_config('preprocessed_dir')
        self.uid_voc = self.get_value_from_config('uid_voc')
        self.mid_voc = self.get_value_from_config('mid_voc')
        self.cat_voc = self.get_value_from_config('cat_voc')
        self.item_info = self.get_value_from_config('item_info')
        self.reviews_info = self.get_value_from_config('reviews_info')
        self.mid_his_batch = self.get_value_from_config('mid_his_batch')
        self.cat_his_batch = self.get_value_from_config('cat_his_batch')
        self.cat_batch = self.get_value_from_config('cat_batch')
        self.mid_batch = self.get_value_from_config('mid_batch')
        self.uid_batch = self.get_value_from_config('uid_batch')
        self.mask = self.get_value_from_config('mask')
        self.seq_len = self.get_value_from_config('seq_len')
        self.skip_dump = self.get_value_from_config('skip_dump')
        self.batch = self.get_value_from_config('batch')
        self.max_len = self.get_value_from_config('max_len')
        self.subsample_size = self.get_value_from_config('subsample_size')

    @staticmethod
    def prepare_data(source, target, maxlen=None):
        # x: a list of sentences
        lengths_x = [len(s[4]) for s in source]
        seqs_mid = [inp[3] for inp in source]
        seqs_cat = [inp[4] for inp in source]
        if maxlen is not None:
            new_seqs_mid = []
            new_seqs_cat = []
            new_lengths_x = []
            for l_x, inp in zip(lengths_x, source):
                if l_x > maxlen:
                    new_seqs_mid.append(inp[3][l_x - maxlen:])
                    new_seqs_cat.append(inp[4][l_x - maxlen:])
                    new_lengths_x.append(maxlen)
                else:
                    new_seqs_mid.append(inp[3])
                    new_seqs_cat.append(inp[4])
                    new_lengths_x.append(l_x)
            lengths_x = new_lengths_x
            seqs_mid = new_seqs_mid
            seqs_cat = new_seqs_cat

        n_samples = len(seqs_mid)
        maxlen_x = np.max(lengths_x)
        maxlen_x = max(maxlen, maxlen_x) if maxlen is not None else maxlen_x

        mid_his = np.zeros((n_samples, maxlen_x)).astype('int64')
        cat_his = np.zeros((n_samples, maxlen_x)).astype('int64')
        mid_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
        for idx, [s_x, s_y] in enumerate(zip(seqs_mid, seqs_cat)):
            mid_mask[idx, :lengths_x[idx]] = 1.
            mid_his[idx, :lengths_x[idx]] = s_x
            cat_his[idx, :lengths_x[idx]] = s_y

        uids = np.array([inp[0] for inp in source])
        mids = np.array([inp[1] for inp in source])
        cats = np.array([inp[2] for inp in source])

        return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x)

    def convert(self, check_content=False, **kwargs):
        test_file = get_path(self.data_dir / self.test_data, is_directory=False)
        uid_voc = get_path(self.data_dir / self.uid_voc, is_directory=False)
        mid_voc = get_path(self.data_dir / self.mid_voc, is_directory=False)
        cat_voc = get_path(self.data_dir / self.cat_voc, is_directory=False)
        item_info = get_path(self.data_dir / self.item_info, is_directory=False)
        reviews_info = get_path(self.data_dir / self.reviews_info, is_directory=False)

        test_data = DataIterator(str(test_file), str(uid_voc), str(mid_voc), str(cat_voc), str(item_info),
                                 str(reviews_info), self.batch, self.max_len)
        preprocessed_folder = Path(self.preprocessed_dir)
        if not self.skip_dump and not preprocessed_folder.exists():
            preprocessed_folder.mkdir(exist_ok=True, parents=True)

        input_folder = preprocessed_folder / "bs{}".format(self.batch) / 'input'

        if not input_folder.exists() and not self.skip_dump:
            input_folder.mkdir(parents=True)

        annotations = []

        subfolder = 0
        filecnt = 0
        iteration = 0

        for src, tgt in test_data:
            uids, mids, cats, mid_his, cat_his, mid_mask, gt, sl = self.prepare_data(src, tgt, maxlen=self.max_len)
            c_input = input_folder / "{:02d}".format(subfolder)
            c_input = c_input / "{:06d}.npz".format(iteration)

            if not self.skip_dump:
                if not c_input.parent.exists():
                    c_input.parent.mkdir(parents=True)

                sample = {
                    self.mid_his_batch: mid_his,
                    self.cat_his_batch: cat_his,
                    self.uid_batch: uids,
                    self.mid_batch: mids,
                    self.cat_batch: cats,
                    self.mask: mid_mask,
                    self.seq_len: sl
                }
                np.savez_compressed(str(c_input), **sample)

            filecnt += 1
            filecnt %= 0x100

            subfolder = subfolder + 1 if filecnt == 0 else subfolder

            c_file = str(c_input.relative_to(preprocessed_folder))
            identifiers = [
                "{}_{}{}".format(self.mid_his_batch, self.separator, c_file),
                "{}_{}{}".format(self.cat_his_batch, self.separator, c_file),
                "{}_{}{}".format(self.uid_batch, self.separator, c_file),
                "{}_{}{}".format(self.mid_batch, self.separator, c_file),
                "{}_{}{}".format(self.cat_batch, self.separator, c_file),
                "{}_{}{}".format(self.mask, self.separator, c_file),
                "{}_{}{}".format(self.seq_len, self.separator, c_file),
            ]
            if not self.subsample_size or (self.subsample_size and (iteration < self.subsample_size)):
                annotations.append(ClassificationAnnotation(identifiers, gt[:, 0].tolist()))
            iteration += 1
            if self.subsample_size and (iteration > self.subsample_size):
                break

        return ConverterReturn(annotations, None, None)
