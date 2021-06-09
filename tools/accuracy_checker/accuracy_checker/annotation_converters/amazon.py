"""
Copyright (c) 2018-2021 Intel Corporation

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

import pickle, random
from pathlib import Path
import numpy as np

from ..representation import ClassificationAnnotation
from ..config import NumberField, StringField, PathField, BoolField
from ..utils import get_path
from .format_converter import BaseFormatConverter
from .format_converter import ConverterReturn

def unicode_to_utf8(d):
    return dict((key, value) for (key,value) in d.items())

def load_dict(filename):
    with open(filename, 'rb') as f:
        return unicode_to_utf8(pickle.load(f))
    # try:
    #     with open(filename, 'rb') as f:
    #         return unicode_to_utf8(json.load(f))
    # except:
    #     with open(filename, 'rb') as f:
    #         return unicode_to_utf8(pickle.load(f))


# def fopen(filename, mode='r'):
#     if filename.endswith('.gz'):
#         return gzip.open(filename, mode)
#     return open(filename, mode)


class DataIterator:

    def __init__(self, source,
                 uid_voc,
                 mid_voc,
                 cat_voc,
                 item_info,
                 reviews_info,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 sort_by_length=True,
                 max_batch_size=20,
                 minlen=None):
        self.source = open(source, 'r')
        self.source_dicts = []
        for source_dict in [uid_voc, mid_voc, cat_voc]:
            self.source_dicts.append(load_dict(source_dict))

        f_meta = open(item_info, "r")
        meta_map = {}
        for line in f_meta:
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1]
        self.meta_id_map ={}
        for key in meta_map:
            val = meta_map[key]
            if key in self.source_dicts[1]:
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx

        f_review = open(reviews_info, "r")
        self.mid_list_for_random = []
        for line in f_review:
            arr = line.strip().split("\t")
            tmp_idx = 0
            if arr[1] in self.source_dicts[1]:
                tmp_idx = self.source_dicts[1][arr[1]]
            self.mid_list_for_random.append(tmp_idx)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])

        self.shuffle = False
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat

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
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

            # sort by  history behavior length
            if self.sort_by_length:
                his_length = np.array([len(s[4].split("")) for s in self.source_buffer])
                tidx = his_length.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()

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

                #if len(mid_list) > self.maxlen:
                #    continue
                if self.minlen != None:
                    if len(mid_list) >= self.minlen:
                        continue
                if self.skip_empty and (not mid_list):
                    continue

                noclk_mid_list = []
                noclk_cat_list = []
                for pos_mid in mid_list:
                    noclk_tmp_mid = []
                    noclk_tmp_cat = []
                    noclk_index = 0
                    while True:
                        noclk_mid_indx = random.randint(0, len(self.mid_list_for_random)-1)
                        noclk_mid = self.mid_list_for_random[noclk_mid_indx]
                        if noclk_mid == pos_mid:
                            continue
                        noclk_tmp_mid.append(noclk_mid)
                        noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
                        noclk_index += 1
                        if noclk_index >= 5:
                            break
                    noclk_mid_list.append(noclk_tmp_mid)
                    noclk_cat_list.append(noclk_tmp_cat)
                source.append([uid, mid, cat, mid_list, cat_list, noclk_mid_list, noclk_cat_list])
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
            "batch": NumberField(optional=True, default=1, description="Batch size"),
            "max_len": NumberField(optional=True, default=100, description="Maximum sequence length"),
            "subsample_size": NumberField(optional=True, default=0, description="Number of sentences to process"),
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
        self.batch = int(self.get_value_from_config('batch'))
        self.max_len = int(self.get_value_from_config('max_len'))
        self.subsample_size = int(self.get_value_from_config('subsample_size'))

    @staticmethod
    def prepare_data(input, target, maxlen=None, return_neg=False):
        # x: a list of sentences
        lengths_x = [len(s[4]) for s in input]
        seqs_mid = [inp[3] for inp in input]
        seqs_cat = [inp[4] for inp in input]
        noclk_seqs_mid = [inp[5] for inp in input]
        noclk_seqs_cat = [inp[6] for inp in input]
        if maxlen is not None:
            new_seqs_mid = []
            new_seqs_cat = []
            new_noclk_seqs_mid = []
            new_noclk_seqs_cat = []
            new_lengths_x = []
            for l_x, inp in zip(lengths_x, input):
                if l_x > maxlen:
                    new_seqs_mid.append(inp[3][l_x - maxlen:])
                    new_seqs_cat.append(inp[4][l_x - maxlen:])
                    new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                    new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                    new_lengths_x.append(maxlen)
                else:
                    new_seqs_mid.append(inp[3])
                    new_seqs_cat.append(inp[4])
                    new_noclk_seqs_mid.append(inp[5])
                    new_noclk_seqs_cat.append(inp[6])
                    new_lengths_x.append(l_x)
            lengths_x = new_lengths_x
            seqs_mid = new_seqs_mid
            seqs_cat = new_seqs_cat
            noclk_seqs_mid = new_noclk_seqs_mid
            noclk_seqs_cat = new_noclk_seqs_cat

            if len(lengths_x) < 1:
                return None, None, None, None

        n_samples = len(seqs_mid)
        maxlen_x = np.max(lengths_x)
        neg_samples = len(noclk_seqs_mid[0][0])

        mid_his = np.zeros((n_samples, maxlen_x)).astype('int64')
        cat_his = np.zeros((n_samples, maxlen_x)).astype('int64')
        noclk_mid_his = np.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
        noclk_cat_his = np.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
        mid_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
        for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
            mid_mask[idx, :lengths_x[idx]] = 1.
            mid_his[idx, :lengths_x[idx]] = s_x
            cat_his[idx, :lengths_x[idx]] = s_y
            noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
            noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

        uids = np.array([inp[0] for inp in input])
        mids = np.array([inp[1] for inp in input])
        cats = np.array([inp[2] for inp in input])

        if return_neg:
            return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(
                lengths_x), noclk_mid_his, noclk_cat_his

        else:
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
        n_uid, n_mid, n_cat = test_data.get_n()
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
            uids, mids, cats, mid_his, cat_his, mid_mask, gt, sl, _, _ = self.prepare_data(src, tgt, return_neg=True)
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

        return ConverterReturn(annotations, None, None)
