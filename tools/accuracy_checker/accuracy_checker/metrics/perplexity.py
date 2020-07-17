import numpy as np
from .metric import PerImageEvaluationMetric
from ..representation import LM1BAnnotation, LanguageModelingPrediction
from ..config import ConfigError


class Perplexity(PerImageEvaluationMetric):
    __provider__ = 'softmax_perplexity'
    annotation_types = (LM1BAnnotation, )
    prediction_types = (LanguageModelingPrediction, )

    def configure(self):
        self.sentences_perplexity = []

    def update(self, annotation, prediction):
        sentences_softmax = prediction.logits
        tgts = np.array([[annotation.target_ids[0]]])
        if len(annotation.target_ids) != len(sentences_softmax):
            raise ConfigError("The number of annotation.target_ids is not equal to predictions")
        sum_num = 0.0
        sum_den = 0.0
        sentence_perplexity = 0.0
        target_weights_in = np.ones([1, 1], np.float32)
        for idx, (target_in, softmax) in enumerate(zip(annotation.target_ids, sentences_softmax)):
            tgts = np.array([[target_in]])
            cal_log_perp = CalculatePerplexity(tgts, softmax, softmax)
            log_perp = cal_log_perp._log_perplexity_out()
            if log_perp == -1:
                continue

            sum_num += log_perp * target_weights_in.mean()
            sum_den += target_weights_in.mean()
            if sum_den > 0:
                sentence_perplexity = np.exp(sum_num / sum_den)
        self.sentences_perplexity.append(sentence_perplexity) #only get the last character's perplexity of one sentence
        return sentence_perplexity

    def evaluate(self, annotations, predictions):
        return np.mean(self.sentences_perplexity)


class CalculatePerplexity:
  def __init__(self, targets, biasadd, softmax_out=None):
    self._targets_in = targets
    self._biasadd = biasadd
    self._softmax_out = softmax_out
    self._target_weights_in = np.ones([1, 1], np.float32)
    self._y = 1
    self._start = 0
    self._delta = 1
    self._shape = np.array([-1])
    self._yy = 0.00009999999747378752
    self._values_1 = 800000
    self._y_float = 1.0
    self._default_value = 0.0
    self._shape_1 = np.array([-1, 1], dtype=np.int32)

  def _transpose_1(self):
    array = np.arange(self._start, self._target_weights_in.ndim, self._delta, dtype=np.int32)
    sub_1 = np.subtract(array, (self._target_weights_in.ndim - self._y))
    return np.transpose(self._target_weights_in, sub_1)

  def _Reshape_2(self):
    return self._transpose_1().reshape(self._shape)

  def _transpose_2(self, pack):
    array = np.arange(self._start, pack.ndim, self._delta, dtype=np.int32)
    sub = np.subtract(pack.ndim, self._y, dtype=np.int32)
    sub_1 = np.subtract(array, sub)
    return np.transpose(pack, sub_1)

  def _transpose(self, targets_in):
    array = np.arange(self._start, targets_in.ndim, self._delta, dtype=np.int32)
    sub = np.subtract(targets_in.ndim, self._y, dtype=np.int32)
    sub_1 = np.subtract(array, sub, dtype=np.int32)
    return np.transpose(targets_in, sub_1)

  def _Reshape_1(self):
    return self._transpose(self._targets_in).reshape(self._shape_1)

  def _Reshape_4(self):
    return self._Reshape_1().reshape(self._shape)

  def _SparseToDense_(self, cast_1, cast_2, ones, default_value):
    out_shape = np.asarray(cast_2)
    dst = np.zeros(out_shape, dtype=np.float)
    dst[:] = default_value
    src = np.asarray(cast_1)
    dst[src.T[0], src.T[1]] = ones
    return dst

  def _SparseToDense(self):
    reshape_4 = self._Reshape_4()
    ones = np.full(reshape_4.shape, self._y_float, dtype=np.float32)

    slice_1 = np.array([reshape_4.ndim])[0:1]
    squeeze = np.squeeze(slice_1)
    pack_1 = np.append(squeeze, self._values_1)
    cast_2 = pack_1.astype('int64')

    range_1 = np.arange(self._start, squeeze, self._delta, dtype=np.int32)
    pack = np.stack((range_1, reshape_4))
    transpose_2 = self._transpose_2(pack)
    cast_1 = transpose_2.astype('int64')

    dst = self._SparseToDense_(cast_1, cast_2, ones, self._default_value)
    return dst

  def _softmax(self, x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)

  def _cross_entropy(self, a, y):
    return np.mean(-np.sum(a*np.log(y), axis=1))

  def _SoftmaxCrossEntropyWithLogits(self):
    expect = self._SparseToDense()
    if self._softmax_out is not None:
      lastlayer = self._softmax_out
    else:
      lastlayer = self._softmax(self._biasadd)
    if 0 in lastlayer:
        return -1
    return self._cross_entropy(expect, lastlayer)

  def _log_perplexity_out(self):
    softmax = self._SoftmaxCrossEntropyWithLogits()
    reshape = self._Reshape_2()
    mul = np.multiply(reshape, softmax)
    range_2 = np.arange(self._start, mul.ndim, self._delta, dtype=np.int32)
    Sum = np.sum(mul, axis=tuple(range_2), keepdims=False)

    range_3 = np.arange(self._start, reshape.ndim, self._delta, dtype=np.int32)
    Sum_1 = np.sum(reshape, axis=tuple(range_3), keepdims=False)

    Add = np.add(Sum_1, self._yy)

    return np.divide(Sum, Add)

