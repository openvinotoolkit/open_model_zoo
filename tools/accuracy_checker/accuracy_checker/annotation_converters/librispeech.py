"""
Copyright (c) 2019 Intel Corporation

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

from pathlib import Path
import numpy as np
import re, math, wave, time, codecs

from ..representation import CharacterRecognitionAnnotation
from ..config import NumberField, StringField, PathField, BoolField
from .format_converter import DirectoryBasedAnnotationConverter
from .format_converter import ConverterReturn
from scipy.fftpack import dct
from scipy.signal import stft

class LibrispeechConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'librispeech'
    annotation_types = (CharacterRecognitionAnnotation, )
    HZ2MEL = 1127.0

    # @classmethod
    # def parameters(cls):
    #     parameters = super().parameters()
    #     parameters.update({
    #         "feature_win_len": NumberField(optional=True, default=32, description="feature extraction audio window length in milliseconds"),
    #         "feature_win_step": NumberField(optional=True, default=20, description="feature extraction window step length in milliseconds"),
    #         "audio_sample_rate": NumberField(optional=True, default=16000, description='sample rate value expected by model'),
    #         "magnitude_squared": BoolField(optional=True, default=True, description="Return spectrum magnitude as square of absolute value."),
    #         "filterbank_channel_count": NumberField(optional=True, default=40, description='number of channels in filter bank'),
    #         "lower_frequency_limit": NumberField(optional=True, default=20, description='filter passband lower boundary'),
    #         "upper_frequency_limit": NumberField(optional=True, default=4000, description='filter passband upper boundary'),
    #         "preemphasis": NumberField(optional=True, default=0.97, description='pre-emphasis filter coefficient'),
    #         "n_hidden": NumberField(optional=True, default=2048, description="layer width to use when initialising layers"),
    #         # "n_input": NumberField(optional=True, default=26, description="Number of model inputs"),
    #         "n_context": NumberField(optional=True, default=9, description="Number of sequential cepstral samples in model context"),
    #         "numcep": NumberField(optional=True, default=26, description="Number of cepstral coefs, model input width"),
    #         "append_energy": BoolField(optional=True, default=True, description="Replace zero cepstral coefficient with total frame energy"),
    #         "numfilt": NumberField(optional=True, default=26, description="Number of filters"),
    #         "beamwidth": NumberField(optional=True, default=10, description="Beam width"),
    #         "hanning": BoolField(optional=True, default=True, description="Apply hanning window before spectrogram calculation."),
    #         "preprocessed_dir": PathField(optional=False, is_directory=True, check_exists=True,
    #                                       description="Preprocessed dataset location")
    #     })

        # return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        # self.winlen = self.get_value_from_config('feature_win_len')
        # self.stride = self.get_value_from_config('feature_win_step')
        # self.sample_rate = self.get_value_from_config('audio_sample_rate')
        # self.n_hidden = self.get_value_from_config('n_hidden')
        # self.n_context = self.get_value_from_config('n_context')
        # # self.output = self.get_value_from_config('output')
        # self.preprocessed_dir = self.get_value_from_config('preprocessed_dir')
        # self.filterbank_channel_count = self.get_value_from_config('filterbank_channel_count')
        # self.lower_frequency_limit = self.get_value_from_config('lower_frequency_limit')
        # self.upper_frequency_limit = self.get_value_from_config('upper_frequency_limit')
        # self.numcep = self.get_value_from_config('numcep')
        # self.numfilt = self.get_value_from_config('numfilt')
        # self.hanning = self.get_value_from_config('hanning')
        # self.beamwidth = self.get_value_from_config('beamwidth')
        # self.preemph = self.get_value_from_config('preemphasis')
        # self.append_energy = self.get_value_from_config('append_energy')
        # self.magnitude_squared = self.get_value_from_config('magnitude_squared')
        # self.frame_len = int(self.sample_rate * self.winlen / 1000.)
        # self.frame_step = int(self.sample_rate * self.stride / 1000.)
        # p2 = np.floor(np.log2(self.frame_len))
        # self.nfft = 1 << p2.astype(np.int32)
        # self.window = np.hanning(self.frame_len) if self.hanning else np.ones(self.frame_len)
        #
        # self.filterbank = MFCC_Filterbank(self.sample_rate,
        #                                   self.filterbank_channel_count,
        #                                   self.lower_frequency_limit,
        #                                   self.upper_frequency_limit,
        #                                   self.nfft // 2 + 1)
        # self.filterbank.initialize()
        #
        # self.dct = DCT(self.filterbank_channel_count, self.numcep)
        # self.dct.initialize()

    def convert(self, check_content=False, **kwargs):

        pattern = re.compile(r'([0-9\-]+)\s+(.+)')
        annotations = []
        data_folder = Path(self.data_dir)
        txts = list(data_folder.glob('**/*.txt'))
        for txt in txts:
            content = open(txt).readlines()
            for line in content:
                res = pattern.search(line)
                if res:
                    name = res.group(1)
                    transcript = res.group(2)
                # name, transcript = line.split(' ')
                    fname = txt.parent / name
                    fname = fname.with_suffix('.wav')

                    # features = self.convertWave(fname)

                annotations.append(CharacterRecognitionAnnotation(str(fname), transcript))

        # after = time.time()
        # print ("time elapsed: {}/{}, {}".format(after - before, cnt, (after-before)/cnt))

        return ConverterReturn(annotations, None, None)

    def convertWave(self, fname):

        self.filterbank = MFCC_Filterbank(self.sample_rate,
                                          self.filterbank_channel_count,
                                          self.lower_frequency_limit,
                                          self.upper_frequency_limit,
                                          self.nfft // 2 + 1)
        self.filterbank.initialize()

        self.dct = DCT(self.filterbank_channel_count, self.numcep)
        self.dct.initialize()

        # Speech feature extration
        _wave = wave.open(str(fname), 'rb')
        rate = _wave.getframerate()
        features = None
        if rate == self.sample_rate:
            audio = np.frombuffer(_wave.readframes(_wave.getnframes()), dtype=np.dtype('<h'))
            audio = audio / np.float32(0x8000)  # normalize to -1 to 1, int 16 to float32
            # audio = audio.reshape(-1, 1)

            # pspec = self.audio_spectrogram(audio, self.frame_len, self.frame_step, True)

            frames = self.framesig(audio)
            pspec = np.absolute(np.fft.rfft(frames, self.nfft))
            pspec = np.square(pspec)

            mfcc_feat = self.mfcc(pspec.reshape(1, pspec.shape[0], -1))#, self.sample_rate, self.numcep)

            # mfcc_feat, frames, pspec, fb = self.mfcc(audio)
            # d_mfcc_feat = self.delta(mfcc_feat, 2)
            # fbank_feat = self.logfbank(audio)

            num_strides, _ = mfcc_feat.shape
            empty_context = np.zeros((self.n_context, self.numcep), dtype=mfcc_feat.dtype)
            mfcc_feat = np.concatenate((empty_context, mfcc_feat, empty_context))

            # num_strides -= self.n_context * 2

            # num_strides = len(mfcc_feat) - (self.n_context * 2)
            # Create a view into the array with overlapping strides of size
            # numcontext (past) + 1 (present) + numcontext (future)
            window_size = 2 * self.n_context + 1
            features = np.lib.stride_tricks.as_strided(
                mfcc_feat,
                (num_strides, window_size, self.numcep),
                (mfcc_feat.strides[0], mfcc_feat.strides[0], mfcc_feat.strides[1]),
                writeable=False)
            # features = np.zeros([num_strides, window_size, self.numcep], dtype = mfcc_feat.dtype)
            # for i in range(num_strides - 1):
            #     features[i, ...] = mfcc_feat[i + window_size, ...]

        _wave.close()
        return features, None, None, pspec, None

    def mfcc_disabled(self, data):

        # ,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
        #      nfilt=26,nfft=None,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
        #      winfunc=lambda x:numpy.ones((x,))):
        """Compute MFCC features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the sample rate of the signal we are working with, in Hz.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param numcep: the number of cepstrum to return, default 13
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is None, which uses the calculate_nfft function to choose the smallest size that does not drop sample data.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
        :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
        :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
        """
        feat, energy, frames, pspec, fb = self.fbank(data)
        # ,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
        feat = np.log(feat)
        # feat = dct(feat, type=2, axis=1, norm='ortho')[:, :self.numcep]
        num_mel_bins = feat.shape[-1]
        feat = dct(feat, type=2, axis=1)[:, :self.numcep]
        # dct2 = dct_ops.dct(log_mel_spectrograms, type=2)
        # return dct2 * math_ops.rsqrt(
        #     math_ops.cast(num_mel_bins, dtypes.float32) * 2.0)
        feat /= np.sqrt(2 * num_mel_bins)
        # feat = lifter(feat,ceplifter)
        if self.append_energy: feat[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy
        return feat, frames, pspec, fb

    def mfcc(self, spectrogram): # sample_rate, dct_coefficient_count):
        audio_channels, spectrogram_samples, spectrogram_channels = spectrogram.shape
        # print (spectrogram.shape)
        kFilterbankFloor = 1e-12
        # filterbank_channel_count = 40

        mfcc_output = np.zeros((spectrogram_samples, self.numcep))
        for i in range(audio_channels):
            # start_index, end_index, weights, band_mapper = \
            #     self.mfcc_mel_filiterbank_init(self.sample_rate, spectrogram_channels)
            # cosine = self.dct_init(self.filterbank_channel_count, self.numcep)
            for j in range(spectrogram_samples):
                mfcc_input = spectrogram[i, j, :]

                # mel_filiter = self.mfcc_mel_filiterbank_compute(mfcc_input, spectrogram_channels,
                #                                            start_index, end_index,
                #                                            weights, band_mapper)
                filtered = self.filterbank.compute(mfcc_input)

                # for k in range(filtered.shape[0]):
                #     val = filtered[k]
                #     if (val < kFilterbankFloor):
                #         val = kFilterbankFloor
                #
                #     filtered[k] = np.log(val)

                filtered = np.where(filtered < kFilterbankFloor, kFilterbankFloor, filtered)
                filtered = np.log(filtered)

                # mfcc_output[j, :] =self.dct_compute(mel_filiter,
                #                                 self.filterbank_channel_count,
                #                                 self.numcep,
                #                                 cosine)

                mfcc_output[j, :] =self.dct.compute(filtered)

        return mfcc_output

    # def mfcc_mel_filiterbank_init(self, sample_rate, input_length):
    #     # init
    #     # filterbank_channel_count_ = 40
    #     # lower_frequency_limit_ = 20
    #     # upper_frequency_limit_ = 4000
    #
    #     def freq2mel(freq):
    #         return 1127.0 * np.log1p(freq / 700)
    #
    #     center_frequencies = np.zeros((self.filterbank_channel_count + 1))
    #     mel_low = freq2mel(self.lower_frequency_limit)
    #     mel_hi = freq2mel(self.upper_frequency_limit)
    #     mel_span = mel_hi - mel_low
    #     mel_sapcing = mel_span / (self.filterbank_channel_count + 1)
    #
    #     for i in range((self.filterbank_channel_count + 1)):
    #         center_frequencies[i] = mel_low + (mel_sapcing * (1 + i))
    #
    #     hz_per_sbin = 0.5 * sample_rate / (input_length - 1)
    #     start_index = int(1.5 + (self.lower_frequency_limit / hz_per_sbin))
    #     end_index = int(self.upper_frequency_limit / hz_per_sbin)
    #
    #     band_mapper = np.zeros(input_length)
    #     channel = 0
    #
    #     for i in range(input_length):
    #         melf = freq2mel(i * hz_per_sbin)
    #
    #         if ((i < start_index) or (i > end_index)):
    #             band_mapper[i] = -2
    #         else:
    #             while ((center_frequencies[int(channel)] < melf) and
    #                    (channel < self.filterbank_channel_count)):
    #                 channel += 1
    #             band_mapper[i] = channel - 1
    #
    #     weights = np.zeros(input_length)
    #     for i in range(input_length):
    #         channel = band_mapper[i]
    #         if ((i < start_index) or (i > end_index)):
    #             weights[i] = 0.0
    #         else:
    #             if (channel >= 0):
    #                 weights[i] = ((center_frequencies[int(channel) + 1] - freq2mel(i * hz_per_sbin)) /
    #                               (center_frequencies[int(channel) + 1] - center_frequencies[int(channel)]))
    #             else:
    #                 weights[i] = ((center_frequencies[0] - freq2mel(i * hz_per_sbin)) /
    #                               (center_frequencies[0] - mel_low))
    #
    #     return start_index, end_index, weights, band_mapper
    #
    # def mfcc_mel_filiterbank_compute(self, mfcc_input, input_length, start_index, end_index, weights, band_mapper):
    #     # filterbank_channel_count_ = 40
    #     # Compute
    #     output_channels = np.zeros(self.filterbank_channel_count)
    #     for i in range(start_index, (end_index + 1)):
    #         spec_val = np.sqrt(mfcc_input[i])
    #         weighted = spec_val * weights[i]
    #         channel = band_mapper[i]
    #         if (channel >= 0):
    #             output_channels[int(channel)] += weighted
    #         channel += 1
    #         if (channel < self.filterbank_channel_count):
    #             output_channels[int(channel)] += (spec_val - weighted)
    #
    #     return output_channels

    # def audio_spectrogram(self, samples, window_size, stride, magnitude_squared):
    #     # window_size, #=(16000 * (32 / 1000)), #Config.audio_window_samples,
    #     # stride, # =(16000 * (20 / 1000)), #(Config.audio_step_samples,
    #     # magnitude_squared) : # =True) :
    #     if (len(samples.shape) != 2):
    #         print("input must be 2-dimensional")
    #
    #     window_size = int(window_size)
    #     stride = int(stride)
    #
    #     sample_count = samples.shape[0]
    #     channel_count = samples.shape[1]  # == 1
    #
    #     def output_frequency_channels(n):
    #         _log = np.floor(np.log2(n))
    #         _log = _log if ((n == (n & ~(n - 1)))) else _log + 1
    #
    #         fft_length = 1 << _log.astype(np.int32)
    #
    #         return 1 + fft_length / 2, fft_length.astype(np.int32)
    #
    #     output_width, fft_length = output_frequency_channels(window_size)
    #     output_width = output_width.astype(np.int32)
    #
    #     length_minus_windows = sample_count - window_size
    #
    #     output_height = 0 if length_minus_windows < 0 else (1 + (length_minus_windows / stride))
    #     output_height = int(output_height)
    #     output_slices = channel_count
    #
    #     __output = np.zeros((output_slices, output_height, output_width))
    #     hann_window = np.hanning(window_size)
    #
    #     for i in range(channel_count):
    #         input_for_channel = samples[:, i]
    #
    #         input_for_compute = np.zeros(stride)
    #         spectrogram = np.zeros((output_height, output_width))
    #
    #         fft_input_output = np.zeros(fft_length)
    #
    #         for j in range(output_height):
    #             start = j * stride
    #             end = start + window_size
    #             if (end < sample_count):
    #                 input_for_compute = input_for_channel[start: end]
    #
    #                 fft_input_output[0:window_size] = input_for_compute * hann_window
    #                 fft_input_output[window_size:] = 0
    #
    #                 _f = np.fft.rfft(fft_input_output.astype(np.float32), n=fft_length)
    #
    #                 spectrogram[j] = np.real(_f) ** 2 + np.imag(_f) ** 2
    #
    #         __output = spectrogram if (magnitude_squared) else np.sqrt(spectrogram)
    #
    #         return __output


    # def dct_init(self, input_length, dct_coefficient_count):
    #     # init
    #     if (input_length < dct_coefficient_count):
    #         print("Error input_length need to larger than dct_coefficient_count")
    #
    #     cosine = np.zeros((dct_coefficient_count, input_length))
    #     fnorm = np.sqrt(2.0 / input_length)
    #     arg = np.pi / input_length
    #     for i in range(dct_coefficient_count):
    #         for j in range(input_length):
    #             cosine[i][j] = fnorm * np.cos(i * arg * (j + 0.5))
    #
    #     return cosine
    #
    # def dct_compute(self, worked_filiter, input_length, dct_coefficient_count, cosine):
    #     # compute
    #     output_dct = np.zeros(dct_coefficient_count)
    #     worked_length = worked_filiter.shape[0]
    #
    #     if (worked_length > input_length):
    #         worked_length = input_length
    #
    #     for i in range(dct_coefficient_count):
    #         _sum = 0.0
    #         for j in range(worked_length):
    #             _sum += cosine[i][j] * worked_filiter[j]
    #         output_dct[i] = _sum
    #
    #     return output_dct

    def fbank(self, data):
              # ,samplerate=16000,winlen=0.025,winstep=0.01,
              # nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
              # winfunc=lambda x:numpy.ones((x,))):
        """Compute Mel-filterbank energy features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the sample rate of the signal we are working with, in Hz.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
        :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
            second return value is the energy in each frame (total energy, unwindowed)
        """
        # highfreq= highfreq or samplerate/2
        # data = self.preemphasis(data, self.preemph)
        data = data[:,0]
        frames = self.framesig(data)
        # pspec = self.powspec(frames, self.nfft)

        pspec = np.absolute(np.fft.rfft(frames, self.nfft))

        tfdata = np.load("./features.npz")
        pspec = tfdata['spectrogram'][0,...]
        # if self.magnitude_squared:
        #     pspec = np.square(pspec)
        # pspec = 1.0 / self.nfft * pspec
        # f, t, pspec = stft(data, fs = self.sample_rate,  nperseg = self.frame_len, noverlap = self.frame_len - self.frame_step)
        # pspec = np.absolute(pspec).T

        energy = np.sum(pspec, 1) # this stores the total energy in each frame
        energy = np.where(energy == 0, np.finfo(float).eps, energy) # if energy is zero, we get problems with log

        fb = self.get_filterbanks()  #nfilt,nfft,samplerate,lowfreq,highfreq)
        feat = np.dot(pspec, fb.T) # compute the filterbank energies
        feat = np.where(feat == 0, np.finfo(float).eps, feat) # if feat is zero, we get problems with log

        return feat, energy, frames, pspec, fb

    def logfbank(self, data):
        # ,samplerate=16000,winlen=0.025,winstep=0.01,
        #          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
        #          winfunc=lambda x:numpy.ones((x,))):
        """Compute log Mel-filterbank energy features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the sample rate of the signal we are working with, in Hz.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
        :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
        """
        feat, energy = self.fbank(data)  #,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
        return np.log(feat)

    def ssc(self, signal,samplerate=16000,winlen=0.025,winstep=0.01,
            nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
            winfunc=lambda x:np.ones((x,))):
        """Compute Spectral Subband Centroid features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the sample rate of the signal we are working with, in Hz.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
        :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
        """
        highfreq= highfreq or samplerate/2
        signal = sigproc.preemphasis(signal,preemph)
        frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
        pspec = sigproc.powspec(frames,nfft)
        pspec = np.where(pspec == 0,np.finfo(float).eps,pspec) # if things are all zeros we get problems

        fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
        feat = np.dot(pspec,fb.T) # compute the filterbank energies
        R = np.tile(np.linspace(1,samplerate/2,np.size(pspec,1)),(np.size(pspec,0),1))

        return np.dot(pspec*R,fb.T) / feat

    def hz2mel(self, hz):
        """Convert a value in Hertz to Mels

        :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 1127.0 * np.log1p(hz / 700.)
        # return 2595 * numpy.log10(1 + hz/700.)

    def mel2hz(self, mel):
        """Convert a value in Mels to Hertz

        :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        # return 700*(10**(mel/2595.0) - 1)
        return 700*np.expm1(mel/1127.0)

    def get_filterbanks(self):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        # highfreq= highfreq or samplerate/2
        # assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        lowmel = self.hz2mel(self.lower_frequency_limit)
        highmel = self.hz2mel(self.upper_frequency_limit)
        mels = np.linspace(lowmel, highmel, self.filterbank_channel_count + 2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bins = np.floor(self.nfft * self.mel2hz(mels) / self.sample_rate)

        fbank = np.zeros([self.filterbank_channel_count, self.nfft//2 + 1])
        for j in range(0, self.filterbank_channel_count):
            for i in range(int(bins[j]), int(bins[j+1])):
                fbank[j,i] = (i - bins[j]) / (bins[j+1] - bins[j])
            for i in range(int(bins[j + 1]), int(bins[j + 2])):
                fbank[j,i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
        # const
        # double
        # hz_per_sbin =
        # 0.5 * sample_rate_ / static_cast < double > (input_length_ - 1);
        # start_index_ = static_cast < int > (1.5 + (lower_frequency_limit / hz_per_sbin));
        # end_index_ = static_cast < int > (upper_frequency_limit / hz_per_sbin);

        # weights_.resize(input_length_);
        # for (int i = 0; i < input_length_; ++i) {
        #     channel = band_mapper_[i];
        # if ((i < start_index_) | | (i > end_index_)) {
        # weights_[i] = 0.0;
        # } else {
        # if (channel >= 0) {
        # weights_[i] =
        # (center_frequencies_[channel + 1] - FreqToMel(i * hz_per_sbin)) /
        # (center_frequencies_[channel + 1] - center_frequencies_[channel]);
        # } else {
        # weights_[i] = (center_frequencies_[0] - FreqToMel(i * hz_per_sbin)) /
        # (center_frequencies_[0] - mel_low);
        # }
        # }
        # }
        return fbank

    def lifter(self, cepstra, L=22):
        """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
        magnitude of the high frequency DCT coeffs.

        :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
        :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
        """
        if L > 0:
            nframes,ncoeff = np.shape(cepstra)
            n = np.arange(ncoeff)
            lift = 1 + (L/2.)*np.sin(np.pi*n/L)
            return lift*cepstra
        else:
            # values of L <= 0, do nothing
            return cepstra

    def delta(feat, N):
        """Compute delta features from a feature vector sequence.

        :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
        :param N: For each frame, calculate delta features based on preceding and following N frames
        :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
        """
        if N < 1:
            raise ValueError('N must be an integer >= 1')
        NUMFRAMES = len(feat)
        denominator = 2 * sum([i**2 for i in range(1, N+1)])
        delta_feat = np.empty_like(feat)
        padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
        for t in range(NUMFRAMES):
            delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
        return delta_feat

    # def round_half_up(number):
    #     return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


    # def rolling_window(a, window, step=1):
    #     # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    #     shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    #     strides = a.strides + (a.strides[-1],)
    #     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

    def framesig(self, data):
        # frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
        """Frame a signal into overlapping frames.

        :param sig: the audio signal to frame.
        :param frame_len: length of each frame measured in samples.
        :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied.
        :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
        :returns: an array of frames. Size is NUMFRAMES by frame_len.
        """
        slen = len(data)
        # frame_len = int(round_half_up(frame_len))
        # frame_step = int(round_half_up(frame_step))
        if slen <= self.frame_len:
            numframes = 1
        else:
            numframes = 1 + int(math.ceil((1.0 * slen - self.frame_len) / self.frame_step))

        padlen = int((numframes - 1) * self.frame_step + self.frame_len)

        zeros = np.zeros((padlen - slen,))
        padsignal = np.concatenate((data, zeros))
        # if stride_trick:
            # win = winfunc(frame_len)
            # frames = rolling_window(padsignal, window=frame_len, step=frame_step)
        shape = data.shape[:-1] + (data.shape[-1] - self.frame_len + 1, self.frame_len)
        strides = data.strides + (data.strides[-1],)
        frames = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)[::self.frame_step]
        # else:
        #     indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
        #         np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        #     indices = np.array(indices, dtype=np.int32)
        #     frames = padsignal[indices]
        #     win = np.tile(winfunc(frame_len), (numframes, 1))

        # return frames
        return frames * self.window


    def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
        """Does overlap-add procedure to undo the action of framesig.

        :param frames: the array of frames.
        :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
        :param frame_len: length of each frame measured in samples.
        :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
        :param winfunc: the analysis window to apply to each frame. By default no window is applied.
        :returns: a 1-D signal.
        """
        frame_len = round_half_up(frame_len)
        frame_step = round_half_up(frame_step)
        numframes = np.shape(frames)[0]
        assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        padlen = (numframes - 1) * frame_step + frame_len

        if siglen <= 0: siglen = padlen

        rec_signal = np.zeros((padlen,))
        window_correction = np.zeros((padlen,))
        win = winfunc(frame_len)

        for i in range(0, numframes):
            window_correction[indices[i, :]] = window_correction[
                                                   indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
            rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

        rec_signal = rec_signal / window_correction
        return rec_signal[0:siglen]


    # def magspec(frames, NFFT):
    #     """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    #
    #     :param frames: the array of frames. Each row is a frame.
    #     :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    #     :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    #     """
    #     if np.shape(frames)[1] > NFFT:
    #         logging.warn(
    #             'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
    #             np.shape(frames)[1], NFFT)
    #     complex_spec = np.fft.rfft(frames, NFFT)
    #     return np.absolute(complex_spec)
    #
    #
    # def powspec(frames, NFFT):
    #     """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    #
    #     :param frames: the array of frames. Each row is a frame.
    #     :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    #     :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    #     """
    #     return 1.0 / NFFT * np.square(magspec(frames, NFFT))


    def logpowspec(frames, NFFT, norm=1):
        """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

        :param frames: the array of frames. Each row is a frame.
        :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
        :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
        :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
        """
        ps = powspec(frames, NFFT);
        ps[ps <= 1e-30] = 1e-30
        lps = 10 * np.log10(ps)
        if norm:
            return lps - np.max(lps)
        else:
            return lps


    def preemphasis(self, data, coeff=0.95):
        """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
        :returns: the filtered signal.
        """
        return np.append(data[0], data[1:] - coeff * data[:-1])

    def ctc_beam_search_decoder(self, input_tensor, config, beamwidth):
        alphabet = Alphabet(config)
        text_label, blank = alphabet._label_to_str, alphabet.string_from_label(-1)
        pred = input_tensor.squeeze()

        t_step = pred.shape[0]
        # idx_b = text_label.index(blank)
        idx_b = pred.shape[1] - 1

        _pB = {}
        _pNB = {}
        _pT = {}

        _init = () # init state, to make sure the first index is not blank ****

        for __t in ['c', 'l']:
            _pB[__t] = {}
            _pNB[__t] = {}
            _pT[__t] = {}

        _pB['l'][_init] = 1
        _pNB['l'][_init] = 0
        _pT['l'][_init] = 1

        for _t in range(t_step):
            _pB['c'] = {}
            _pNB['c'] = {}
            _pT['c'] = {}

            for _candidate in _pNB['l']:
                _TpNB = 0
                if _candidate != _init:
                    _TpNB = _pNB['l'][_candidate] * pred[_t][_candidate[-1]]
                _TpB = _pT['l'][_candidate] * pred[_t][idx_b]
                if _candidate in _pNB['c']:
                    _pNB['c'][_candidate] += _TpNB
                else:
                    _pNB['c'][_candidate] = _TpNB
                _pB['c'][_candidate] = _TpB
                _pT['c'][_candidate] = _pNB['c'][_candidate] + _pB['c'][_candidate]

                for i, v in np.ndenumerate(pred[_t]):
                    if i < (idx_b,):
                        extand_t = _candidate + (i,)
                        if len(_candidate) > 0 and _candidate[-1] == i:
                            _TpNB = v * _pB['l'][_candidate]

                        else:
                            _TpNB = v * _pT['l'][_candidate]

                        if extand_t in _pT['c']:
                            _pT['c'][extand_t] += _TpNB
                            _pNB['c'][extand_t] += _TpNB
                        else:
                            _pB['c'][extand_t] = 0
                            _pT['c'][extand_t] = _TpNB
                            _pNB['c'][extand_t] = _TpNB

            sorted_c = sorted(_pT['c'].items(), reverse=True, key=lambda item:item[1])
            _pB['l'] = {}
            _pNB['l'] = {}
            _pT['l'] = {}
            for _sent in sorted_c[:beamwidth]:
                _pB['l'][_sent[0]] = _pB['c'][_sent[0]]
                _pNB['l'][_sent[0]] = _pNB['c'][_sent[0]]
                _pT['l'][_sent[0]] = _pT['c'][_sent[0]]

        res = sorted(_pT['l'].items(), reverse=True, key=lambda item:item[1])[0]
        return res

        text = ''
        for idx, _r in enumerate(res[0]):
            text += text_label[_r[0]]

        return text

# /* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/
#
# // Copied from tensorflow/core/util/ctc/ctc_beam_search.h
# // TODO(b/111524997): Remove this file.
# #ifndef TENSORFLOW_LITE_EXPERIMENTAL_KERNELS_CTC_BEAM_SEARCH_H_
# #define TENSORFLOW_LITE_EXPERIMENTAL_KERNELS_CTC_BEAM_SEARCH_H_
#
# #include <algorithm>
# #include <cmath>
# #include <limits>
# #include <memory>
# #include <vector>
#
# #include "third_party/eigen3/Eigen/Core"
# #include "tensorflow/lite/experimental/kernels/ctc_beam_entry.h"
# #include "tensorflow/lite/experimental/kernels/ctc_beam_scorer.h"
# #include "tensorflow/lite/experimental/kernels/ctc_decoder.h"
# #include "tensorflow/lite/experimental/kernels/ctc_loss_util.h"
# #include "tensorflow/lite/experimental/kernels/top_n.h"
# #include "tensorflow/lite/kernels/internal/compatibility.h"
#
# namespace tflite {
# namespace experimental {
# namespace ctc {
#
# template <typename CTCBeamState = ctc_beam_search::EmptyBeamState,
#           typename CTCBeamComparer =
#               ctc_beam_search::BeamComparer<CTCBeamState>>
# class CTCBeamSearchDecoder : public CTCDecoder {
#   // Beam Search
#   //
#   // Example (GravesTh Fig. 7.5):
#   //         a    -
#   //  P = [ 0.3  0.7 ]  t = 0
#   //      [ 0.4  0.6 ]  t = 1
#   //
#   // Then P(l = -) = P(--) = 0.7 * 0.6 = 0.42
#   //      P(l = a) = P(a-) + P(aa) + P(-a) = 0.3*0.4 + ... = 0.58
#   //
#   // In this case, Best Path decoding is suboptimal.
#   //
#   // For Beam Search, we use the following main recurrence relations:
#   //
#   // Relation 1:
#   // ---------------------------------------------------------- Eq. 1
#   //      P(l=abcd @ t=7) = P(l=abc  @ t=6) * P(d @ 7)
#   //                      + P(l=abcd @ t=6) * (P(d @ 7) + P(- @ 7))
#   // where P(l=? @ t=7), ? = a, ab, abc, abcd are all stored and
#   // updated recursively in the beam entry.
#   //
#   // Relation 2:
#   // ---------------------------------------------------------- Eq. 2
#   //      P(l=abc? @ t=3) = P(l=abc @ t=2) * P(? @ 3)
#   // for ? in a, b, d, ..., (not including c or the blank index),
#   // and the recurrence starts from the beam entry for P(l=abc @ t=2).
#   //
#   // For this case, the length of the new sequence equals t+1 (t
#   // starts at 0).  This special case can be calculated as:
#   //   P(l=abc? @ t=3) = P(a @ 0)*P(b @ 1)*P(c @ 2)*P(? @ 3)
#   // but we calculate it recursively for speed purposes.
#   typedef ctc_beam_search::BeamEntry<CTCBeamState> BeamEntry;
#   typedef ctc_beam_search::BeamRoot<CTCBeamState> BeamRoot;
#   typedef ctc_beam_search::BeamProbability BeamProbability;
#
#  public:
#   typedef BaseBeamScorer<CTCBeamState> DefaultBeamScorer;
#
#   // The beam search decoder is constructed specifying the beam_width (number of
#   // candidates to keep at each decoding timestep) and a beam scorer (used for
#   // custom scoring, for example enabling the use of a language model).
#   // The ownership of the scorer remains with the caller. The default
#   // implementation, CTCBeamSearchDecoder<>::DefaultBeamScorer, generates the
#   // standard beam search.
#   CTCBeamSearchDecoder(int num_classes, int beam_width,
#                        BaseBeamScorer<CTCBeamState>* scorer, int batch_size = 1,
#                        bool merge_repeated = false)
#       : CTCDecoder(num_classes, batch_size, merge_repeated),
#         beam_width_(beam_width),
#         leaves_(beam_width),
#         beam_scorer_(scorer) {
#     Reset();
#   }
#
#   ~CTCBeamSearchDecoder() override {}
#
#   // Run the hibernating beam search algorithm on the given input.
#   bool Decode(const CTCDecoder::SequenceLength& seq_len,
#               const std::vector<CTCDecoder::Input>& input,
#               std::vector<CTCDecoder::Output>* output,
#               CTCDecoder::ScoreOutput* scores) override;
#
#   // Calculate the next step of the beam search and update the internal state.
#   template <typename Vector>
#   void Step(const Vector& log_input_t);
#
#   template <typename Vector>
#   float GetTopK(const int K, const Vector& input,
#                 std::vector<float>* top_k_logits,
#                 std::vector<int>* top_k_indices);
#
#   // Retrieve the beam scorer instance used during decoding.
#   BaseBeamScorer<CTCBeamState>* GetBeamScorer() const { return beam_scorer_; }
#
#   // Set label selection parameters for faster decoding.
#   // See comments for label_selection_size_ and label_selection_margin_.
#   void SetLabelSelectionParameters(int label_selection_size,
#                                    float label_selection_margin) {
#     label_selection_size_ = label_selection_size;
#     label_selection_margin_ = label_selection_margin;
#   }
#
#   // Reset the beam search
#   void Reset();
#
#   // Extract the top n paths at current time step
#   bool TopPaths(int n, std::vector<std::vector<int>>* paths,
#                 std::vector<float>* log_probs, bool merge_repeated) const;
#
#  private:
#   int beam_width_;
#
#   // Label selection is designed to avoid possibly very expensive scorer calls,
#   // by pruning the hypotheses based on the input alone.
#   // Label selection size controls how many items in each beam are passed
#   // through to the beam scorer. Only items with top N input scores are
#   // considered.
#   // Label selection margin controls the difference between minimal input score
#   // (versus the best scoring label) for an item to be passed to the beam
#   // scorer. This margin is expressed in terms of log-probability.
#   // Default is to do no label selection.
#   // For more detail: https://research.google.com/pubs/pub44823.html
#   int label_selection_size_ = 0;       // zero means unlimited
#   float label_selection_margin_ = -1;  // -1 means unlimited.
#
#   gtl::TopN<BeamEntry*, CTCBeamComparer> leaves_;
#   std::unique_ptr<BeamRoot> beam_root_;
#   BaseBeamScorer<CTCBeamState>* beam_scorer_;
#
#   CTCBeamSearchDecoder(const CTCBeamSearchDecoder&) = delete;
#   void operator=(const CTCBeamSearchDecoder&) = delete;
# };
#
# template <typename CTCBeamState, typename CTCBeamComparer>
# bool CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::Decode(
#     const CTCDecoder::SequenceLength& seq_len,
#     const std::vector<CTCDecoder::Input>& input,
#     std::vector<CTCDecoder::Output>* output, ScoreOutput* scores) {
#   // Storage for top paths.
#   std::vector<std::vector<int>> beams;
#   std::vector<float> beam_log_probabilities;
#   int top_n = output->size();
#   if (std::any_of(output->begin(), output->end(),
#                   [this](const CTCDecoder::Output& output) -> bool {
#                     return output.size() < this->batch_size_;
#                   })) {
#     return false;
#   }
#   if (scores->rows() < batch_size_ || scores->cols() < top_n) {
#     return false;
#   }
#
#   for (int b = 0; b < batch_size_; ++b) {
#     int seq_len_b = seq_len[b];
#     Reset();
#
#     for (int t = 0; t < seq_len_b; ++t) {
#       // Pass log-probabilities for this example + time.
#       Step(input[t].row(b));
#     }  // for (int t...
#
#     // O(n * log(n))
#     std::unique_ptr<std::vector<BeamEntry*>> branches(leaves_.Extract());
#     leaves_.Reset();
#     for (int i = 0; i < branches->size(); ++i) {
#       BeamEntry* entry = (*branches)[i];
#       beam_scorer_->ExpandStateEnd(&entry->state);
#       entry->newp.total +=
#           beam_scorer_->GetStateEndExpansionScore(entry->state);
#       leaves_.push(entry);
#     }
#
#     bool status =
#         TopPaths(top_n, &beams, &beam_log_probabilities, merge_repeated_);
#     if (!status) {
#       return status;
#     }
#
#     TFLITE_DCHECK_EQ(top_n, beam_log_probabilities.size());
#     TFLITE_DCHECK_EQ(beams.size(), beam_log_probabilities.size());
#
#     for (int i = 0; i < top_n; ++i) {
#       // Copy output to the correct beam + batch
#       (*output)[i][b].swap(beams[i]);
#       (*scores)(b, i) = -beam_log_probabilities[i];
#     }
#   }  // for (int b...
#   return true;
# }
#
# template <typename CTCBeamState, typename CTCBeamComparer>
# template <typename Vector>
# float CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::GetTopK(
#     const int K, const Vector& input, std::vector<float>* top_k_logits,
#     std::vector<int>* top_k_indices) {
#   // Find Top K choices, complexity nk in worst case. The array input is read
#   // just once.
#   TFLITE_DCHECK_EQ(num_classes_, input.size());
#   top_k_logits->clear();
#   top_k_indices->clear();
#   top_k_logits->resize(K, -INFINITY);
#   top_k_indices->resize(K, -1);
#   for (int j = 0; j < num_classes_ - 1; ++j) {
#     const float logit = input(j);
#     if (logit > (*top_k_logits)[K - 1]) {
#       int k = K - 1;
#       while (k > 0 && logit > (*top_k_logits)[k - 1]) {
#         (*top_k_logits)[k] = (*top_k_logits)[k - 1];
#         (*top_k_indices)[k] = (*top_k_indices)[k - 1];
#         k--;
#       }
#       (*top_k_logits)[k] = logit;
#       (*top_k_indices)[k] = j;
#     }
#   }
#   // Return max value which is in 0th index or blank character logit
#   return std::max((*top_k_logits)[0], input(num_classes_ - 1));
# }
#
# template <typename CTCBeamState, typename CTCBeamComparer>
# template <typename Vector>
# void CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::Step(
#     const Vector& raw_input) {
#   std::vector<float> top_k_logits;
#   std::vector<int> top_k_indices;
#   const bool top_k =
#       (label_selection_size_ > 0 && label_selection_size_ < raw_input.size());
#   // Number of character classes to consider in each step.
#   const int max_classes = top_k ? label_selection_size_ : (num_classes_ - 1);
#   // Get max coefficient and remove it from raw_input later.
#   float max_coeff;
#   if (top_k) {
#     max_coeff = GetTopK(label_selection_size_, raw_input, &top_k_logits,
#                         &top_k_indices);
#   } else {
#     max_coeff = raw_input.maxCoeff();
#   }
#
#   // Get normalization term of softmax: log(sum(exp(logit[j]-max_coeff))).
#   float logsumexp = 0.0;
#   for (int j = 0; j < raw_input.size(); ++j) {
#     logsumexp += Eigen::numext::exp(raw_input(j) - max_coeff);
#   }
#   logsumexp = Eigen::numext::log(logsumexp);
#   // Final normalization offset to get correct log probabilities.
#   float norm_offset = max_coeff + logsumexp;
#
#   const float label_selection_input_min =
#       (label_selection_margin_ >= 0) ? (max_coeff - label_selection_margin_)
#                                      : -std::numeric_limits<float>::infinity();
#
#   // Extract the beams sorted in decreasing new probability
#   TFLITE_DCHECK_EQ(num_classes_, raw_input.size());
#
#   std::unique_ptr<std::vector<BeamEntry*>> branches(leaves_.Extract());
#   leaves_.Reset();
#
#   for (BeamEntry* b : *branches) {
#     // P(.. @ t) becomes the new P(.. @ t-1)
#     b->oldp = b->newp;
#   }
#
#   for (BeamEntry* b : *branches) {
#     if (b->parent != nullptr) {  // if not the root
#       if (b->parent->Active()) {
#         // If last two sequence characters are identical:
#         //   Plabel(l=acc @ t=6) = (Plabel(l=acc @ t=5)
#         //                          + Pblank(l=ac @ t=5))
#         // else:
#         //   Plabel(l=abc @ t=6) = (Plabel(l=abc @ t=5)
#         //                          + P(l=ab @ t=5))
#         float previous = (b->label == b->parent->label) ? b->parent->oldp.blank
#                                                         : b->parent->oldp.total;
#         b->newp.label =
#             LogSumExp(b->newp.label,
#                       beam_scorer_->GetStateExpansionScore(b->state, previous));
#       }
#       // Plabel(l=abc @ t=6) *= P(c @ 6)
#       b->newp.label += raw_input(b->label) - norm_offset;
#     }
#     // Pblank(l=abc @ t=6) = P(l=abc @ t=5) * P(- @ 6)
#     b->newp.blank = b->oldp.total + raw_input(blank_index_) - norm_offset;
#     // P(l=abc @ t=6) = Plabel(l=abc @ t=6) + Pblank(l=abc @ t=6)
#     b->newp.total = LogSumExp(b->newp.blank, b->newp.label);
#
#     // Push the entry back to the top paths list.
#     // Note, this will always fill leaves back up in sorted order.
#     leaves_.push(b);
#   }
#
#   // we need to resort branches in descending oldp order.
#
#   // branches is in descending oldp order because it was
#   // originally in descending newp order and we copied newp to oldp.
#
#   // Grow new leaves
#   for (BeamEntry* b : *branches) {
#     // A new leaf (represented by its BeamProbability) is a candidate
#     // iff its total probability is nonzero and either the beam list
#     // isn't full, or the lowest probability entry in the beam has a
#     // lower probability than the leaf.
#     auto is_candidate = [this](const BeamProbability& prob) {
#       return (prob.total > kLogZero &&
#               (leaves_.size() < beam_width_ ||
#                prob.total > leaves_.peek_bottom()->newp.total));
#     };
#
#     if (!is_candidate(b->oldp)) {
#       continue;
#     }
#
#     for (int ind = 0; ind < max_classes; ind++) {
#       const int label = top_k ? top_k_indices[ind] : ind;
#       const float logit = top_k ? top_k_logits[ind] : raw_input(ind);
#       // Perform label selection: if input for this label looks very
#       // unpromising, never evaluate it with a scorer.
#       // We may compare logits instead of log probabilities,
#       // since the difference is the same in both cases.
#       if (logit < label_selection_input_min) {
#         continue;
#       }
#       BeamEntry& c = b->GetChild(label);
#       if (!c.Active()) {
#         //   Pblank(l=abcd @ t=6) = 0
#         c.newp.blank = kLogZero;
#         // If new child label is identical to beam label:
#         //   Plabel(l=abcc @ t=6) = Pblank(l=abc @ t=5) * P(c @ 6)
#         // Otherwise:
#         //   Plabel(l=abcd @ t=6) = P(l=abc @ t=5) * P(d @ 6)
#         beam_scorer_->ExpandState(b->state, b->label, &c.state, c.label);
#         float previous = (c.label == b->label) ? b->oldp.blank : b->oldp.total;
#         c.newp.label = logit - norm_offset +
#                        beam_scorer_->GetStateExpansionScore(c.state, previous);
#         // P(l=abcd @ t=6) = Plabel(l=abcd @ t=6)
#         c.newp.total = c.newp.label;
#
#         if (is_candidate(c.newp)) {
#           // Before adding the new node to the beam, check if the beam
#           // is already at maximum width.
#           if (leaves_.size() == beam_width_) {
#             // Bottom is no longer in the beam search.  Reset
#             // its probability; signal it's no longer in the beam search.
#             BeamEntry* bottom = leaves_.peek_bottom();
#             bottom->newp.Reset();
#           }
#           leaves_.push(&c);
#         } else {
#           // Deactivate child.
#           c.oldp.Reset();
#           c.newp.Reset();
#         }
#       }
#     }
#   }  // for (BeamEntry* b...
# }
#
# template <typename CTCBeamState, typename CTCBeamComparer>
# void CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::Reset() {
#   leaves_.Reset();
#
#   // This beam root, and all of its children, will be in memory until
#   // the next reset.
#   beam_root_.reset(new BeamRoot(nullptr, -1));
#   beam_root_->RootEntry()->newp.total = 0.0;  // ln(1)
#   beam_root_->RootEntry()->newp.blank = 0.0;  // ln(1)
#
#   // Add the root as the initial leaf.
#   leaves_.push(beam_root_->RootEntry());
#
#   // Call initialize state on the root object.
#   beam_scorer_->InitializeState(&beam_root_->RootEntry()->state);
# }
#
# template <typename CTCBeamState, typename CTCBeamComparer>
# bool CTCBeamSearchDecoder<CTCBeamState, CTCBeamComparer>::TopPaths(
#     int n, std::vector<std::vector<int>>* paths, std::vector<float>* log_probs,
#     bool merge_repeated) const {
#   TFLITE_DCHECK(paths);
#   TFLITE_DCHECK(log_probs);
#   paths->clear();
#   log_probs->clear();
#   if (n > beam_width_) {
#     return false;
#   }
#   if (n > leaves_.size()) {
#     return false;
#   }
#
#   gtl::TopN<BeamEntry*, CTCBeamComparer> top_branches(n);
#
#   // O(beam_width_ * log(n)), space complexity is O(n)
#   for (auto it = leaves_.unsorted_begin(); it != leaves_.unsorted_end(); ++it) {
#     top_branches.push(*it);
#   }
#   // O(n * log(n))
#   std::unique_ptr<std::vector<BeamEntry*>> branches(top_branches.Extract());
#
#   for (int i = 0; i < n; ++i) {
#     BeamEntry* e((*branches)[i]);
#     paths->push_back(e->LabelSeq(merge_repeated));
#     log_probs->push_back(e->newp.total);
#   }
#   return true;
# }
#
# }  // namespace ctc
# }  // namespace experimental
# }  // namespace tflite
#
# #endif  // TENSORFLOW_LITE_EXPERIMENTAL_KERNELS_CTC_BEAM_SEARCH_H_


#     def audio_spectrogram(self, samples, window_size, stride, magnitude_squared):
#         # window_size, #=(16000 * (32 / 1000)), #Config.audio_window_samples,
#         # stride, # =(16000 * (20 / 1000)), #(Config.audio_step_samples,
#         # magnitude_squared) : # =True) :
#         if (len(samples.shape) != 2) :
#             print ("input must be 2-dimensional")
#
#         window_size = int(window_size)
#         stride = int(stride)
#
#         sample_count = samples.shape[0]
#         channel_count = samples.shape[1] # == 1
#
#         def output_frequency_channels(n) :
#             _log = np.floor(np.log2(n))
#             _log = _log if ((n == (n & ~(n - 1)))) else _log + 1
#
#             fft_length = 1 << _log.astype(np.int32)
#
#             return 1 + fft_length / 2, fft_length.astype(np.int32)
#
#         output_width, fft_length  = output_frequency_channels(window_size)
#         output_width = output_width.astype(np.int32)
#
#         length_minus_windows = sample_count - window_size
#
#         output_height = 0 if length_minus_windows < 0 else (1 + (length_minus_windows / stride))
#         output_height = int(output_height)
#         output_slices = channel_count
#
#         __output = np.zeros((output_slices, output_height, output_width))
#         hann_window = np.hanning(window_size)
#
#         for i in range(channel_count) :
#             input_for_channel = samples[:, i]
#
#             input_for_compute = np.zeros(stride)
#             spectrogram = np.zeros((output_height, output_width))
#
#             fft_input_output = np.zeros(fft_length)
#
#             for j in range (output_height) :
#                 start = j * stride
#                 end = start + window_size
#                 if (end < sample_count) :
#                     input_for_compute = input_for_channel[start: end]
#
#                     fft_input_output[0 :window_size] = input_for_compute * hann_window
#                     fft_input_output[window_size:] = 0
#
#                     _f = np.fft.rfft(fft_input_output.astype(np.float32), n=fft_length)
#
#                     spectrogram[j] = np.real(_f) ** 2 + np.imag(_f) ** 2
#
#             __output = spectrogram if (magnitude_squared) else np.sqrt(spectrogram)
#
#             return __output
#
#     def mfcc_mel_filiterbank_init(self, sample_rate, input_length):
#         # init
#         # filterbank_channel_count_   = 40
#         # lower_frequency_limit_      = 20
#         # upper_frequency_limit_      = 4000
#
#         def freq2mel(freq):
#             return 1127.0 * np.log1p(freq / 700)
#
#         center_frequencies = np.zeros((self.filterbank_channel_count_ + 1))
#         mel_low = freq2mel(self.lower_frequency_limit_)
#         mel_hi = freq2mel(self.upper_frequency_limit_)
#         mel_span = mel_hi - mel_low
#         mel_sapcing = mel_span / (filterbank_channel_count_ + 1)
#
#         for i in range((filterbank_channel_count_ + 1)) :
#             center_frequencies[i] = mel_low + (mel_sapcing * (1 + i))
#
#         hz_per_sbin = 0.5 * sample_rate / (input_length - 1)
#         start_index = int(1.5 + (lower_frequency_limit_ / hz_per_sbin))
#         end_index = int(upper_frequency_limit_ / hz_per_sbin)
#
#         band_mapper = np.zeros(input_length)
#         channel = 0
#
#         for i in range(input_length) :
#             melf = freq2mel(i * hz_per_sbin)
#
#             if ((i < start_index) or (i > end_index)) :
#                 band_mapper[i] = -2
#             else :
#                 while ((center_frequencies[int(channel)] < melf) and
#                        (channel < filterbank_channel_count_)) :
#                     channel += 1
#                 band_mapper[i] = channel - 1
#
#         weights = np.zeros(input_length)
#         for i in range(input_length) :
#             channel = band_mapper[i]
#             if ((i < start_index) or (i > end_index)) :
#                 weights[i] = 0.0
#             else :
#                 if (channel >= 0) :
#                     weights[i] = ((center_frequencies[int(channel) + 1] - freq2mel(i * hz_per_sbin)) /
#                                   (center_frequencies[int(channel) + 1] - center_frequencies[int(channel)]))
#                 else :
#                     weights[i] = ((center_frequencies[0] - freq2mel(i * hz_per_sbin)) /
#                                   (center_frequencies[0] - mel_low))
#
#         return start_index, end_index, weights, band_mapper
#
#     def mfcc_mel_filiterbank_compute(mfcc_input, input_length, start_index, end_index, weights, band_mapper) :
#         filterbank_channel_count_   = 40
#         # Compute
#         output_channels = np.zeros(filterbank_channel_count_)
#         for i in range(start_index, (end_index + 1)) :
#             spec_val = np.sqrt(mfcc_input[i])
#             weighted = spec_val * weights[i]
#             channel = band_mapper[i]
#             if (channel >= 0) :
#                 output_channels[int(channel)] += weighted
#             channel += 1
#             if (channel < filterbank_channel_count_) :
#                 output_channels[int(channel)] += (spec_val - weighted)
#
#         return output_channels
#
#     def dct_init(input_length, dct_coefficient_count) :
#         # init
#         if (input_length < dct_coefficient_count) :
#             print ("Error input_length need to larger than dct_coefficient_count")
#
#         cosine = np.zeros((dct_coefficient_count, input_length))
#         fnorm = np.sqrt(2.0 / input_length)
#         arg = np.pi / input_length
#         for i in range(dct_coefficient_count) :
#             for j in range (input_length) :
#                 cosine[i][j] = fnorm * np.cos(i * arg * (j + 0.5))
#
#         return cosine
#
#     def dct_compute(worked_filiter, input_length, dct_coefficient_count, cosine) :
#         # compute
#         output_dct = np.zeros(dct_coefficient_count)
#         worked_length = worked_filiter.shape[0]
#
#         if (worked_length > input_length) :
#             worked_length = input_length
#
#         for i in range(dct_coefficient_count) :
#             _sum = 0.0
#             for j in range(worked_length) :
#                 _sum += cosine[i][j] * worked_filiter[j]
#             output_dct[i] = _sum
#
#         return output_dct
#
#     def mfcc(spectrogram, sample_rate, dct_coefficient_count) :
#         audio_channels, spectrogram_samples, spectrogram_channels  = spectrogram.shape
#         #print (spectrogram.shape)
#         kFilterbankFloor = 1e-12
#         filterbank_channel_count = 40
#
#         mfcc_output = np.zeros((spectrogram_samples, dct_coefficient_count))
#         for i in range(audio_channels) :
#             start_index, end_index, weights, band_mapper = \
#                 mfcc_mel_filiterbank_init(sample_rate, spectrogram_channels)
#             cosine = dct_init(filterbank_channel_count, dct_coefficient_count)
#             for j in range(spectrogram_samples) :
#                 mfcc_input = spectrogram[i, j, :]
#
#                 mel_filiter = mfcc_mel_filiterbank_compute(mfcc_input, spectrogram_channels,
#                                                            start_index, end_index,
#                                                            weights, band_mapper)
#                 for k in range(mel_filiter.shape[0]) :
#                     val = mel_filiter[k]
#                     if (val < kFilterbankFloor) :
#                         val = kFilterbankFloor
#
#                     mel_filiter[k] = np.log(val)
#
#                 mfcc_output[j, :] = dct_compute(mel_filiter,
#                                              filterbank_channel_count,
#                                              dct_coefficient_count,
#                                              cosine)
#
#         return mfcc_output
#
# n_input    = 26
# n_context  = 9
# n_steps    = 16
# numcep     = n_input
# numcontext = n_context
# beamwidth  = 10
#
# def main():
#     log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
#     args = build_argparser().parse_args()
#
#     alphabet = Alphabet(os.path.abspath(args.alphabet))
#
#     # Speech feature extration
#     _wave = wave.open(args.input, 'rb')
#     fs = _wave.getframerate()
#     if fs is not 16000:
#         log.error("Please using 16kHz wave file, not {}Hz\n".format(fs))
#     _length = _wave.getnframes()
#     audio = np.frombuffer(_wave.readframes(_length), dtype=np.dtype('<h'))
#
#     audio = audio/np.float32(32768) # normalize to -1 to 1, int 16 to float32
#
#     audio = audio.reshape(-1, 1)
#     spectrogram = audio_spectrogram(audio, (16000 * 32 / 1000), (16000 * 20 / 1000), True)
#     features = mfcc(spectrogram.reshape(1, spectrogram.shape[0], -1), fs, 26)
#
#     empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
#     features = np.concatenate((empty_context, features, empty_context))
#
#     num_strides = len(features) - (n_context * 2)
#     # Create a view into the array with overlapping strides of size
#     # numcontext (past) + 1 (present) + numcontext (future)
#     window_size = 2*n_context+1
#     features = np.lib.stride_tricks.as_strided(
#         features,
#         (num_strides, window_size, n_input),
#         (features.strides[0], features.strides[0], features.strides[1]),
#         writeable=False)
#



class MFCC_Filterbank:
    def __init__(self, sample_rate, filterbank_channel_count, lower_frequency_limit, upper_frequency_limit, input_length):
        self.sample_rate = sample_rate
        self.filterbank_channel_count = filterbank_channel_count
        self.lower_frequency_limit = lower_frequency_limit
        self.upper_frequency_limit = upper_frequency_limit
        self.input_length = input_length

    def freq2mel(self, freq):
        return 1127.0 * np.log1p(freq / 700)

    def initialize(self):

        # init
        # filterbank_channel_count_ = 40
        # lower_frequency_limit_ = 20
        # upper_frequency_limit_ = 4000


        center_frequencies = np.zeros((self.filterbank_channel_count + 1))
        mel_low = self.freq2mel(self.lower_frequency_limit)
        mel_hi = self.freq2mel(self.upper_frequency_limit)
        mel_span = mel_hi - mel_low
        mel_sapcing = mel_span / (self.filterbank_channel_count + 1)

        for i in range((self.filterbank_channel_count + 1)):
            center_frequencies[i] = mel_low + (mel_sapcing * (1 + i))

        hz_per_sbin = 0.5 * self.sample_rate / (self.input_length - 1)
        self.start_index = int(1.5 + (self.lower_frequency_limit / hz_per_sbin))
        self.end_index = int(self.upper_frequency_limit / hz_per_sbin)

        self.band_mapper = np.zeros(self.input_length)
        channel = 0

        for i in range(self.input_length):
            melf = self.freq2mel(i * hz_per_sbin)

            if ((i < self.start_index) or (i > self.end_index)):
                self.band_mapper[i] = -2
            else:
                while ((center_frequencies[int(channel)] < melf) and
                       (channel < self.filterbank_channel_count)):
                    channel += 1
                self.band_mapper[i] = channel - 1

        self.weights = np.zeros(self.input_length)
        for i in range(self.input_length):
            channel = self.band_mapper[i]
            if ((i < self.start_index) or (i > self.end_index)):
                self.weights[i] = 0.0
            else:
                if (channel >= 0):
                    self.weights[i] = ((center_frequencies[int(channel) + 1] - self.freq2mel(i * hz_per_sbin)) /
                                  (center_frequencies[int(channel) + 1] - center_frequencies[int(channel)]))
                else:
                    self.weights[i] = ((center_frequencies[0] - self.freq2mel(i * hz_per_sbin)) /
                                  (center_frequencies[0] - mel_low))

        return self.start_index, self.end_index, self.weights, self.band_mapper

    def compute(self, mfcc_input):#, input_length, start_index, end_index, weights, band_mapper):
        # filterbank_channel_count_ = 40
        # Compute
        output_channels = np.zeros(self.filterbank_channel_count)
        for i in range(self.start_index, (self.end_index + 1)):
            spec_val = np.sqrt(mfcc_input[i])
            weighted = spec_val * self.weights[i]
            channel = self.band_mapper[i]
            if (channel >= 0):
                output_channels[int(channel)] += weighted
            channel += 1
            if (channel < self.filterbank_channel_count):
                output_channels[int(channel)] += (spec_val - weighted)

        return output_channels

class DCT:
    def __init__(self, input_length, dct_coefficient_count):
        self.input_length = input_length
        self.dct_coefficient_count = dct_coefficient_count

    def initialize(self):# input_length, dct_coefficient_count):
        # init
        if (self.input_length < self.dct_coefficient_count):
            print("Error input_length need to larger than dct_coefficient_count")

        self.cosine = np.zeros((self.dct_coefficient_count, self.input_length))
        fnorm = np.sqrt(2.0 / self.input_length)
        arg = np.pi / self.input_length
        for i in range(self.dct_coefficient_count):
            for j in range(self.input_length):
                self.cosine[i][j] = fnorm * np.cos(i * arg * (j + 0.5))

        return self.cosine

    def compute(self, worked_filiter):#, input_length, dct_coefficient_count, cosine):
        # compute
        output_dct = np.zeros(self.dct_coefficient_count)
        worked_length = worked_filiter.shape[0]

        if (worked_length > self.input_length):
            worked_length = self.input_length

        for i in range(self.dct_coefficient_count):
            _sum = 0.0
            for j in range(worked_length):
                _sum += self.cosine[i][j] * worked_filiter[j]
            output_dct[i] = _sum

        return output_dct


class Alphabet(object):
    def __init__(self, config_file):
        self._label_to_str = []
        self._str_to_label = {}
        self._size = 0
        with codecs.open(config_file, 'r', 'utf-8') as fin:
            for line in fin:
                if line[0:2] == '\\#':
                    line = '#\n'
                elif line[0] == '#':
                    continue
                self._label_to_str += line[:-1] # remove the line ending
                self._str_to_label[line[:-1]] = self._size
                self._size += 1

    def string_from_label(self, label):
        return self._label_to_str[label]

    def label_from_string(self, string):
        return self._str_to_label[string]

    def size(self):
        return self._size

