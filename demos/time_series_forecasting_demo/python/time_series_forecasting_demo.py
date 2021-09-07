#!/usr/bin/env python3

"""
 Copyright (c) 2021 Intel Corporation
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
import argparse
import logging as log
import sys
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from openvino.inference_engine import IECore
from accuracy_checker.dataset import read_annotation


class ForecastingEngine:
    """ OpenVINO engine for Time Series Forecasting.

    Arguments:
        model_xml (str): path to model's .xml file.
        model_bin (str): path to model's .bin file.
        input_name (str): name of input blob of model.
        output_name (str): name of output blob of model.
    """
    def __init__(self, model_xml, model_bin, input_name, output_name, quantiles):
        self.logger = log.getLogger("ForecastingEngine")
        self.logger.info("loading network")
        self.logger.info(f"model_xml: {model_xml}")
        self.logger.info(f"model_bin: {model_bin}")
        self.logger.info(f"input_name: {input_name}")
        self.logger.info(f"output_name: {output_name}")
        self.logger.info(f"quantiles: {quantiles}")
        self.ie = IECore()
        self.net = self.ie.read_network(
            model=model_xml,
            weights=model_bin
        )
        self.net_exec = self.ie.load_network(self.net, "CPU")
        self.input_name = input_name
        self.output_name = output_name
        self.quantiles = quantiles
        assert self.output_name != "", "there is not output in model"

    def __call__(self, inputs):
        """ Main forecasting method.

        Arguments:
            inputs (np.array): input data.

        Returns:
            preds (dict): predicted quantiles.
        """
        preds = self.net_exec.infer(inputs={self.input_name: inputs})[self.output_name]
        out = {}
        for i, q in enumerate(self.quantiles):
            out[q] = preds[:, :, i].flatten()
        return out


class ForecastingDataset:
    """ Wrapper for dataset pickled by accuracy_checker.

    Arguments:
        data_path (str): path to .pickle dataset
    """
    def __init__(self, data_path):
        self.data = read_annotation(data_path)

    def get_total_time(self):
        """ Get prediction interval size.

        Returns:
            total_time (int): prediction interval size.
        """
        return len(self.data[0].outputs.flatten())

    def __len__(self):
        """ Get dataset size.

        Returns:
            size (int): dataset size.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """ Get sample.

        Arguments:
            idx (idx): idx of the sample.

        Returns:
            sample: get annotation object.
        """
        try:
            return self.data[idx]
        except KeyError:
            raise IndexError


class ForecastingActors:
    """ Container for actors management.

    Arguments:
        ax (Axes): subplot.
        quantiles (list<str>): quantiles list.
        total_time (int): prediction interval size.
    """
    def __init__(self, ax, quantiles, total_time):
        self.ax = ax
        self.quantiles = quantiles
        self.total_time = total_time
        self._init_curves()

    def update_curves(self, preds, gt):
        """ Update curves.

        Arguments:
            preds (dict<np.array>): dict of predicted quantiles.
            gt (Annotation): annotation object.
        """
        self.curves["gt"].set_ydata(gt.inorm(gt.outputs.flatten()))
        for key, val in preds.items():
            self.curves[key].set_ydata(gt.inorm(val.flatten()))
        return self.curves.values()

    def _init_curves(self):
        self.curves = OrderedDict()
        self.curves["gt"], = self.ax.plot([0] * self.total_time, label="gt")
        for q in self.quantiles:
            self.curves[q], = self.ax.plot([0] * self.total_time, label=q)


def build_argparser():
    """ Build argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=Path,
                        help="Required. Path to an .xml file with a trained model")
    parser.add_argument('--input-name', type=str, default='timestamps',
                        help='Optional. Name of the models input node.')
    parser.add_argument('--output-name', type=str, default='quantiles',
                        help='Optional. Name of the models output node.')
    parser.add_argument('-i', '--input', type=str,
                        help='Required. Path to the dataset file in .pickle format.')
    parser.add_argument('--quantiles', type=str, default='p10,p50,p90',
                        help='Optional. Names of predicted quantiles.')
    return parser


def main(args):
    log.basicConfig(format="[ %(levelname)s ] [ %(name)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    logger = log.getLogger("main")
    logger.info("creating model")

    quantiles = args.quantiles.split(",")
    model = ForecastingEngine(
        model_xml=args.model,
        model_bin=args.model.with_suffix(".bin"),
        input_name=args.input_name,
        output_name=args.output_name,
        quantiles=quantiles
    )
    dataset = ForecastingDataset(args.input)

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('time_series_forecasting')
    ax.get_yaxis().set_visible(False)
    actors = ForecastingActors(
        ax=ax,
        quantiles=quantiles,
        total_time=dataset.get_total_time()
    )

    def animate(idx):
        sample = dataset[idx]
        preds = model(sample.inputs)
        curves = actors.update_curves(preds, sample)
        ax.relim()
        ax.autoscale_view(True, True, True)
        return curves

    _ = animation.FuncAnimation(
        fig,
        animate,
        frames=len(dataset),
        interval=50,
        blit=True
    )

    ax.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
