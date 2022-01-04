from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseOpenVINOModel
from ...data_readers import create_reader
from collections import OrderedDict
import numpy as np


class online_MSTCN_plus(BaseOpenVINOModel):
    def __init__(self, network_info, launcher):
        super(online_MSTCN_plus, self).__init__(network_info, launcher)
        reader_config = network_info.get('reader', {})
        self.reader = create_reader(reader_config)

    def infer(self, feed_dict):
        return self.exec_network.infer(feed_dict)

    def prepare_data(self, data):
        return self.reader(data)

    def predict(self, identifiers, input_data):
        input_dict = {'input': input_data[0].data, 'fhis_in_0': input_data[1].data[0],
                      'fhis_in_1': input_data[1].data[1], 'fhis_in_2': input_data[1].data[2],
                      'fhis_in_3': input_data[1].data[3]}
        return self.infer(input_dict)


class MSTCNEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        self.metric_result = []

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, launcher_config = cls.get_dataset_and_launcher_info(config)
        model = online_MSTCN_plus(config.get('network_info', {}), launcher)

        return cls(dataset_config, launcher, model, orig_config)

    @staticmethod
    def get_dataset_info(dataset):
        identifiers = dataset.identifiers
        annotation_reader = create_reader(
            {'type': 'numpy_reader', 'data_source': dataset.dataset_config["data_source"]})
        annotation = annotation_reader(identifiers)
        return annotation, identifiers

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        annotation, identifiers = self.get_dataset_info(self.dataset)
        for batch_id in range(len(identifiers) // 2):
            batch_annotation = [item for item in annotation.data[batch_id + 2]] + [annotation.data[batch_id]]
            batch_inputs_1 = self.model.prepare_data("input_1_%d.npy" % batch_id)
            batch_inputs_2 = self.model.prepare_data("input_2_%d.npz" % batch_id)
            batch_prediction = list(self.model.predict(batch_id, [batch_inputs_1, batch_inputs_2]).values())

            metrics_result = OrderedDict()
            result = np.mean(
                [np.linalg.norm(batch_prediction[i] - batch_annotation[i]) for i in range(len(batch_prediction))])
            metrics_result[batch_id] = result
            self.metric_result.append(result)
            if output_callback:
                output_callback(batch_prediction, metrics_result=metrics_result,
                                element_identifiers=batch_id, dataset_indices=batch_id)
            self._update_progress(progress_reporter, metric_config, batch_id, 1, csv_file)

    def extract_metrics_results(self, print_results=True, ignore_results_formatting=False,
                                ignore_metric_reference=False):

        extracted_meta = []
        return self.metric_result, extracted_meta
