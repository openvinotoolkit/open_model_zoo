from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseCascadeModel, create_model, create_encoder

class DGLEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        print('create evaluator')
        self.model = model
        # if hasattr(self.model.decoder, 'adapter'):
        #     self.adapter_type = self.model.decoder.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = DGLGraphModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        pass

class DGLGraphModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
    
    def predict(self, identifiers, input_data, encoder_callback=None):
        pass
    
    def reset(self):
        pass

    def save_encoder_predictions(self):
        pass

    def _add_raw_encoder_predictions(self, encoder_prediction):
        pass