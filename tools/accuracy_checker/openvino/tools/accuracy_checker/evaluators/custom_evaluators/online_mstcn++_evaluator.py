from .base_custom_evaluator import BaseCustomEvaluator

def create_model(model_config, launcher, data_source, launcher_model_mapping, suffix=None, delayed_model_loading=False):
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, data_source, suffix, delayed_model_loading)

class MSTCNEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
            super().__init__(dataset_config, launcher, orig_config)
            self.model = model      
    
    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, launcher_config = cls.get_dataset_and_launcher_info(config)
        data_source = dataset_config[0].get('data_source', None)

        model = create_model()

        adapter.output_blob = model.output_blob
        return cls(dataset_config, launcher, adapter, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        annotation, identifiers = self.get_dataset_info(self.dataset)
        for batch_id, (batch_annotation, batch_identifiers) in enumerate(zip(annotation, identifiers)):
            batch_inputs_images = self.model.rgb_model.prepare_data(batch_identifiers)
            batch_inputs_flow = self.model.flow_model.prepare_data(batch_identifiers)

            extr_batch_inputs_images, _ = extract_image_representations([batch_inputs_images])
            extr_batch_inputs_flow, _ = extract_image_representations([batch_inputs_flow])

            batch_raw_prediction_rgb = self.model.rgb_model.predict(batch_identifiers, extr_batch_inputs_images)
            batch_raw_prediction_flow = self.model.flow_model.predict(batch_identifiers, extr_batch_inputs_flow)
            batch_raw_out = self.combine_predictions(batch_raw_prediction_rgb, batch_raw_prediction_flow)

            batch_prediction = self.adapter.process([batch_raw_out], identifiers, [{}])

            if self.metric_executor.need_store_predictions:
                self._annotations.extend([batch_annotation])
                self._predictions.extend(batch_prediction)

            if self.metric_executor:
                self.metric_executor.update_metrics_on_batch(
                    [batch_id], [batch_annotation], batch_prediction
                )
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)