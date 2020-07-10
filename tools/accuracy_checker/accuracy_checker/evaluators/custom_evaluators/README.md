# Custom Evaluators for Accuracy Checker
Standard Accuracy Checker validation pipeline: Annotation Reading -> Data Reading -> Preprocessing -> Inference -> Postprocessing -> Metrics.
In some cases it can be unsuitable (e.g. if you have sequence of models). You are able to customize validation pipeline using own evaluator.
Suggested approach based on writing python module which will describe validation approach

## Implementation
Adding new evaluator process similar with adding any other entities in the tool.
Custom evaluator is the class which should be inherited from BaseEvaluator and overwrite all abstract methods.

The most important methods for overwriting:

* `from_configs` - create new instance using configuration dictionary.
* `process_dataset` - determine validation cycle across all data batches in dataset.
* `compute_metrics` - metrics evaluation after dataset processing.
* `reset` - reset evaluation progress

## Configuration
Each custom evaluation config should start with keyword `evaluation` and contain:
 * `name` - model name
 * `module` - evaluation module for loading.
Before running, please make sure that prefix to module added to your python path or use `python_path` parameter in config for it specification.
Optionally you can provide `module_config` section which contains config for custom evaluator (Depends from realization, it can contains evaluator specific parameters).

## Examples
* **Sequential Action Recognition Evaluator** demonstrates how to run Action Recognition models with encoder + decoder architecture.
  <a href="https://github.com/opencv/open_model_zoo/blob/develop/tools/accuracy_checker/accuracy_checker/evaluators/custom_evaluators/sequential_action_recognition_evaluator.py">Evaluator code</a>.
  Configuration file example: <a href="https://github.com/opencv/open_model_zoo/blob/develop/tools/accuracy_checker/configs/action-recognition-0001.yml">action-recognition-0001</a>.

* **MTCNN Evaluator** shows how to run MTCNN model.
  <a href="https://github.com/opencv/open_model_zoo/blob/develop/tools/accuracy_checker/accuracy_checker/evaluators/custom_evaluators/mtcnn_evaluator.py">Evaluator code</a>.
  Configuration file example: <a href="https://github.com/opencv/open_model_zoo/blob/develop/tools/accuracy_checker/configs/mtcnn.yml">mtcnn</a>.

* **Text Spotting Evaluator** demonstrates how to evaluate the `text-spotting-0002` model via Accuracy Checker.
  <a href="https://github.com/opencv/open_model_zoo/blob/develop/tools/accuracy_checker/accuracy_checker/evaluators/custom_evaluators/text_spotting_evaluator.py">Evaluator code</a>.
  Configuration file example: <a href="https://github.com/opencv/open_model_zoo/blob/develop/tools/accuracy_checker/configs/text-spotting-0002.yml">text-spotting-0002</a>.
