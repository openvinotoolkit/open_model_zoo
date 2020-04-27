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
  [Evaluator code](sequential_action_recognition_evaluator.py)
  Configuration file examples:
    * [action-recognition-0001-encoder](../../../configs/action-recognition-0001-encoder.yml) - running full pipeline of action recognition model.
    * [action-recognition-0001-decoder](../../../configs/action-recognition-0001-decoder.yml) - running only decoder stage with dumped embeddings of encoder.

* **MTCNN Evaluator** shows how to run MTCNN model.
  [Evaluator code](mtcnn_evaluator.py)
  Configuration file examples:
    * [mtcnn-p](../../../configs/mtcnn-p.yml) - running proposal stage of MTCNN as usual model.
    * [mtcnn-r](../../../configs/mtcnn-r.yml) - running only refine stage of MTCNN using dumped proposal stage results.
    * [mtcnn-o](../../../configs/mtcnn-o.yml) - running full MTCNN pipeline.

* **Text Spotting Evaluator** demonstrates how to evaluate text-spotting-0002 model via Accuracy Checker.
  [Evaluator code](text_spotting_evaluator.py)
  Configuration file examples:
    * [text-spotting-0002](../../../configs/text-spotting-0002.yml)
