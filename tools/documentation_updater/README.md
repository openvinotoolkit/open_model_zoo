# Documentation updater

This script updates description in `model.yml` file according to description in markdown documentation from section *Use Case and High-Level Description*.

## Prerequisites

Install `ruamel.yaml` package:
```
pip install ruamel.yaml
```

## Usage
```
usage: documentation_updater.py [-h] [-d MODEL_DIR] [--mode {check,update}]
                                [--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}]
                                [--ignored-files IGNORED_FILES]
                                [--ignored-files-list IGNORED_FILES_LIST]

optional arguments:
  -h, --help            show this help message and exit
  -d MODEL_DIR, --model-dir MODEL_DIR
                        Path to root directory with models documentation and
                        configuration files
  --mode {check,update}
                        Script work mode: "check" only finds diffs, "update" -
                        updates values
  --log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        Level of logging
  --ignored-files IGNORED_FILES
                        List of files which will be ignored
  --ignored-files-list IGNORED_FILES_LIST
                        Path to file with ignored files

```

## Examples

To update descrtiption of single model:

```
python documentation_updater.py -d <OMZ dir>/models/public/<model dir>
```

To check descriptions of all public models:
```buildoutcfg
python documentation_updater.py -d <OMZ dir>/models --mode check
```
