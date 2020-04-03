# Documentation updater

This script updates description in `model.yml` file according to description in markdown documentation from section *Use Case and High-Level Description*.

## Prerequisites

Install `ruamel.yaml` package:
```
pip install ruamel.yaml
```

## Usage
```
usage: documentation_updater.py [-h] [-r README_DIR] [-c CONFIG_DIR]
                                [--deprecated_representation DEPRECATED_REPRESENTATION]
                                [-l LIST] [-o OUT_FILE]
                                [--regime {check,update}]
                                [--log-level {critical,error,warning,info,debug}]

optional arguments:
  -h, --help            show this help message and exit
  -r README_DIR, --readme-dir README_DIR
                        Path to root directory with models descriptions
  -c CONFIG_DIR, --config-dir CONFIG_DIR
                        Path to root directory with topologies configs (by
                        default used directory from "--readme-dir" key
  --deprecated_representation DEPRECATED_REPRESENTATION
                        Used for old topology's representation
  -l LIST, --list LIST  DEPRECATED: file with topologies list
  -o OUT_FILE, --out-file OUT_FILE
                        DEPRECATED: output file with topologies list (by
                        default used original file from --out-file key)
  --regime {check,update}
                        Script work regime: "check" only finds diffs, "update"
                        - updates values
  --log-level {critical,error,warning,info,debug}
                        Level of logging
```

## Examples

To update descrtiption of single model:

```
python documentation_updater.py -r <OMZ dir>/models/public/<model dir>
```

To update descriptions of all public models:
```buildoutcfg
python documentation_updater.py -r <OMZ dir>/models/public 
```