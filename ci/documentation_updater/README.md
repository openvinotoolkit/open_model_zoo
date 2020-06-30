# Documentation updater

This script updates description in `model.yml` file according to description in markdown documentation from section *Use Case and High-Level Description*.

## Prerequisites

Install `ruamel.yaml` package:
```
pip install ruamel.yaml
```

## Usage

To update description of single model:
```
python documentation_updater.py -d <OMZ dir>/models/public/<model dir> --mode update
```

To check descriptions of all public models:
```
python documentation_updater.py -d <OMZ dir>/models/public
```
