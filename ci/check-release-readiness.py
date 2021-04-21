#!/usr/bin/env python3

"""
This script does various consistency checks that should pass before a release
but that can't pass in every commit due to process imperfections.

Specifically, we publish Intel models separately from their documentation and
demo support code. Because of that, we can have situations where a model's
config file is added, but documentation isn't, or vice versa. These need to be
rectified before a release, so the script will warn about them.

Also, we can only add a model to models.lst files once its configs are added.
In order to not forget to do that, we can add TODO comments and the script
will warn if any models.lst files contain such comments.
"""

import re
import sys

from pathlib import Path

OMZ_ROOT = Path(__file__).resolve().parent.parent

RE_TODO_COMMENT = re.compile(r'#.*\b(?:TODO|FIXME)\b')

def main():
    all_passed = True

    def complain(format, *args):
        nonlocal all_passed
        print(format.format(*args), file=sys.stderr)
        all_passed = False

    for model_dir in OMZ_ROOT.glob('models/*/*/'):
        has_model_yml = (model_dir / 'model.yml').is_file()
        has_composite_model_yml = (model_dir / 'composite-model.yml').is_file()

        has_config = has_model_yml or has_composite_model_yml

        has_doc = (model_dir / 'README.md').is_file()

        if has_config and not has_doc:
            complain('model {} has no documentation', model_dir.name)

        if has_doc and not has_config:
            complain('model {} has no config file', model_dir.name)

        if has_composite_model_yml:
            if not list(model_dir.glob('*/model.yml')):
                complain('composite model {} has no components', model_dir.name)

    for models_lst_path in OMZ_ROOT.glob('demos/**/models.lst'):
        with models_lst_path.open() as models_lst:
            for line_num, line in enumerate(models_lst):
                if RE_TODO_COMMENT.search(line):
                    complain('{}:{}: line contains TODO comment',
                        models_lst_path.relative_to(OMZ_ROOT), line_num + 1)

    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()
