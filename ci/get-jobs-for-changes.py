#!/usr/bin/env python3

# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A script that prints a JSON description of the CI jobs necessary to validate
the changes made between a base commit and the current commit.

The output format is a an object where each key is the identifier of the job
and the corresponding value represents that job's parameters. The value
is usually just "true" (which just means that the job should be run), but
for the "models" job the value can be an array of names of models that should
be validated.
"""

import argparse
import json
import re
import subprocess
import sys

from pathlib import Path, PurePosixPath

OMZ_ROOT = Path(__file__).resolve().parents[1]

RE_ATTRIB_NAME = re.compile(r"omz\.ci\.job-for-change\.(.+)")

def group_by_n(iterable, n):
    return zip(*[iter(iterable)] * n)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_commit', metavar='COMMIT')
    args = parser.parse_args()

    git_diff_output = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", args.base_commit + "...HEAD"],
        cwd=OMZ_ROOT)
    changed_files = list(map(PurePosixPath, git_diff_output.decode()[:-1].split("\0")))

    models_dir = PurePosixPath("models")

    jobs = {}

    for changed_file in changed_files:
        if models_dir in changed_file.parents:
            if changed_file.name == "model.yml":
                if (OMZ_ROOT / changed_file).exists(): # it might've been deleted in the branch
                    jobs.setdefault("models", set()).add(changed_file.parent.name)
                else:
                    # make sure no models.lst files reference the deleted model
                    jobs["models_lst"] = True

            if changed_file.suffix == ".py":
                for parent in changed_file.parents:
                    if (OMZ_ROOT / parent / "model.yml").exists():
                        jobs.setdefault("models", set()).add(parent.name)

    if "models" in jobs:
        jobs["models"] = sorted(jobs["models"]) # JSON can't work with a set

    git_check_attr_output = subprocess.run(
        ["git", "check-attr", "--stdin", "-z", "--all"],
        input=git_diff_output, stdout=subprocess.PIPE, check=True, cwd=OMZ_ROOT).stdout

    for path, attribute, value in group_by_n(git_check_attr_output.decode()[:-1].split("\0"), 3):
        attribute_match = RE_ATTRIB_NAME.fullmatch(attribute)
        if value != 'unset' and attribute_match:
            jobs[attribute_match.group(1)] = True

    json.dump(jobs, sys.stdout)
    print()

if __name__ == "__main__":
    main()
