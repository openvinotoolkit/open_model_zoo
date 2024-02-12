# Copyright (c) 2021-2024 Intel Corporation
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

import re
import shutil

from omz_tools.download_engine import base, validation

class Postproc(base.TaggedBase):
    types = {}

class PostprocRegexReplace(Postproc):
    def __init__(self, file, pattern, replacement, count):
        self.file = file
        self.pattern = pattern
        self.replacement = replacement
        self.count = count

    @classmethod
    def deserialize(cls, postproc):
        return cls(
            validation.validate_relative_path('"file"', postproc['file']),
            re.compile(validation.validate_string('"pattern"', postproc['pattern'])),
            validation.validate_string('"replacement"', postproc['replacement']),
            validation.validate_nonnegative_int('"count"', postproc.get('count', 0)),
        )

    def apply(self, reporter, output_dir):
        postproc_file = output_dir / self.file

        reporter.print_section_heading('Replacing text in {}', postproc_file)

        postproc_file_text = postproc_file.read_text(encoding='utf-8')

        orig_file = postproc_file.with_name(postproc_file.name + '.orig')
        if not orig_file.exists():
            postproc_file.replace(orig_file)

        postproc_file_text, num_replacements = self.pattern.subn(
            self.replacement, postproc_file_text, count=self.count)

        if num_replacements == 0:
            raise RuntimeError('Invalid pattern: no occurrences found')

        if self.count != 0 and num_replacements != self.count:
            raise RuntimeError('Invalid pattern: expected at least {} occurrences, but only {} found'.format(
                self.count, num_replacements))

        postproc_file.write_text(postproc_file_text, encoding='utf-8')

Postproc.types['regex_replace'] = PostprocRegexReplace

class PostprocUnpackArchive(Postproc):
    def __init__(self, file, format):
        self.file = file
        self.format = format

    @classmethod
    def deserialize(cls, postproc):
        return cls(
            validation.validate_relative_path('"file"', postproc['file']),
            validation.validate_string('"format"', postproc['format']),
        )

    def apply(self, reporter, output_dir):
        postproc_file = output_dir / self.file

        reporter.print_section_heading('Unpacking {}', postproc_file)

        shutil.unpack_archive(str(postproc_file), str(postproc_file.parent), self.format)
        postproc_file.unlink()  # Remove the archive

Postproc.types['unpack_archive'] = PostprocUnpackArchive
