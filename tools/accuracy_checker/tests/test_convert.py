"""
Copyright (c) 2018-2024 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import threading
import warnings
from accuracy_checker.annotation_converters.convert import AtomicWriteFileHandle

def thread_access_file(file_path, data_dict, thread_id, write_lines):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            read_lines = len(file.readlines())
            # when a new thread reads a file, all lines must already be written
            if read_lines != write_lines:
                warn_message = f"Thread {thread_id}: Incorrect number of lines read from {file_path} ({read_lines} != {write_lines})"
                warnings.warn(warn_message)
                data_dict['assert'] = warn_message
    else:
        with AtomicWriteFileHandle(file_path, 'wt') as file:
            for i in range(write_lines):
                file.write(f"Thread {thread_id}:Line{i} {data_dict[thread_id]}\n")

class TestAtomicWriteFileHandle:

    def test_multithreaded_atomic_file_write(self):
        target_file_path = "test_atomic_file.txt"
        threads = []
        num_threads = 10
        write_lines = 10
        data_chunks = [f"Data chunk {i}" for i in range(num_threads)]
        threads_dict = {i: data_chunks[i] for i in range(len(data_chunks))}

        if os.path.exists(target_file_path):
            os.remove(target_file_path)

        for i in range(num_threads):
            thread = threading.Thread(target=thread_access_file, args=(target_file_path, threads_dict, i, write_lines))
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        with open(target_file_path, 'r') as file:
            lines = file.readlines()

        os.remove(target_file_path)

        # check asserts passed from threads
        assert 'assert' not in threads_dict.keys() , threads_dict['assert']

        assert sum(1 for line in lines for data_chunk in data_chunks if data_chunk in line) == write_lines, f"data_chunks data not found in the {target_file_path} file"
