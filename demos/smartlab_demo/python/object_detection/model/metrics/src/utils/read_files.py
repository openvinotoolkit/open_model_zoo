import json
import os


class File(object):
    """[summary]

    Args:
        object ([type]): [description]
    """
    def __init__(self, filepath: str) -> None:
        if not filepath.endswith('.json'):
            raise ValueError("filepath must be an .json file:", filepath)

        self.filepath = filepath

    def read(self) -> dict:
        with open(self.filepath, "r") as f:
            data = json.load(f)

        return data


if __name__ == "__main__":

    this_dir = os.path.dirname(os.path.abspath(__file__))
    example_file = os.path.join(this_dir, 'example.json')

    annot_file = File(example_file)
    data = annot_file.read()
