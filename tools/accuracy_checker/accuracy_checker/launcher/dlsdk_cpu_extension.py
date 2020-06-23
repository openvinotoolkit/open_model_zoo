from ..config import PathField
from ..utils import get_path
from pathlib import Path


class CPUExtensionPathField(PathField):
    def __init__(self, **kwargs):
        super().__init__(is_directory=False, **kwargs)

    def validate(self, entry, field_uri=None):
        if entry is None:
            return

        field_uri = field_uri or self.field_uri
        validation_entry = ''
        try:
            validation_entry = Path(entry)
        except TypeError:
            self.raise_error(entry, field_uri, "values is expected to be path-like")
        is_directory = False
        if validation_entry.parts[-1] == 'AUTO':
            validation_entry = validation_entry.parent
            is_directory = True
        try:
            get_path(validation_entry, is_directory)
        except FileNotFoundError:
            self.raise_error(validation_entry, field_uri, "path does not exist")
        except NotADirectoryError:
            self.raise_error(validation_entry, field_uri, "path is not a directory")
        except IsADirectoryError:
            self.raise_error(validation_entry, field_uri, "path is a directory, regular file expected")
