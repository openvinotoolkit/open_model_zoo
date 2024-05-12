import numpy as np

from ..config import StringField, ConfigError
from .data_reader import BaseReader
from ..utils import get_path, read_json
import dgl


class DGLGraphReader(BaseReader):
    __provider__ = 'graph(dgl)_reader'

    def configure(self):
        if not self.data_source:
            if not self._postpone_data_source:
                raise ConfigError('data_source parameter is required to create "{}" '
                                  'data reader and read data'.format(self.__provider__))
        else:
            self.data_source = get_path(self.data_source, is_directory=False)

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        print('read data')
        
        graph = dgl.data.utils.load_graphs(Path(self.graph_path).__str__())
        g = graph[0][0]

        return g
