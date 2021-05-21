# Copyright (c) 2021 Intel Corporation
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

import collections

import mistune

_parse_markdown = mistune.create_markdown(renderer=mistune.AstRenderer())

def _get_all_ast_nodes(ast_nodes):
    for node in ast_nodes:
        yield node
        if 'children' in node:
            # workaround for https://github.com/lepture/mistune/issues/269
            if isinstance(node['children'], str):
                yield {'type': 'text', 'text': node['children']}
            else:
                yield from _get_all_ast_nodes(node['children'])

def _get_text_from_ast(ast_nodes):
    def get_text_from_node(node):
        if node['type'] == 'text':
            return node['text']
        elif node['type'] == 'link':
            return _get_text_from_ast(node['children'])
        raise RuntimeError(f'unsupported node type: {node["type"]}')

    return ''.join(map(get_text_from_node, ast_nodes))

ExternalReference = collections.namedtuple('ExternalReference', ['type', 'url'])

class DocumentationPage:
    def __init__(self, markdown_text):
        self._ast = ast = _parse_markdown(markdown_text)

        self._title = None
        if ast and ast[0]['type'] == 'heading' and ast[0]['level'] == 1:
            self._title = _get_text_from_ast(ast[0]['children'])

    @property
    def title(self):
        return self._title

    def external_references(self):
        for node in _get_all_ast_nodes(self._ast):
            if node['type'] == 'image':
                yield ExternalReference('image', node['src'])
            elif node['type'] == 'link':
                yield ExternalReference('link', node['link'])

    def code_spans(self):
        for node in _get_all_ast_nodes(self._ast):
            if node['type'] == 'codespan':
                yield node['text']

    def html_fragments(self):
        for node in _get_all_ast_nodes(self._ast):
            if node['type'] == 'inline_html':
                yield node['text']
