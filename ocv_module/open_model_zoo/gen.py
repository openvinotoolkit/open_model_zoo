# This script is used to generate the models.

import yaml

def generate(entry):
    config = {}

    name = topology['name'].replace('-', '_')
    config['description'] = topology['description'].replace('\n', ' ').replace('\"', '\\"')
    config['license'] = topology['license']

    files = topology['files']
    assert(len(files) > 0)
    config['model_url'] = files[0]['source']
    config['model_sha256'] = files[0]['sha256']
    config['model_name'] = files[0]['name']

    s = ', '.join(['{"%s", "%s"}' % (key, value) for key, value in config.items()])

    print("""
Ptr<Topology> %s()
{
    Ptr<Topology> t(new Topology({%s}));
    t->download();
    return t;
}
    """ % (name, s))

# with open('/home/dkurt/open_model_zoo/tools/downloader/list_topologies.yml', 'rt') as f:
with open('list_topologies.yml', 'rt') as f:
    content = yaml.safe_load(f)
    for topology in content['topologies']:
        generate(topology)


# def tokenizeBlock(content, i, level):
#     # Loop over lines
#     while True:
#         # Skip indentation
#         for j in range(level * 2):
#             assert(content[i] == ' ')
#             i += 1
#
#
# def tokenize(content):
#     token = ''
#     tokens = []
#     isComment = False
#
#     i = 0
#     while i < len(content):
#         # Skip empty lines and indentations
#         if content[i] == '\n':
#             if not isComment:
#                 if token:
#                     tokens.append(token)
#                 while content[i] in [' ', '\n']:
#                     i += 1
#                     if i == len(content):
#                         return tokens
#             isComment = False
#
#         symbol = content[i]
#
#         if isComment:
#             if symbol == '\n':
#                 isComment = False
#         elif symbol in [':', '\n']:
#             if token:
#                 tokens.append(token)
#                 token = ''
#             # if symbol == ':':
#             #     i += 1
#             #     while content[i] == ' ':
#             #         i += 1
#             #         if i == len(content):
#             #             return tokens
#         elif symbol == '#':
#             isComment = True
#         else:
#             token += symbol
#
#         i += 1
#
#     return tokens
