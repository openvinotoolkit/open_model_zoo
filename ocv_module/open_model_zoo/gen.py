# This script is used to generate the models.

import yaml
import sys


def getSource(entry):
    name = entry['name']
    sha = entry['sha256']
    source = entry['source']
    if isinstance(source, str):
        url = source
    elif isinstance(source, dict):
        sourceType = source['$type']
        if sourceType == 'google_drive':
            url = 'https://drive.google.com/uc?export=download&id=' + source['id']
        else:
            print('Unknown source type: %s', sourceType)
            sys.exit(1)
    else:
        print('Unexpected source instance: %s', type(source))
        sys.exit(1)

    return url, sha, name


def generate(topology, output_hdr, impl_hdr):
    config = {}

    name = topology['name'].replace('-', '_').replace('.', '_')
    config['description'] = topology['description'].replace('\n', ' ') \
                                                   .replace('\\', '\\\\') \
                                                   .replace('\"', '\\"')

    config['license'] = topology['license']
    if 'model_optimizer_args' in topology:
        config['model_optimizer_args'] = ' '.join(topology['model_optimizer_args'])

    files = topology['files']
    assert(len(files) > 0)

    config['config_url'], config['config_sha256'], config['config_name'] = getSource(files[0])
    if len(files) > 1:
        config['model_url'], config['model_sha256'], config['model_name'] = getSource(files[1])

    s = ', '.join(['{"%s", "%s"}' % (key, value) for key, value in config.items()])

    impl_hdr.write("""
    Ptr<Topology> %s()
    {
        Ptr<Topology> t(new Topology({%s}));
        t->download();
        return t;
    }\n""" % (name, s))

    output_hdr.write('    CV_EXPORTS_W Ptr<Topology> %s();\n' % name)


list_topologies = sys.argv[1]
output_hdr_path = sys.argv[2]
impl_hdr_path = sys.argv[3]

with open(output_hdr_path, 'wt') as output_hdr:
    output_hdr.write("#ifndef __OPENCV_OPEN_MODEL_ZOO_TOPOLOGIES_HPP__\n")
    output_hdr.write("#define __OPENCV_OPEN_MODEL_ZOO_TOPOLOGIES_HPP__\n\n")
    output_hdr.write("namespace cv { namespace open_model_zoo {\n")

    with open(impl_hdr_path, 'wt') as impl_hdr:
        impl_hdr.write("#ifdef HAVE_OPENCV_OPEN_MODEL_ZOO\n\n")
        impl_hdr.write("namespace cv { namespace open_model_zoo {")

        with open(list_topologies, 'rt') as f:
            content = yaml.safe_load(f)
            for topology in content['topologies']:
                generate(topology, output_hdr, impl_hdr)

        impl_hdr.write("}}  // namespace cv::open_model_zoo\n\n")
        impl_hdr.write("#endif  // HAVE_OPENCV_OPEN_MODEL_ZOO")

    output_hdr.write("}}  // namespace cv::open_model_zoo\n\n")
    output_hdr.write("#endif  // __OPENCV_OPEN_MODEL_ZOO_TOPOLOGIES_HPP__")


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
