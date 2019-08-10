# This script is used to generate the models.

import yaml
import sys
import re


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


# List of models which have versions (mostly DLDT models). For such kind of models
# with the highest version we will add short names. In example, for
# "face_detection_retail_0004" and "face_detection_retail_0005" method with name
# "face_detection_retail" returns result of "face_detection_retail_0005". The same
# approach for precisions: "face_detection_retail_fp16" for "face_detection_retail_0005_fp16"
versionedNames = {}

def registerVersionedName(fullName, precision=''):
    global versionedNames

    matches = re.search('(.+)_(\d{4})$', fullName)
    if matches:
        shortName = matches.group(1) + ('_%s' % precision if precision else '')
        version = matches.group(2)
        if not shortName in versionedNames or int(version) > int(versionedNames[shortName][0]):
            originName = fullName + ('_%s' % precision if precision else '')
            versionedNames[shortName] = [version, originName]


def generate(topology, topologies_hdr, py_impl, cpp_impl):
    name = topology['name'].replace('-', '_').replace('.', '_')

    # DLDT models come with multiple files foe different precision
    files = topology['files']
    assert(len(files) > 0), topology['name']
    if len(files) > 2:
        assert(topology['framework'] == 'dldt'), topology['name']
        assert(len(files) % 2 == 0), topology['name']

        for i in range(len(files) / 2):
            subTopology = topology.copy()
            subTopology['files'] = [files[i * 2], files[i * 2 + 1]]
            # Detect precision by the first file
            precision = subTopology['files'][0]['name']
            precision = precision[:precision.find('/')].lower()
            if precision != 'fp32':  # Keep origin name for FP32
                subTopology['name'] += '_' + precision
                registerVersionedName(name, precision)
            generate(subTopology, topologies_hdr, py_impl, cpp_impl)
        return


    registerVersionedName(name)

    config = {}
    config['description'] = topology['description'].replace('\n', ' ') \
                                                   .replace('\\', '\\\\') \
                                                   .replace('\"', '\\"')

    config['license'] = topology['license']
    config['framework'] = topology['framework']

    config['topology_name'] = name
    if 'model_optimizer_args' in topology:
        config['model_optimizer_args'] = ' '.join(topology['model_optimizer_args'])

    fileURL, fileSHA, fileName = getSource(files[0])
    if fileName.endswith('tar.gz'):
        config['archive_url'], config['archive_sha256'], config['archive_name'] = fileURL, fileSHA, fileName
    else:
        config['config_url'], config['config_sha256'], config['config_path'] = fileURL, fileSHA, fileName
        if len(files) > 1:
            config['model_url'], config['model_sha256'], config['model_path'] = getSource(files[1])

    s = ', '.join(['{"%s", "%s"}' % (key, value) for key, value in config.items()])

    for impl in [py_impl, cpp_impl]:
        impl.write("""
    Topology %s(bool download)
    {
        Topology t({%s});
        if (download)
            t.download();
        return t;
    }\n""" % (name, s))

    topologies_hdr.write('    CV_EXPORTS_W Topology %s(bool download = true);\n' % name)

list_topologies = sys.argv[1]
topologies_hdr = open(sys.argv[2], 'wt')
py_impl = open(sys.argv[3], 'wt')
cpp_impl = open(sys.argv[4], 'wt')

topologies_hdr.write("#ifndef __OPENCV_OPEN_MODEL_ZOO_TOPOLOGIES_HPP__\n")
topologies_hdr.write("#define __OPENCV_OPEN_MODEL_ZOO_TOPOLOGIES_HPP__\n\n")
topologies_hdr.write("using namespace cv::open_model_zoo;\n\n")
topologies_hdr.write("namespace cv { namespace open_model_zoo { namespace topologies {\n")

py_impl.write("#ifdef HAVE_OPENCV_OPEN_MODEL_ZOO\n\n")
py_impl.write("namespace cv { namespace open_model_zoo { namespace topologies {")

cpp_impl.write("#include <opencv2/open_model_zoo.hpp>\n")
cpp_impl.write("#include <opencv2/open_model_zoo/topologies.hpp>\n\n")
cpp_impl.write("namespace cv { namespace open_model_zoo { namespace topologies {")

with open(list_topologies, 'rt') as f:
    content = yaml.safe_load(f)
    for topology in content['topologies']:
        generate(topology, topologies_hdr, py_impl, cpp_impl)

    # Register DLDT aliases
    for alias, data in versionedNames.items():
        _, originName = data

        for impl in [py_impl, cpp_impl]:
            impl.write("""
    Topology %s(bool download)
    {
        return %s(download);
    }\n""" % (alias, originName))

        topologies_hdr.write('    CV_EXPORTS_W Topology %s(bool download = true);\n' % alias)

    py_impl.write("}}}  // namespace cv::open_model_zoo::topologies\n\n")
    py_impl.write("#endif  // HAVE_OPENCV_OPEN_MODEL_ZOO")
    cpp_impl.write("}}}  // namespace cv::open_model_zoo::topologies\n\n")

topologies_hdr.write("}}}  // namespace cv::open_model_zoo::topologies\n\n")
topologies_hdr.write("#endif  // __OPENCV_OPEN_MODEL_ZOO_TOPOLOGIES_HPP__")

topologies_hdr.close()
py_impl.close()
cpp_impl.close()
