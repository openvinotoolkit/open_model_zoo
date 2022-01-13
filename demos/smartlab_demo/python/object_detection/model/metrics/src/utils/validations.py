import json
import os
import xml.etree.ElementTree as ET

import pandas as pd
from src.utils.enumerators import CoordinatesType
from src.utils.general_utils import get_files_recursively

from .enumerators import BBFormat, BBType, FileFormat


def validate_formats(arg_format, arg_name, errors):
    """ Verify if string format that represents the bounding box format is valid.

        Parameters
        ----------
        arg_format : str
            Received argument with the format to be validated.
        arg_name : str
            Argument name that represents the bounding box format.
        errors : list
            List with error messages to be appended with error message in case an error occurs.

        Returns
        -------
        BBFormat : Enum
            If arg_format is valid, it will return the enum representing the correct format. If
            format is not valid, return None.
    """
    if arg_format.lower() == 'xywh':
        return BBFormat.XYWH
    elif arg_format.lower() == 'xyrb':
        return BBFormat.XYX2Y2
    elif arg_format.lower() == 'yolo':
        return BBFormat.YOLO
    elif arg_format.lower() in ['pascal', 'pascalvoc', 'pascal_voc', 'vocpascal', 'voc_pascal']:
        return BBFormat.PASCAL
    elif arg_format is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(
            f'argument {arg_name}: invalid value. It must be either \'xywh\', \'xyrb\', \'yolo\' or \'pascal\''
        )
        return None


def xml_contains_tags(file_path, tags):
    """ Verify if a xml file contains specific tags.

    Parameters
    ----------
    file_path : str
        Path of the file.
    tags : list
        List containing strings representing the tags to be found (e.g. ['annotation', './object', '.object/bndbox']).

    Returns
    -------
    bool
        True if XML file contains all tags, False otherwise.
    """
    total_match = 0
    for tag in tags:
        if ET.parse(file_path)._root.tag == tag:
            total_match += 1
        elif ET.parse(file_path).find(tag) not in [[], None]:
            total_match += 1
    return total_match == len(tags)


def get_all_keys(items):
    """ Get all keys in a list of dictionary.

    Parameters
    ----------
    items : list
        List of dictionaries.

    Returns
    -------
    list
        List containing all keys in the dictionary.
    """
    ret = []
    if not hasattr(items, '__iter__'):
        return ret
    if isinstance(items, str):
        return ret
    for i, item in enumerate(items):
        if isinstance(item, list):
            ret.append(get_all_keys(item))
        elif isinstance(item, dict):
            [ret.append(it) for it in item.keys() if it not in ret]
    return ret


def json_contains_tags(file_path, tags):
    """ Verify if a given JSON file contains all tags in a list.

    Parameters
    ----------
    file_path : str
        Path of the file.
    tags : list
        List containing strings representing the tags to be found.

    Returns
    -------
    bool
        True if XML file contains all tags, False otherwise.
    """
    with open(file_path, "r") as f:
        json_object = json.load(f)
    all_keys = []
    for key, item in json_object.items():
        keys = get_all_keys(item)
        if len(keys) == 0:
            all_keys.append(key)
        for k in keys:
            all_keys.append(f'{key}/{k}')

    tags_matching = 0
    for tag in tags:
        if tag in all_keys:
            tags_matching += 1
    return tags_matching == len(tags)


def csv_contains_columns(file_path, columns, sep=','):
    """ Verify if a given csv file contains all columns.

    Parameters
    ----------
    file_path : str
        Path of the file.
    columns : list
        List containing strings representing the columns to be verified.
    columns : str (optional)
        List containing strings representing the columns to be verified.

    Returns
    -------
    bool
        True if the csv file contains all columns, False otherwise.
    """
    csv = pd.read_csv(file_path, sep=sep)
    cols_1 = [col.lower() for col in list(csv.columns)]
    cols_2 = [col.lower() for col in columns]
    cols_1.sort()
    cols_2.sort()
    return cols_1 == cols_2


def is_xml(file_path):
    """ Verify by the extension if a given file path represents a XML file.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file ends with .xml, False otherwise.
    """
    return os.path.splitext(file_path)[-1].lower() == '.xml'


def is_json(file_path):
    """ Verify by the extension if a given file path represents a json file.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file ends with .json, False otherwise.
    """
    return os.path.splitext(file_path)[-1].lower() == '.json'


def is_text(file_path):
    """ Verify by the extension if a given file path represents a txt file.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file ends with .txt, False otherwise.
    """
    # Text file with annotations can be a .txt file or have no extension
    return os.path.splitext(file_path)[-1].lower() in ['.txt', '']


def is_csv(file_path):
    """ Verify by the extension if a given file path represents a csv file.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file ends with .csv, False otherwise.
    """
    return os.path.splitext(file_path)[-1].lower() == '.csv'


def is_pascal_format(file_path):
    """ Verify if a given file path represents a file with annotations in pascal format.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file contains annotations in pascal format, False otherwise.
    """
    return is_xml(file_path) and xml_contains_tags(file_path,
                                                   ['annotation', './size/width', './size/height'])


def is_imagenet_format(file_path):
    """ Verify if a given file path represents a file with annotations in imagenet format.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file contains annotations in imagenet format, False otherwise.
    """
    # "The imagenet annotations are saved in XML files in PASCAL VOC format."
    return is_pascal_format(file_path)


def is_labelme_format(file_path, allow_empty_detections=True):
    """ Verify if a given file path represents a file with annotations in labelme format.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file contains annotations in labelme format, False otherwise.
    """
    tags = ['imagePath', 'imageWidth', 'imageHeight']
    if not allow_empty_detections:
        tags.append('shapes/label')
        tags.append('shapes/points')
    return is_json(file_path) and json_contains_tags(file_path, tags)


def is_valid_coco_dir(dir):
    bb_files = get_files_recursively(dir)
    if len(bb_files) != 1:
        return False
    for file_path in bb_files:
        return verify_format(file_path, FileFormat.COCO)


def is_valid_cvat_dir(dir):
    bb_files = get_files_recursively(dir)
    if len(bb_files) != 1:
        return False
    for file_path in bb_files:
        return verify_format(file_path, FileFormat.CVAT)


def is_coco_format(file_path):
    """ Verify if a given file path represents a file with annotations in coco format.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file contains annotations in coco format, False otherwise.
    """
    return is_json(file_path) and json_contains_tags(file_path, [
        'annotations/bbox',
        'annotations/image_id',
    ])


def is_cvat_format(file_path):
    """ Verify if a given file path represents a file with annotations in cvat format.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file contains annotations in cvat format, False otherwise.
    """
    return is_xml(file_path) and xml_contains_tags(file_path, ['annotations', './image/box'])


def is_specific_text_format(file_path,
                            type_coordinates=CoordinatesType.ABSOLUTE,
                            bb_type=BBType.GROUND_TRUTH):
    if type_coordinates == CoordinatesType.ABSOLUTE:
        if bb_type == BBType.GROUND_TRUTH and is_absolute_text_format(
                file_path, num_blocks=[5], blocks_abs_values=[4]):
            return True
        if bb_type == BBType.DETECTED and is_absolute_text_format(
                file_path, num_blocks=[6], blocks_abs_values=[4]):
            return True
    elif type_coordinates == CoordinatesType.RELATIVE:
        if bb_type == BBType.GROUND_TRUTH and is_relative_text_format(
                file_path, num_blocks=[5], blocks_rel_values=[4]):
            return True
        if bb_type == BBType.DETECTED and is_relative_text_format(
                file_path, num_blocks=[6], blocks_rel_values=[4]):
            return True
    return False


def is_absolute_text_format(file_path, num_blocks=[6, 5], blocks_abs_values=[4]):
    """ Verify if a given file path represents a file with annotations in text format with absolute coordinates.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file contains annotations in text format with absolute coordinates, False otherwise.
    """
    if not is_text(file_path):
        return False
    if not is_empty_file(file_path):
        return all_lines_have_blocks(file_path,
                                     num_blocks=num_blocks) and all_blocks_have_absolute_values(
                                         file_path, blocks_abs_values=blocks_abs_values)
    return True


def is_relative_text_format(file_path, num_blocks=[6, 5], blocks_rel_values=[4]):
    if not is_text(file_path):
        return False
    if not is_empty_file(file_path):
        return all_lines_have_blocks(file_path,
                                     num_blocks=num_blocks) and all_blocks_have_relative_values(
                                         file_path, blocks_rel_values=blocks_rel_values)
    return True


def is_yolo_format(file_path, bb_types=[BBType.GROUND_TRUTH, BBType.DETECTED]):
    """ Verify if a given file path represents a file with annotations in yolo format.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file contains annotations in yolo format, False otherwise.
    """
    assert len(bb_types) > 0
    for bb_type in bb_types:
        assert bb_type in [BBType.GROUND_TRUTH, BBType.DETECTED]

    num_blocks = []
    for bb_type in bb_types:
        if bb_type == BBType.GROUND_TRUTH:
            num_blocks.append(5)
        elif bb_type == BBType.DETECTED:
            num_blocks.append(6)
    return is_text(file_path) and all_lines_have_blocks(
        file_path, num_blocks=num_blocks) and all_blocks_have_relative_values(file_path,
                                                                              blocks_rel_values=[4])


def is_openimage_format(file_path):
    """ Verify if a given file path represents a file with annotations in openimage format.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file contains annotations in openimage format, False otherwise.
    """
    return is_csv(file_path) and csv_contains_columns(
        file_path,
        columns=[
            'ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax',
            'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'
        ])


def all_lines_have_blocks(file_path, num_blocks=[]):
    """ Verify if all annotations given file path represent a text with annotations separated into num_blocks.

    Parameters
    ----------
    file_path : str
        Path of the file.
    num_blocks : list
        List containing possible amounts of blocks.
        (e.g. if your annotation file is in the format 'person 1 0.23 0.8 0.3 0.75', it contains 6 blocks)

    Returns
    -------
    bool
        True if all the annotations contains at least 1 block specified in the num_blocks , False otherwise.
    """
    with open(file_path, 'r+') as f:
        for line in f:
            line = line.replace('\n', '').strip()
            if line == '':
                continue
            passed = False
            for block in num_blocks:
                if len(line.split(' ')) == block:
                    passed = True
            if passed is False:
                return False
    return True


def all_blocks_have_absolute_values(file_path, blocks_abs_values=[]):
    """ Verify if all annotations given file path represent a text with annotations with absolute values in all blocks.

    Parameters
    ----------
    file_path : str
        Path of the file.
    blocks_abs_values : list
        List containing possible amounts of blocks.
        (e.g. if your annotation file is in the format '32 1 23 180 300 750', it contains 6 blocks with absolute values)

    Returns
    -------
    bool
        True if all the annotations in the file pass contain at least 1 block specified in the blocks_abs_values and all blocks contain absolute values. False otherwise.
    """
    with open(file_path, 'r+') as f:
        for line in f:
            line = line.replace('\n', '').strip()
            if line == '':
                continue
            passed = False
            splitted = line.split(' ')
            for block in blocks_abs_values:
                if len(splitted) < block:
                    return False
                try:
                    if float(splitted[block]) == int(float(splitted[block])):
                        passed = True
                except:
                    passed = False
            if passed is False:
                return False
    return True


def all_blocks_have_relative_values(file_path, blocks_rel_values=[]):
    """ Verify if all annotations given file path represent a text with annotations with relative values in all blocks.

    Parameters
    ----------
    file_path : str
        Path of the file.
    blocks_rel_values : list
        List containing possible amounts of blocks.
        (e.g. if your annotation file is in the format '32 1 23 180 300 750', it contains 6 blocks with relative values)

    Returns
    -------
    bool
        True if all the annotations in the file pass contain at least 1 block specified in the blocks_rel_values and all blocks contain relative values. False otherwise.
    """
    with open(file_path, 'r+') as f:
        for line in f:
            line = line.replace('\n', '').strip()
            if line == '':
                continue
            passed = False
            splitted = line.split(' ')
            for block in blocks_rel_values:
                if len(splitted) < block:
                    return False
                try:
                    float(splitted[block])
                    passed = True
                except:
                    passed = False
            if passed is False:
                return False
    return True


def is_empty_file(file_path):
    """ Verify if an annotation file is not empty.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file contains no annotations. False otherwise.
    """
    # An empty file is considered a file with empty lines or spaces
    with open(file_path, 'r+') as f:
        for line in f:
            if line.strip() != '':
                return False
    return True


def verify_format(file_path, verification_format):
    """ Verify if a file contains annotations in a specific format.

    Parameters
    ----------
    file_path : str
        Path of the file.
    verification_format : enum (FileFormat)
        Format of the file.

    Returns
    -------
    bool
        True if the file contains annotations in the given format. False otherwise.
    """
    # Given a file path with annotations, get the format of the annotations
    if os.path.isfile(file_path) is False:
        return False

    if verification_format.name == FileFormat.ABSOLUTE_TEXT.name:
        return is_absolute_text_format(file_path)

    if verification_format.name == FileFormat.PASCAL.name:
        return is_pascal_format(file_path)

    if verification_format.name == FileFormat.LABEL_ME.name:
        return is_labelme_format(file_path)

    if verification_format.name == FileFormat.COCO.name:
        return is_coco_format(file_path)

    if verification_format.name == FileFormat.CVAT.name:
        return is_cvat_format(file_path)

    if verification_format.name == FileFormat.YOLO.name:
        return is_yolo_format(file_path)

    if verification_format.name == FileFormat.OPENIMAGE.name:
        return is_openimage_format(file_path)

    if verification_format.name == FileFormat.IMAGENET.name:
        return is_imagenet_format(file_path)

    return False


def get_format(file_path):
    """ Tries to anticipate the format of an annotation file.

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    enum (FileFormat)
        Enumerator representing the format.
    """
    # Given a file path with annotations, get the format of the annotations
    if os.path.isfile(file_path) is False:
        return FileFormat.UNKNOWN

    # PASCAL
    if is_pascal_format(file_path):
        return FileFormat.PASCAL

    # Text file
    if is_absolute_text_format(file_path):
        return FileFormat.ABSOLUTE_TEXT

    # Labelme format
    if is_labelme_format(file_path):
        return FileFormat.LABEL_ME

    # COCO format
    if is_coco_format(file_path):
        return FileFormat.COCO

    # CVAT format
    if is_cvat_format(file_path):
        return FileFormat.CVAT

    # YOLO format
    if is_yolo_format(file_path):
        return FileFormat.YOLO

    return FileFormat.UNKNOWN
