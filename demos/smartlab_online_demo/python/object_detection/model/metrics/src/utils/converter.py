import base64
import json
import os
import xml.etree.ElementTree as ET

import pandas as pd
import src.utils.general_utils as general_utils
import src.utils.validations as validations
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBFormat, BBType, CoordinatesType


def _get_annotation_files(file_path):
    # Path can be a directory containing all files or a directory containing multiple files
    if file_path is None:
        return []
    annotation_files = []
    if os.path.isfile(file_path):
        annotation_files = [file_path]
    elif os.path.isdir(file_path):
        annotation_files = general_utils.get_files_recursively(file_path)
    return sorted(annotation_files)


def coco2bb(path, bb_type=BBType.GROUND_TRUTH):
    ret = []
    # Get annotation files in the path
    annotation_files = _get_annotation_files(path)
    # Loop through each file
    for file_path in annotation_files:
        if not validations.is_coco_format(file_path):
            continue

        with open(file_path, "r") as f:
            json_object = json.load(f)

        # COCO json file contains basically 3 lists:
        # categories: containing the classes
        # images: containing information of the images (width, height and filename)
        # annotations: containing information of the bounding boxes (x1, y1, bb_width, bb_height)
        classes = {}
        if 'categories' in json_object:
            classes = json_object['categories']
            # into dictionary
            classes = {c['id']: c['name'] for c in classes}
        images = {}
        # into dictionary
        for i in json_object['images']:
            images[i['id']] = {
                'file_name': i['file_name'],
                'img_size': (int(i['width']), int(i['height']))
            }
        annotations = []
        if 'annotations' in json_object:
            annotations = json_object['annotations']

        for annotation in annotations:
            img_id = annotation['image_id']
            x1, y1, bb_width, bb_height = annotation['bbox']
            if bb_type == BBType.DETECTED and 'score' not in annotation.keys():
                print('Warning: Confidence not found in the JSON file!')
                return ret
            confidence = annotation['score'] if bb_type == BBType.DETECTED else None
            # Make image name only the filename, without extension
            img_name = images[img_id]['file_name']
            img_name = general_utils.get_file_name_only(img_name)
            # create BoundingBox object
            bb = BoundingBox(image_name=img_name,
                             class_id=classes[annotation['category_id']],
                             coordinates=(x1, y1, bb_width, bb_height),
                             type_coordinates=CoordinatesType.ABSOLUTE,
                             img_size=images[img_id]['img_size'],
                             confidence=confidence,
                             bb_type=bb_type,
                             format=BBFormat.XYWH)
            ret.append(bb)
    return ret


def cvat2bb(path):
    '''This format supports ground-truth only'''
    ret = []
    # Get annotation files in the path
    annotation_files = _get_annotation_files(path)
    # Loop through each file
    for file_path in annotation_files:
        if not validations.is_cvat_format(file_path):
            continue

        # Loop through the images
        for image_info in ET.parse(file_path).iter('image'):
            img_size = (int(image_info.attrib['width']), int(image_info.attrib['height']))
            img_name = image_info.attrib['name']
            img_name = general_utils.get_file_name_only(img_name)

            # Loop through the boxes
            for box_info in image_info.iter('box'):
                x1, y1, x2, y2 = float(box_info.attrib['xtl']), float(
                    box_info.attrib['ytl']), float(box_info.attrib['xbr']), float(
                        box_info.attrib['ybr'])
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                bb = BoundingBox(image_name=img_name,
                                 class_id=box_info.attrib['label'],
                                 coordinates=(x1, y1, x2, y2),
                                 img_size=img_size,
                                 type_coordinates=CoordinatesType.ABSOLUTE,
                                 bb_type=BBType.GROUND_TRUTH,
                                 format=BBFormat.XYX2Y2)
                ret.append(bb)
    return ret


def openimage2bb(annotations_path, images_dir, bb_type=BBType.GROUND_TRUTH):
    ret = []
    # Get annotation files in the path
    annotation_files = _get_annotation_files(annotations_path)
    # Loop through each file
    for file_path in annotation_files:
        if not validations.is_openimage_format(file_path):
            continue
        images_shapes = {}
        # Open csv
        csv = pd.read_csv(file_path, sep=',')
        for i, row in csv.iterrows():
            # Get image resolution if it was not loaded yet
            if row['ImageID'] not in images_shapes:
                img_name = row['ImageID']
                image_file = general_utils.find_file(images_dir, img_name)
                images_shapes[image_file] = general_utils.get_image_resolution(image_file)
            if images_shapes[image_file] is None:
                print(f'Warning: It was not possible to find the resolution of image {img_name}')
                continue
            # Three is no bounding box for the given image
            if pd.isna(row['LabelName']) or pd.isna(row['XMin']) or pd.isna(row['XMax']) or pd.isna(
                    row['YMin']) or pd.isna(row['YMax']):
                continue
                # images_shapes[image_file] = general_utils.get_image_resolution(image_file)
            img_size = (images_shapes[image_file]['width'], images_shapes[image_file]['height'])
            x1, x2, y1, y2 = (row['XMin'], row['XMax'], row['YMin'], row['YMax'])
            x1 = x1.replace(',', '.') if isinstance(x1, str) else x1
            x2 = x2.replace(',', '.') if isinstance(x2, str) else x2
            y1 = y1.replace(',', '.') if isinstance(y1, str) else y1
            y2 = y2.replace(',', '.') if isinstance(y2, str) else y2
            x1, x2, y1, y2 = float(x1), float(x2), float(y1), float(y2)
            confidence = None if pd.isna(row['Confidence']) else float(row['Confidence'])
            if bb_type == BBType.DETECTED and confidence is None:
                print(f'Warning: Confidence value found in the CSV file for the image {img_name}')
                return ret
            bb = BoundingBox(image_name=general_utils.get_file_name_only(row['ImageID']),
                             class_id=row['LabelName'],
                             coordinates=(x1, y1, x2, y2),
                             img_size=img_size,
                             confidence=confidence,
                             type_coordinates=CoordinatesType.RELATIVE,
                             bb_type=bb_type,
                             format=BBFormat.XYX2Y2)
            ret.append(bb)
    return ret


def imagenet2bb(annotations_path):
    ret = []
    # Get annotation files in the path
    annotation_files = _get_annotation_files(annotations_path)
    # Loop through each file
    for file_path in annotation_files:
        if not validations.is_imagenet_format(file_path):
            continue
        # Open XML
        img_name = ET.parse(file_path).find('filename').text
        img_name = general_utils.get_file_name_only(img_name)
        img_width = int(ET.parse(file_path).find('size/width').text)
        img_height = int(ET.parse(file_path).find('size/height').text)
        img_size = (img_width, img_height)
        # Loop through the detections
        for box_info in ET.parse(file_path).iter('object'):
            obj_class = box_info.find('name').text
            x1 = int(float(box_info.find('bndbox/xmin').text))
            y1 = int(float(box_info.find('bndbox/ymin').text))
            x2 = int(float(box_info.find('bndbox/xmax').text))
            y2 = int(float(box_info.find('bndbox/ymax').text))
            bb = BoundingBox(image_name=img_name,
                             class_id=obj_class,
                             coordinates=(x1, y1, x2, y2),
                             img_size=img_size,
                             type_coordinates=CoordinatesType.ABSOLUTE,
                             bb_type=BBType.GROUND_TRUTH,
                             format=BBFormat.XYX2Y2)
            ret.append(bb)
    return ret


def vocpascal2bb(annotations_path):
    return imagenet2bb(annotations_path)


def labelme2bb(annotations_path):
    ret = []
    # Get annotation files in the path
    annotation_files = _get_annotation_files(annotations_path)
    # Loop through each file
    for file_path in annotation_files:
        if not validations.is_labelme_format(file_path):
            continue
        # Parse the JSON file
        with open(file_path, "r") as f:
            json_object = json.load(f)
        img_path = json_object['imagePath']
        img_path = os.path.basename(img_path)
        img_path = general_utils.get_file_name_only(img_path)
        img_size = (int(json_object['imageWidth']), int(json_object['imageHeight']))
        # If there are annotated objects
        if 'shapes' in json_object:
            # Loop through bounding boxes
            for obj in json_object['shapes']:
                obj_label = obj['label']
                ((x1, y1), (x2, y2)) = obj['points']
                # If there is no bounding box annotations, bb coordinates could have been set to None
                if x1 is None and y1 is None and x2 is None and y2 is None:
                    continue
                x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))

                bb = BoundingBox(image_name=img_path,
                                 class_id=obj_label,
                                 coordinates=(x1, y1, x2, y2),
                                 img_size=img_size,
                                 confidence=None,
                                 type_coordinates=CoordinatesType.ABSOLUTE,
                                 bb_type=BBType.GROUND_TRUTH,
                                 format=BBFormat.XYX2Y2)
                ret.append(bb)
    return ret


def text2bb(annotations_path,
            bb_type=BBType.GROUND_TRUTH,
            bb_format=BBFormat.XYWH,
            type_coordinates=CoordinatesType.ABSOLUTE,
            img_dir=None):
    ret = []

    # Get annotation files in the path
    annotation_files = _get_annotation_files(annotations_path)
    for file_path in annotation_files:
        if type_coordinates == CoordinatesType.ABSOLUTE:
            if bb_type == BBType.GROUND_TRUTH and not validations.is_absolute_text_format(
                    file_path, num_blocks=[5], blocks_abs_values=[4]):
                continue
            if bb_type == BBType.DETECTED and not validations.is_absolute_text_format(
                    file_path, num_blocks=[6], blocks_abs_values=[4]):
                continue
        elif type_coordinates == CoordinatesType.RELATIVE:
            if bb_type == BBType.GROUND_TRUTH and not validations.is_relative_text_format(
                    file_path, num_blocks=[5], blocks_rel_values=[4]):
                continue
            if bb_type == BBType.DETECTED and not validations.is_relative_text_format(
                    file_path, num_blocks=[6], blocks_rel_values=[4]):
                continue
        # Loop through lines
        with open(file_path, "r") as f:

            img_filename = os.path.basename(file_path)
            img_filename = os.path.splitext(img_filename)[0]

            img_size = None
            # If coordinates are relative, image size must be obtained in the img_dir
            if type_coordinates == CoordinatesType.RELATIVE:
                img_path = general_utils.find_image_file(img_dir, img_filename)
                if img_path is None or os.path.isfile(img_path) is False:
                    print(
                        f'Warning: Image not found in the directory {img_path}. It is required to get its dimensions'
                    )
                    return ret
                resolution = general_utils.get_image_resolution(img_path)
                img_size = (resolution['width'], resolution['height'])
            for line in f:
                if line.replace(' ', '') == '\n':
                    continue
                splitted_line = line.split(' ')
                class_id = splitted_line[0]
                if bb_type == BBType.GROUND_TRUTH:
                    confidence = None
                    x1 = float(splitted_line[1])
                    y1 = float(splitted_line[2])
                    w = float(splitted_line[3])
                    h = float(splitted_line[4])
                elif bb_type == BBType.DETECTED:
                    confidence = float(splitted_line[1])
                    x1 = float(splitted_line[2])
                    y1 = float(splitted_line[3])
                    w = float(splitted_line[4])
                    h = float(splitted_line[5])
                bb = BoundingBox(image_name=img_filename,
                                 class_id=class_id,
                                 coordinates=(x1, y1, w, h),
                                 img_size=img_size,
                                 confidence=confidence,
                                 type_coordinates=type_coordinates,
                                 bb_type=bb_type,
                                 format=bb_format)
                # If the format is correct, x,y,w,h,x2,y2 must be positive
                x, y, w, h = bb.get_absolute_bounding_box(format=BBFormat.XYWH)
                _, _, x2, y2 = bb.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
                if x < 0 or y < 0 or w < 0 or h < 0 or x2 < 0 or y2 < 0:
                    continue
                ret.append(bb)
    return ret


def yolo2bb(annotations_path, images_dir, file_obj_names, bb_type=BBType.GROUND_TRUTH):
    ret = []
    if not os.path.isfile(file_obj_names):
        print(f'Warning: File with names of classes {file_obj_names} not found.')
        return ret
    # Load classes
    all_classes = []
    with open(file_obj_names, "r") as f:
        all_classes = [line.replace('\n', '') for line in f]
    # Get annotation files in the path
    annotation_files = _get_annotation_files(annotations_path)
    # Loop through each file
    for file_path in annotation_files:
        if not validations.is_yolo_format(file_path, bb_types=[bb_type]):
            continue
        img_name = os.path.basename(file_path)
        img_file = general_utils.find_image_file(images_dir, img_name)
        img_resolution = general_utils.get_image_resolution(img_file)
        if img_resolution is None:
            print(f'Warning: It was not possible to find the resolution of image {img_name}')
            continue
        img_size = (img_resolution['width'], img_resolution['height'])
        # Loop through lines
        with open(file_path, "r") as f:
            for line in f:
                if line.replace(' ', '') == '\n':
                    continue
                splitted_line = line.split(' ')
                class_id = splitted_line[0]
                if not general_utils.is_str_int(class_id):
                    print(
                        f'Warning: Class id represented in the {file_path} is not a valid integer.')
                    return []
                class_id = int(class_id)
                if class_id not in range(len(all_classes)):
                    print(
                        f'Warning: Class id represented in the {file_path} is not in the range of classes specified in the file {file_obj_names}.'
                    )
                    return []
                if bb_type == BBType.GROUND_TRUTH:
                    confidence = None
                    x1 = float(splitted_line[1])
                    y1 = float(splitted_line[2])
                    w = float(splitted_line[3])
                    h = float(splitted_line[4])
                elif bb_type == BBType.DETECTED:
                    confidence = float(splitted_line[1])
                    x1 = float(splitted_line[2])
                    y1 = float(splitted_line[3])
                    w = float(splitted_line[4])
                    h = float(splitted_line[5])
                bb = BoundingBox(image_name=general_utils.get_file_name_only(img_file),
                                 class_id=all_classes[class_id],
                                 coordinates=(x1, y1, w, h),
                                 img_size=img_size,
                                 confidence=confidence,
                                 type_coordinates=CoordinatesType.RELATIVE,
                                 bb_type=bb_type,
                                 format=BBFormat.YOLO)
                ret.append(bb)
    return ret


def xml2csv(xml_path):
    # Adapted from https://stackoverflow.com/questions/63061428/convert-labelimg-xml-rectangles-to-labelme-json-polygons-with-image-data
    xml_list = []
    xml_df = pd.DataFrame()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_width = int(root.find('size')[0].text)
        img_height = int(root.find('size')[1].text)
        filename = root.find('filename').text
        for member in root.findall('object'):
            value = (
                filename,  #ImageID
                '',  #Source
                member[0].text,  #LabelName (class)
                '',  #Confidence
                int(float(member.find('bndbox')[0].text)),  #xmin
                int(float(member.find('bndbox')[2].text)),  #xmax
                int(float(member.find('bndbox')[1].text)),  #ymin
                int(float(member.find('bndbox')[3].text)),  #ymax
                '',  #IsOccluded
                '',  #IsTruncated
                '',  #IsGroupOf
                '',  #IsDepiction
                '',  #IsInside
                img_width,  #width
                img_height,  #height
            )
            xml_list.append(value)
            column_name = [
                'ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax',
                'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside', 'width',
                'height'
            ]
            xml_df = pd.DataFrame(xml_list, columns=column_name)
    except Exception as e:
        return pd.DataFrame(columns=[
            'ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax',
            'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside', 'width', 'height'
        ])
    if xml_df.empty:
        return pd.DataFrame.from_dict({
            'ImageID': [filename],
            'Source': [''],
            'LabelName': [''],
            'Confidence': [''],
            'XMin': [None],
            'XMax': [None],
            'YMin': [None],
            'YMax': [None],
            'IsOccluded': [''],
            'IsTruncated': [''],
            'IsGroupOf': [''],
            'IsDepiction': [''],
            'IsInside': [''],
            'width': [img_width],
            'height': [img_height]
        })

    else:
        return xml_df


def df2labelme(symbolDict, dir_image):
    try:
        symbolDict.rename(columns={
            'LabelName': 'label',
            'ImageID': 'imagePath',
            'height': 'imageHeight',
            'width': 'imageWidth'
        },
                          inplace=True)
        # Get image path
        image_path = general_utils.find_file(dir_image, symbolDict['imagePath'][0])
        assert image_path is not None
        encoded = base64.b64encode(open(image_path, "rb").read())
        symbolDict.loc[:, 'imageData'] = encoded

        # File without annotations
        if 'XMin' in symbolDict.columns and 'YMin' in symbolDict.columns and 'XMax' in symbolDict.columns and 'YMax' in symbolDict.columns:
            symbolDict['min'] = symbolDict[['XMin', 'YMin']].values.tolist()
            symbolDict['max'] = symbolDict[['XMax', 'YMax']].values.tolist()
            symbolDict['points'] = symbolDict[['min', 'max']].values.tolist()
            symbolDict['shape_type'] = 'rectangle'
            symbolDict['group_id'] = None
            symbolDict = symbolDict.groupby(['imagePath', 'imageWidth', 'imageHeight', 'imageData'])
            symbolDict = (
                symbolDict.apply(lambda x: x[['label', 'points', 'shape_type', 'group_id']].to_dict(
                    'records')).reset_index().rename(columns={0: 'shapes'}))
        converted_json = json.loads(symbolDict.to_json(orient='records'))[0]
    except Exception as e:
        converted_json = {}
        print('error in labelme conversion:{}'.format(e))
    return converted_json
