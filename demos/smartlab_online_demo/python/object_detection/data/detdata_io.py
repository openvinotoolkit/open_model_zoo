import json
import pdb
import random
import os
import cv2
import shutil
from joblib import Parallel, delayed
import multiprocessing


def load_label(f:str, format:str):
  # Yang, Wei @ Intel SMG
  if not os.path.exists(f):
      raise ValueError("file not found")

  if format == 'coco':
    df = None
    with open(f, 'r') as h:
      df = json.load(h)
    return df
  elif format == 'yolo':#f:labels dir
    pass
  else:
    print('[warning by tools.data.load_label] unsupported format ' + format)
    return None


def save_label(f:str, data, format:str):
  # Yang, Wei @ Intel SMG
  if format == 'coco':
    with open(f, 'w') as h:
      json.dump(data, h)
    return True
  elif format == 'list':
    assert isinstance(data, list)
    with open(f, 'w') as h:
      for v in data:
        h.write(str(v))
        h.write('\n')
    return True
  elif format == 'dict':
    assert isinstance(data, dict)
    with open(f, 'w') as h:
      for k, v in data.items():
        h.write(str(k))
        h.write('\t')
        h.write(str(v))
        h.write('\n')
    return True
  else:
    print('[warning by tools.data.save_label] unsupported format ' + format)
    return None


def inspect_label(f:str, format:str):
  # Yang, Wei @ Intel SMG
  df = load_label(f, format)
  print('[info by tools.data.inspect_label] plz inspect: df')
  pdb.set_trace()


def cat_labels(f_inputs:list, f_output:str, format:str):
  # Yang, Wei @ Intel SMG
  '''
    concatenate annotation files from different people
    f_inputs: list of src annotation file paths
    f_output: dst annotation file path
    format: supported format only includes ['coco'].
  '''
  if format == 'coco':
    assert len(f_inputs) > 1
    res = {}
    offset_imgid = 0 # offset of image id in current subset
    offset_annid = 0 # offset of annotation id in current subset
    res['images'] = []
    res['annotations'] = []
    for f in f_inputs: # enumerate every subset
      if f == f_output: # omit & overwrite previous output
        continue
      print('[info by tools.data.cat_labels] concatenate file ' + f)
      df = load_label(f, format)
      if not 'licenses' in res.keys(): # direct copy of shared info
        res['licenses'] = df['licenses']
        res['info'] = df['info']
        res['categories'] = df['categories']
      # shift all image ids in df['images']
      for img_dict in df['images']:
        img_dict['id'] += offset_imgid
        res['images'].append(img_dict)
      # shift all image ids in df['annotations']
      for ann_dict in df['annotations']:
        ann_dict['image_id'] += offset_imgid
        ann_dict['id'] += offset_annid
        res['annotations'].append(ann_dict)
      # update offsets for next subset
      offset_imgid += len(df['images'])
      offset_annid += len(df['annotations'])
    # dump to output json
    return save_label(f_output, res, format)
  else:
    print('[warning by tools.data.cat_labels] unsupported format ' + format)
    return False


def get_coco_labels_subset(src_labels:list, img_ids:list):
  # Yang, Wei @ Intel SMG
  shared_keys = ['licenses', 'info', 'categories']
  select_labels = {k:src_labels[k] for k in shared_keys} # copy shared information
  select_labels['images'] = []
  # record selected images
  image_ids = []
  for img_i in img_ids:
    image_ids.append(img_i+1)
    select_labels['images'].append(src_labels['images'][img_i])
  # export selected labels
  select_labels['annotations'] = []
  for ann in src_labels['annotations']:
    if ann['image_id'] in image_ids: # if corresponding image is selected
      select_labels['annotations'].append(ann)
  return select_labels


def export_data_subset(img_root, f_label:str, label_format:str, output_path:str, n_sample:int, postfix:str='JPG', seed:int=1):
  # Yang, Wei @ Intel SMG
  # load annotations
  labels = load_label(f_label, label_format)
  assert labels is not None
  
  # check image number under img_root
  n_imgs = 0
  is_virtual = False
  if img_root is not None: # real image sampling
    for f in os.listdir(img_root):
      if not f.endswith(postfix):
        continue
      else: # only record indicated image files
        n_imgs += 1
    assert n_imgs > 0
    assert n_sample <= n_imgs and 0 < n_sample
    print('[info by tools.data.export_data_subset] total img # = %d' % len(n_imgs))
  else: # temporarily failed to provide images
    n_imgs = len(labels['images'])
    is_virtual = True # virtual image sampling

  # enumerate selected dataset
  random.seed(seed)
  select_fimgs = []
  select_fpimgs = []
  imageid2newid = {} # map original 1-based image_id to reordered index, useful in recovery of selected labels
  shared_keys = ['licenses', 'info', 'categories']
  select_labels = {k:labels[k] for k in shared_keys} # copy shared information
  select_labels['images'] = []
  for sample_i in range(n_sample):
    # record selected images
    img_i = random.randrange(0, n_imgs) # 0-based image index
    select_img = labels['images'][img_i]
    imageid2newid[select_img['id']] = sample_i # map original 1-based image_id to reordered 0-based index
    select_img['id'] = sample_i + 1 # reorder according to sampling
    if 'frame' not in select_img['file_name']:
      pre, mid, post = select_img['file_name'].split('_')
      select_img['file_name'] = '%s_%s_frame_%s' % (pre, mid, post)
    select_labels['images'].append(select_img)
    select_fname = select_img['file_name']
    if is_virtual: # record image file name
      select_fimgs.append(select_fname)
    else: # record image file path
      select_fpimgs.append(os.path.join(img_root, select_fname))

  # export selected images
  print('[info by tools.data.export_data_subset] saving selected image list or images')
  if is_virtual: # only save selected image filenames
    f_select = os.path.join(output_path, 'select_imgs.txt')
    with open(f_select, 'w') as h:
      #[h.write(f+'\n') for f in select_fimgs]
      save_label(f_select, select_fimgs, format='list')
    f_map = os.path.join(output_path, 'select_img_map.txt')
    with open(f_map, 'w') as h:
      #[h.write('%d\t%d\n' %(k, imageid2newid[k])) for k in imageid2newid.keys()]
      save_label(f_map, imageid2newid, 'dict')
  else: # copy selected images from source root
    n_cpu = multiprocessing.cpu_count()
    __ = Parallel(n_jobs=n_cpu)(delayed(shutil.copy)(fp, output_path) for fp in select_fpimgs)
  
  # export selected labels
  print('[info by tools.data.export_data_subset] saving selected labels.')
  select_labels['annotations'] = []
  for ann in labels['annotations']:
    if ann['image_id'] in imageid2newid.keys(): # if corresponding image is selected
      ann['image_id'] = imageid2newid[ann['image_id']] + 1
      select_labels['annotations'].append(ann)
  f_select_labels = os.path.join(output_path, 'instances_select.json')
  save_label(f_select_labels, select_labels, label_format)
  #inspect_label(f_select_labels, label_format)


def export_coco_label_subsets(f_src_label:str, f_dst_labels:list, ratios:list):
  assert abs(1-sum(ratios)) < 0.01 # ensure ratio sum is 1
  # load source annotations
  with open(f_src_label, 'r') as h:
    src_labels = json.load(h)
  n_imgs = len(src_labels['images'])
  # reshuffle dataset
  assert len(f_dst_labels) == len(ratios) # ensure 1 ratio corresponsds to 1 subset
  img_ids = list(range(n_imgs))
  random.shuffle(img_ids)
  # divide dataset
  n_groups = len(ratios)
  ids_groups = [img_ids[ int(n_imgs*sum(ratios[:i])) : int(n_imgs*sum(ratios[:i+1])) ] for i in range(n_groups)]
  # enumerate every subset
  for group_i, subset_img_ids in enumerate(ids_groups):
    subset_labels = get_coco_labels_subset(src_labels, subset_img_ids)
    fp_label = f_dst_labels[group_i]
    print('[info by tools.data_io.export_coco_label_subsets] saving %s subset labels.' % fp_label)
    save_label(fp_label, subset_labels, format='coco')


def cvt_coco2yolo(f_input:str, out_root:str):
    # original source: https://github.com/alexmihalyk23/COCO2YOLO 
    # Copyright (c) 2020 Alexey Mikhailyuk
    root_path = os.path.dirname(os.path.dirname(f_input))
    imgs_path = os.path.join(root_path,'images')
    dir_imgs_path = os.path.join(out_root, 'images')
    out_label_path = os.path.join(out_root, 'labels')
    if not os.path.exists(f_input):
      raise ValueError("coco json file not found, please ensure --src_prefix is input")
    if not os.path.exists(imgs_path):
      raise ValueError("images folder not found")    
    if not os.path.exists(out_root):
      os.makedirs(out_root)
    if not os.path.exists(dir_imgs_path):
      os.makedirs(dir_imgs_path)
    if not os.path.exists(out_label_path):
      os.makedirs(out_label_path) 
    df = load_label(f_input, 'coco')
 
    print("loading coco_image info")
    images_info = {}
    count = 0 # ensure no repeated images ID
    for image in df['images']:
        count += 1
        img_path = os.path.join(imgs_path,image['file_name'])
        if os.path.exists(img_path):
          shutil.copyfile(img_path,os.path.join(dir_imgs_path,image['file_name']))
        images_info[image['id']] = (image['file_name'], image['width'], image['height'])
    assert count==len(images_info.keys()),'repeated images ID'
    print("loading done, total images", len(images_info)) 
    
    print("start converting from coco to yolo")
    anno_dict = dict()
    for anno in df['annotations']:
        bbox = anno['bbox']
        image_id = anno['image_id']
        category_id = anno['category_id']
        image_info = images_info.get(image_id)    
        image_name = image_info[0]
        img_w = image_info[1]
        img_h = image_info[2]
        #bbox2yolo    
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = bbox[0] + w / 2
        centery = bbox[1] + h / 2
        dw = 1 / img_w
        dh = 1 / img_h
        centerx *= dw
        w *= dw
        centery *= dh
        h *= dh
        yolo_box = tuple([centerx, centery, w, h])
        anno_info = (image_name, category_id, yolo_box)
        anno_infos = anno_dict.get(image_id)
        if not anno_infos:
            anno_dict[image_id] = [anno_info]
        else:
            anno_infos.append(anno_info)
            anno_dict[image_id] = anno_infos
    print("converting done, total labels", len(anno_dict))

    print("saving yolo txt file")
    coco_id_name_map = {}
    for cls in df['categories']:
      coco_id_name_map[cls['id']] = cls['name']
    coco_name_list = list(coco_id_name_map.values())
    with open(os.path.join(out_root,"classes.txt"), 'w', encoding='utf-8') as f:
      for line in coco_id_name_map.values():
        f.write(line + '\n')
    for k, v in anno_dict.items():
      file_name = v[0][0].split(".")[0] + ".txt"
      with open(os.path.join(out_label_path,file_name), 'w', encoding='utf-8') as f:
          # print(k, v)
          for obj in v:
              cat_name = coco_id_name_map.get(obj[1])
              category_id = coco_name_list.index(cat_name)
              box = ['{:.6f}'.format(x) for x in obj[2]]
              box = ' '.join(box)
              line = str(category_id) + ' ' + box
              f.write(line + '\n')
    
    print("coco2yolo convert done")


def cvt_yolo2coco(labels_path:str, imgs_path:str, classes_path:str, out_root:str, out_json_prefix = "instance"):

  out_label_path = os.path.join(out_root, 'annotations') 
  out_images_path = os.path.join(out_root,'images')
  if not os.path.exists(out_root):
    os.mkdir(out_root)
  if not os.path.exists(out_label_path):
    os.mkdir(out_label_path)
  if not os.path.exists(out_images_path):
    os.mkdir(out_images_path)
  if not os.path.exists(labels_path):
    raise ValueError("labels folder not found")
  if not os.path.exists(imgs_path):
    raise ValueError("images folder not found")  
  if not os.path.exists(classes_path):
    raise ValueError("classes.txt not found")     
  with open(classes_path) as f:
      classes = f.read().strip().split()

  # images dir name
  # indexes = os.listdir(imgs_path)
  indexes = os.listdir(labels_path)
  dataset = {'info':[],'license':[],'categories': [], 'annotations': [], 'images': []}
  for i, cls in enumerate(classes, 0):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
  print("loading yolo info")
  print("start converting from yolo to coco")
  ann_id_cnt = 0
  suffix = ['jpg','jpeg','png','PNG','JPG']  
  for k, index in enumerate(indexes):
    # img = index.replace(txt,)  
    img_names = [index.replace('txt',m) for m in suffix]
    img_name = None
    for nm in img_names:
      if not os.path.exists(os.path.join(imgs_path, nm)): continue 
      else: 
        img_name = nm 
        break
    # No img follow label's name 
    if img_name == None: continue 

    im = cv2.imread(os.path.join(imgs_path, img_name))
    height, width, _ = im.shape
    dataset['images'].append({'file_name': img_name,
                                'id': k,
                                'width': width,
                                'height': height})

    ### if labeled, copy img
    img_path = os.path.join(out_images_path,img_name)
    cv2.imwrite(img_path, im)
    with open(os.path.join(labels_path, index), 'r') as fr:
        labelList = fr.readlines()
        for label in labelList:
            label = label.strip().split()
            x = float(label[1])
            y = float(label[2])
            w = float(label[3])
            h = float(label[4])

            # yolobox2coco
            H, W, _ = im.shape
            x1 = (x - w / 2) * W
            y1 = (y - h / 2) * H
            x2 = (x + w / 2) * W
            y2 = (y + h / 2) * H

            cls_id = int(label[0])   
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': cls_id,
                'id': ann_id_cnt,
                'image_id': k,
                'iscrowd': 0,
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })
            ann_id_cnt += 1  
  print('saving coco json file')     
  json_name = os.path.join(out_label_path,out_json_prefix+".json")
  with open(json_name, 'w') as f:
      json.dump(dataset, f)
  print('yolo2coco convert done')


if __name__ == '__main__':

  labels_path = r'C:\Users\Martin\Desktop\yolo\labels'
  classes_path = r'C:\Users\Martin\Desktop\yolo\classes.txt'
  imgs_path = r'C:\Users\Martin\Desktop\yolo\images'
  out_root = os.path.join(r'C:\Users\Martin\Desktop\coco')
  # f_input = os.path.join(root_path,'yolo_from_coco')
  f_output = os.path.join(r'C:\Users\Martin\Desktop' )
  cvt_yolo2coco(labels_path,imgs_path,classes_path,out_root)

"""
def cvt_coco2voc(f_input:str, f_output:str):
  # In processing(add images copy)
  # Qi, Sun @ Intel SMG
  root_path = f_input
  label_path = os.path.join(root_path,'annotations')
  imgs_path = os.path.join(root_path,'images')
  df = load_label(f_input, 'coco')
  if not os.path.exists(f_output):
    os.makedirs(f_output)
  def base_dict(filename, width, height, depth=3):# target voc xml format
    return {
        "annotation": {
            "filename": os.path.split(filename)[-1],
            # "folder": "VOCCOCO", "segmented": "0", "owner": {"name": "unknown"},
            # "source": {'database': "The COCO 2017 database", 'annotation': "COCO 2017", "image": "unknown"},
            "size": {'width': width, 'height': height, "depth": depth},
            "object": []
        }
    }
  def base_object(size_info, name, bbox):
    BBOX_OFFSET = 0
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    width = size_info['width']
    height = size_info['height']

    x1 = max(x1, 0) + BBOX_OFFSET
    y1 = max(y1, 0) + BBOX_OFFSET
    x2 = min(x2, width - 1) + BBOX_OFFSET
    y2 = min(y2, height - 1) + BBOX_OFFSET

    return {
        'name': name, 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0',
        'bndbox': {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
    }
  
  print("loading coco_image info")
  images_info = {}
  count = 0 # ensure no repeated images_info ID
  for im in df['images']:
    count += 1
    # if file_name.find('\\') > -1:
    #     file_name = file_name[file_name.index('\\')+1:]
    img = base_dict(im['file_name'], im['width'], im['height'])
    images_info[im["id"]] = img
  assert count==len(images_info.keys()),'repeated images_info ID'
  print("loading done, total images", len(images_info)) 

  print("start converting from coco to voc")
  cate = {x['id']: x['name'] for x in df['categories']}
  for an in df["annotations"]:
    ann = base_object(images_info[an['image_id']]['annotation']["size"], cate[an['category_id']], an['bbox'])
    images_info[an['image_id']]['annotation']['object'].append(ann)
  print("converting done, total labels", len(images_info))

  print("saving voc xml file")
  from xmltodict import unparse
  for k, im in images_info.items():
    im['annotation']['object'] = im['annotation']['object'] or [None]
    unparse(im,
            open(os.path.join(f_output, "{}.xml".format(str(k).zfill(12))), "w"),
            full_document=False, pretty=True)
  print("coco2voc convert done")

"""

