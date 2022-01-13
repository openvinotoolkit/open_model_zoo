import os
from loguru import logger
import cv2
import numpy as np
from pycocotools.coco import COCO
from data.dataloading import get_yolox_datadir
from data.datasets_wrapper import Dataset


class MythwareDataset(Dataset):
  # modified from: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  """
  COCO-style dataset class.
  """
  def __init__(
    self,
    data_dir=None, ### 图像根目录
    json_file="instances_select.json", ### coco风格标注文件
    name="select_front", ### 数据集名称
    img_size=(416, 416), ### 图像前处理的目标尺寸：高 x 宽
    preproc=None, ### 数据增强过程函数
    cache=False, ### 是否整数据集放入RAM加速迭代
  ):
    """
    COCO-style mythware dataset initialization. Annotation data are read into memory by COCO API.
    Args:
      data_dir (str): dataset root directory
      json_file (str): COCO json file name
      name (str): COCO data name (e.g. 'train2017' or 'val2017')
      img_size (int): target image size after pre-processing
      preproc: data augmentation strategy
    """
    super().__init__(img_size)
    ### 定位图片数据集根目录
    if data_dir is None:
      data_dir = os.path.join(get_yolox_datadir(), "select_front")
    self.data_dir = data_dir
    self.json_file = json_file
    ### 构建数据集实例
    self.coco = COCO(os.path.join(self.data_dir, 'annotations', self.json_file)) ### label文件绝对路径
    self.ids = self.coco.getImgIds() ### 获取image_id列表
    self.class_ids = sorted(self.coco.getCatIds()) ### 重排类别_id列表
    cats = self.coco.loadCats(self.coco.getCatIds())
    self._classes = tuple([c["name"] for c in cats]) ### 获取类别名称列表
    self.imgs = None
    self.name = name ### 数据集名称（图片目录名称）
    self.img_size = img_size
    self.preproc = preproc ### 数据增强wrapper函数
    self.annotations = self._load_coco_annotations() ### 标注矩阵：tuple list, Nx4, (bbox+catid，img_size，img_size_norm，fimg)
    if cache:
      self._cache_images()

  def __len__(self):
    return len(self.ids)

  def __del__(self):
    del self.imgs

  def  _load_coco_annotations(self):
    return [self.load_anno_from_ids(_ids) for _ids in self.ids]

  def _cache_images(self):
    logger.warning( ### 简之，想用RAM缓存COCO数据集，需要200GB+的内存；咱们demo数据集应该没有这么大
      "\n********************************************************************************\n"
      "You are using cached images in RAM to accelerate training.\n"
      "This requires large system RAM.\n"
      "Make sure you have large RAM available for training demo mythware dataset.\n"
      "********************************************************************************\n"
    )
    max_h = self.img_size[0]
    max_w = self.img_size[1]
    cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
    if not os.path.exists(cache_file):
      logger.info(
        "Caching images for the first time. This might take minutes."
      )
      self.imgs = np.memmap( ### 写模式，将磁盘大文件映射到内存，以允许分段导入数据
          cache_file,
          shape=(len(self.ids), max_h, max_w, 3),
          dtype=np.uint8,
          mode="w+",
      )
      from tqdm import tqdm
      from multiprocessing.pool import ThreadPool
      NUM_THREADs = min(8, os.cpu_count()) ### 满物理核并行读取
      loaded_images = ThreadPool(NUM_THREADs).imap(
        lambda x: self.load_resized_img(x),
        range(len(self.annotations)),
      )
      pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
      for k, out in pbar: ### 显示复制进度
        self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
      self.imgs.flush()
      pbar.close()
    else:
      logger.warning( ### 复制期间严禁修改磁盘源数据
        "You are using cached imgs! Make sure your dataset is not changed!!"
      )
    logger.info("Loading cached imgs...")
    self.imgs = np.memmap( ### 转为可读模式
      cache_file,
      shape=(len(self.ids), max_h, max_w, 3),
      dtype=np.uint8,
      mode="r+",
    )

  def load_anno_from_ids(self, id_): ### 返回 bbox标注和分类，图片原尺寸，normalized尺寸，文件名
      im_ann = self.coco.loadImgs(id_)[0] ### 读取一个样本图片
      width = im_ann["width"]
      height = im_ann["height"]
      anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False) ### 读取样本的标注id
      annotations = self.coco.loadAnns(anno_ids) ### 读取样本标注
      objs = []
      for obj in annotations: ### 遍历：收集所有样本的bbox组成列表objs
        x1 = np.max((0, obj["bbox"][0])) ### 获取样本的bbox
        y1 = np.max((0, obj["bbox"][1]))
        x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
        y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
        if obj["area"] > 0 and x2 >= x1 and y2 >= y1: ### 样本面积需要为正，长、宽都为非负
          obj["clean_bbox"] = [x1, y1, x2, y2]
          objs.append(obj)
      num_objs = len(objs)
      res = np.zeros((num_objs, 5))
      for ix, obj in enumerate(objs):
        cls = self.class_ids.index(obj["category_id"])
        res[ix, 0:4] = obj["clean_bbox"]
        res[ix, 4] = cls ### 标注矩阵第5维度插入类别id
      r = min(self.img_size[0] / height, self.img_size[1] / width)
      res[:, :4] *= r ### 以短边作为scaler进行normalize
      img_info = (height, width)
      resized_info = (int(height * r), int(width * r)) ### normalize后的尺寸
      file_name = (
        im_ann["file_name"]
        if "file_name" in im_ann
        else "{:012}".format(id_) + ".jpg"
      )
      return (res, img_info, resized_info, file_name)

  def load_anno(self, index):
    return self.annotations[index][0]

  def load_resized_img(self, index): ### 读取指定标注的图像，并变形为统一尺寸
    img = self.load_image(index)
    r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
    resized_img = cv2.resize(
      img,
      (int(img.shape[1] * r), int(img.shape[0] * r)),
      interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    return resized_img

  def load_image(self, index): ### 读取指定标注的图像，backend: OpenCV
    file_name = self.annotations[index][3]
    img_file = os.path.join(self.data_dir, 'images', file_name)
    img = cv2.imread(img_file)
    assert img is not None
    return img

  def pull_item(self, index): ### 读取指定标注的图像和标注，返回 图像，标注+类别id，图尺寸，单标注id列表
    id_ = self.ids[index]
    res, img_info, resized_info, _ = self.annotations[index]
    if self.imgs is not None:
      pad_img = self.imgs[index]
      img = pad_img[: resized_info[0], : resized_info[1], :].copy()
    else:
      img = self.load_resized_img(index)
    return img, res.copy(), img_info, np.array([id_])

  @Dataset.mosaic_getitem ### 输入如果是 non-int, index，则会记录是否进行mosaic数据增强
  def __getitem__(self, index): ### generator迭代函数
    """
    One image / label pair for the given index is picked up and pre-processed.
    Args:
      index (int): data index
    Returns:
      img (numpy.ndarray): pre-processed image
      padded_labels (torch.Tensor): pre-processed label data.
        The shape is :math:`[max_labels, 5]`.
        each label consists of [class, xc, yc, w, h]:
          class (float): class index.
          xc, yc (float) : center of bbox whose values range from 0 to 1.
          w, h (float) : size of bbox whose values range from 0 to 1.
      info_img : tuple of h, w.
        h, w (int): original shape of the image
      img_id (int): same as the input index. Used for evaluation.
    """
    img, target, img_info, img_id = self.pull_item(index)
    if self.preproc is not None:
        img, target = self.preproc(img, target, self.input_dim)
    return img, target, img_info, img_id
