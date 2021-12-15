import argparse
import os
import time
from loguru import logger
import cv2
import torch
from torch import nn
from data.data_augment import ValTransform
from data.detdata_io import save_label
from settings import MwGlobalExp
from tools.model_inspect import fuse_model, get_model_info
from tools.geometry import postprocess
from data.vis import vis
start_file = -2801
max_files = 300

def make_parser():
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  parser = argparse.ArgumentParser("YOLOX Demo!")
  parser.add_argument(
    "demo", default="images", help="demo type, eg. image, video and webcam"
  )
  parser.add_argument("-expn", "--experiment-name", type=str, default=None)
  parser.add_argument("-n", "--name", type=str, default=None, help="model name")
  parser.add_argument(
    "--path", 
    default=r"C:\Users\wyang2\datasets\mythware\v1\images", 
    help="path to images or video"
  )
  parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
  parser.add_argument(
    "--save_result",
    action="store_true",
    help="whether to save the inference result of image/video",
  )
  # exp file
  parser.add_argument(
    "-f",
    "--exp_file",
    default='exps/demo/mw_yolox_n',
    type=str,
    help="pls input your experiment description file",
  )
  parser.add_argument("-c", "--ckpt", 
    default=r'C:\Users\wyang2\local\weights\mw-a513\mw-A513v1-glb1cls10_yolox-n.pth',
    type=str, help="ckpt for eval")
  parser.add_argument(
    "--device",
    default="cpu",
    type=str,
    help="device to run our model, can either be cpu or gpu",
  )
  parser.add_argument("--conf", default=0.1, type=float, help="test conf")
  parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
  parser.add_argument("--tsize", default=None, type=int, help="test img size")
  parser.add_argument(
    "--fp16",
    dest="fp16",
    default=False,
    action="store_true",
    help="Adopting mix precision evaluating.",
  )
  parser.add_argument(
    "--legacy",
    dest="legacy",
    default=False,
    action="store_true",
    help="To be compatible with older versions",
  )
  parser.add_argument(
    "--fuse",
    dest="fuse",
    default=False,
    action="store_true",
    help="Fuse conv and bn for testing.",
  )
  parser.add_argument(
    "--trt",
    dest="trt",
    default=False,
    action="store_true",
    help="Using TensorRT model for testing.",
  )
  return parser


def get_image_list(path):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  image_names = []
  IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
  for maindir, subdir, file_name_list in os.walk(path):
    for filename in file_name_list:
      apath = os.path.join(maindir, filename)
      ext = os.path.splitext(apath)[1]
      if ext in IMAGE_EXT:
        image_names.append(apath)
  return image_names


def get_images(root_path):
  fimgs = []
  for f in os.listdir(root_path):
    if f.endswith('.PNG'):
      fimgs.append(os.path.join(root_path, f))
  return fimgs


class Predictor(object):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  def __init__(
    self,
    model:nn.Module,
    exp,
    cls_names,
    trt_file=None,
    decoder=None,
    device="cpu",
    fp16=False,
    legacy=False,
  ):
    self.model = model
    self.cls_names = cls_names
    self.decoder = decoder
    self.num_classes = exp.num_classes
    self.confthre = exp.confthre
    self.nmsthre = exp.nmsthre
    self.test_size = exp.test_size
    self.device = device
    self.fp16 = fp16
    self.preproc = ValTransform(legacy=legacy)

  def inference(self, img):
    img_info = {"id": 0}
    if isinstance(img, str):
      img_info["file_name"] = os.path.basename(img)
      img = cv2.imread(img)
    else:
      img_info["file_name"] = None
    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img
    ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
    img_info["ratio"] = ratio
    img, _ = self.preproc(img, None, self.test_size)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    if self.device == "gpu":
      img = img.cuda()
      if self.fp16:
        img = img.half()  # to FP16
    with torch.no_grad():
      t0 = time.time()
      outputs = self.model(img)
      #assert isinstance(outputs, torch.Tensor)
      if self.decoder is not None:
        outputs = self.decoder(outputs, dtype=outputs.type())
      outputs = postprocess(
        outputs, self.num_classes, self.confthre,
        self.nmsthre, class_agnostic=True
      )
      logger.info("Infer time: {:.4f}s".format(time.time() - t0))
    return outputs, img_info

  def visual(self, output, img_info, cls_conf=0.35):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
      return img
    if isinstance(output, list):
      if output[0] is None:
        return img
      else:
        output = torch.cat(output)
    output = output.to('cpu')
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
    return vis_res

  def pseudolabel(self, output, img_info, idx_offset, cls_conf=0.35):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    image_id = img_info['id']
    if output is None:
      return img
    output = output.cpu()
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio # [[x0,y0,x1,y1], ...]
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    # enumerate all bbox
    i=0
    res = []
    for box, c, s in zip(bboxes, cls, scores):
      if s < cls_conf:
        continue
      x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
      w, h = x1-x0, y1-y0
      idx = idx_offset + i
      i += 1
      cat_id = int(c)
      res.append({
        'area': w * h, 
        'bbox': [x0, y0, w, h],
        'category_id': cat_id,
        'id': idx,
        'image_id': image_id,
        'iscrowd': 0,
        'segmentation': [[x0, y0, x1, y1]]
      })
    return res


def image_demo(predictor:Predictor, vis_folder, path, current_time, save_result):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  if os.path.isdir(path):
    files = get_image_list(path)
  else:
    files = [path]
  files.sort()
  annotations = []
  images = []
  for image_id, image_name in enumerate(files):
    outputs, img_info = predictor.inference(image_name)
    img_info['id'] = image_id
    #import pdb
    #pdb.set_trace()
    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    sub_annotations = predictor.pseudolabel(outputs[0], img_info, len(annotations))
    annotations.extend(sub_annotations)
    images.append({
      'file_name': img_info['file_name'],
      "id": img_info['id'],
      'width': img_info['width'],
      'height': img_info['height']
    })
    if save_result:
      save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
      )
      os.makedirs(save_folder, exist_ok=True)
      save_file_name = os.path.join(save_folder, os.path.basename(image_name))
      logger.info("Saving detection result in {}".format(save_file_name))
      cv2.imwrite(save_file_name, result_image)
    ch = cv2.waitKey(0)
    if ch == 27 or ch == ord("q") or ch == ord("Q"):
      break
  ann_dict = {
    'license':'',
    'info': {'license':1},
    'annotations': annotations,
    'images': images
  }
  cats = []
  for i, c in enumerate(predictor.cls_names):
    cats.append({'id':i, 'name':c, 'supercategory':'mark'})
  ann_dict['categories']=cats
  fp_outann = os.path.join(vis_folder, 'instances_coco.json')
  save_label(fp_outann, ann_dict, 'coco')


def images_demo(predictor:Predictor, vis_folder, path, current_time, save_result, cls_conf=0.35, is_show=True):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  if os.path.isdir(path):
    files = get_images(path)
    if len(files) > max_files:
      files = files[start_file:start_file+max_files]
  else:
    files = [path]
  files.sort()
  images = [cv2.imread(f) for f in files]
  diffs = []
  win = 'preview'
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  import pdb
  for img, fimg in zip(images, files):
    t1 = time.time()
    outputs, img_info = predictor.inference(img)
    pdb.set_trace()
    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    t2 = time.time()
    diffs.append(t2 - t1)
    if is_show:
      img_label = predictor.visual(outputs, img_info, cls_conf)
      cv2.imshow(win, img_label)
      ch = cv2.waitKey(1)
      if ch == 27 or ch == ord("q") or ch == ord("Q"):
        break
    elif save_result:
      save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
      )
      os.makedirs(save_folder, exist_ok=True)
      save_file_name = os.path.join(save_folder, os.path.basename(fimg))
      logger.info("Saving detection result in {}".format(save_file_name))
      cv2.imwrite(save_file_name, result_image)
  with open('timings.txt', 'w') as h:
    [h.write('%.2f\n' % v) for v in diffs]
  cv2.destroyAllWindows()


def imageflow_demo(predictor, vis_folder, current_time, args):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
  fps = cap.get(cv2.CAP_PROP_FPS)
  save_folder = os.path.join(
    vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
  )
  os.makedirs(save_folder, exist_ok=True)
  if args.demo == "video":
    save_path = os.path.join(save_folder, args.path.split("/")[-1])
  else:
    save_path = os.path.join(save_folder, "camera.mp4")
  logger.info(f"video save_path is {save_path}")
  vid_writer = cv2.VideoWriter(
    save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
  )
  while True:
    ret_val, frame = cap.read()
    if ret_val:
      outputs, img_info = predictor.inference(frame)
      result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
      if args.save_result:
        vid_writer.write(result_frame)
      ch = cv2.waitKey(1)
      if ch == 27 or ch == ord("q") or ch == ord("Q"):
        break
    else:
      break


def main(exp:MwGlobalExp, args):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  if not args.experiment_name:
    args.experiment_name = exp.exp_name
  file_name = os.path.join(exp.output_dir, args.experiment_name)
  os.makedirs(file_name, exist_ok=True)
  vis_folder = None
  if args.save_result:
    vis_folder = os.path.join(file_name, "vis_res")
    os.makedirs(vis_folder, exist_ok=True)
  if args.trt:
    args.device = "gpu"
  logger.info("Args: {}".format(args))
  if args.conf is not None:
    exp.test_conf = args.conf
  if args.tsize is not None:
    exp.test_size = (args.tsize, args.tsize)
  model = exp.get_model()
  assert isinstance(model, nn.Module) 
  logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
  if args.device == "gpu":
    model.cuda()
    if args.fp16:
      model.half()  # to FP16
  model.eval()
  if not args.trt:
    if args.ckpt is None:
      ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
      ckpt_file = exp.fp_model
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
  if args.fuse:
    logger.info("\tFusing model...")
    model = fuse_model(model)
  if args.trt:
    assert not args.fuse, "TensorRT model is not support model fusing!"
    trt_file = os.path.join(file_name, "model_trt.pth")
    assert os.path.exists(
      trt_file
    ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
    model.head.decode_in_inference = False
    decoder = model.head.decode_outputs
    logger.info("Using TensorRT to inference")
  else:
    trt_file = None
    decoder = None
  predictor = Predictor(model, exp, exp.mw_classes, trt_file, decoder, args.device, args.fp16, args.legacy)
  current_time = time.localtime()
  if args.demo == "image":
    image_demo(predictor, vis_folder, exp.root_imgs, current_time, args.save_result)
  elif args.demo == 'images':
    images_demo(predictor, vis_folder, exp.root_imgs, current_time, args.save_result)
  elif args.demo == "video" or args.demo == "webcam":
    imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
  args = make_parser().parse_args()
  exp = MwGlobalExp(
    num_classes = 10,
    fp_model = r'C:\Users\wyang2\local\weights\mw-a513\mw-A513v1-glb1cls10_yolox-n.pth',
    root_input = r'C:\Users\wyang2\datasets\mythware\v1\images',
    conf_thresh= 0.1,
    nms_thresh = 0.3,
    is_show = True)
  main(exp, args)
