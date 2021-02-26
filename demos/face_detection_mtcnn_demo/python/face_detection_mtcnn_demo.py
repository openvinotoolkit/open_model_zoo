import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
import tools_matrix as tools
from openvino.inference_engine import IECore


score_threshold = [0.6, 0.7, 0.7]
iou_threshold = [0.5, 0.7, 0.7, 0.7]

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-i", "--input",
                      help="Required. Path to a test image file.",
                      required=True, type=str)
    args.add_argument("-m_p", "--model_pnet",
                      help="Required. Path to an .xml file with a pnet model.",
                      required=True, type=str, metavar='"<path>"')
    args.add_argument("-m_r", "--model_rnet",
                      help="Required. Path to an .xml file with a rnet model.",
                      required=True, type=str, metavar='"<path>"')
    args.add_argument("-m_o", "--model_onet",
                      help="Required. Path to an .xml file with a onet model.",
                      required=True, type=str, metavar='"<path>"')
    args.add_argument("-th", "--threshold",
                      help="Optional. The threshold to define the face is recognized or not.",
                      type=float, default=0.6, metavar='"<num>"')
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str, metavar='"<device>"')
    args.add_argument("--no_show",
                      help="Optional. Don't show output",
                      action='store_true')

    return parser


def image_reader(image, w, h):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (w, h))
    # Change input shape to [B,C,W,H] for MTCNN
    image = image.transpose((2, 1, 0))
    image = np.expand_dims(image, axis=0)
    return image

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    PNET_model_xml = args.model_pnet
    PNET_model_bin = os.path.splitext(PNET_model_xml)[0] + ".bin"
    RNET_model_xml = args.model_rnet
    RNET_model_bin = os.path.splitext(RNET_model_xml)[0] + ".bin"
    ONET_model_xml = args.model_onet
    ONET_model_bin = os.path.splitext(ONET_model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")

    ie_p = IECore()
    ie_r = IECore()
    ie_o = IECore()

    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(PNET_model_xml, PNET_model_bin))
    p_net = ie_p.read_network(model=PNET_model_xml, weights=PNET_model_bin)
    assert len(p_net.input_info.keys()) == 1, "Pnet supports only single input topologies"
    assert len(p_net.outputs) == 2, "Pnet supports two output topologies"

    log.info("Loading network files:\n\t{}\n\t{}".format(RNET_model_xml, RNET_model_bin))
    r_net = ie_r.read_network(model=RNET_model_xml, weights=RNET_model_bin)
    assert len(r_net.input_info.keys()) == 1, "Rnet supports only single input topologies"
    assert len(r_net.outputs) == 2, "Rnet supports two output topologies"

    log.info("Loading network files:\n\t{}\n\t{}".format(ONET_model_xml, ONET_model_bin))
    o_net = ie_o.read_network(model=ONET_model_xml, weights=ONET_model_bin)
    assert len(o_net.input_info.keys()) == 1, "Onet supports only single input topologies"
    assert len(o_net.outputs) == 3, "Onet supports three output topologies"

    log.info("Preparing pnet input blobs")
    # input of mtcnn
    pnet_input_blob = next(iter(p_net.input_info))
    rnet_input_blob = next(iter(r_net.input_info))
    onet_input_blob = next(iter(o_net.input_info))

    # Read image
    origin_image = cv2.imread(args.input)
    oh, ow, _ = origin_image.shape

    scales = tools.calculateScales(origin_image)

    # *************************************
    # Start Pnet
    # *************************************
    # Loading Pnet model to the plugin
    log.info("Loading Pnet model to the plugin")

    t0 = cv2.getTickCount()
    pnet_res = []
    for scale in scales:
        hs = int(oh*scale)
        ws = int(ow*scale)
        image = image_reader(origin_image, ws, hs)

        p_net.reshape({pnet_input_blob : [1, 3, ws, hs]})  # Change weidth and height of input blob
        exec_pnet = ie_p.load_network(network=p_net, device_name=args.device)

        p_res = exec_pnet.infer(inputs={pnet_input_blob: image})
        pnet_res.append(p_res)

    image_num = len(scales)
    rectangles = []
    for i in range(image_num):
        (layer_name_roi, roi), (layer_name_cls, cls_prob) = pnet_res[i].items()
        _, _, out_h, out_w = cls_prob.shape
        out_side = max(out_h, out_w)
        rectangle = tools.detect_face_12net(cls_prob[0][1], roi[0], out_side, 1/scales[i], ow, oh, score_threshold[0], iou_threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles, iou_threshold[1], 'iou')

    # Start Rnet
    if len(rectangles) > 0:
        # Loading Rnet model to the plugin
        log.info("Loading Rnet model to the plugin")

        r_net.reshape({rnet_input_blob : [len(rectangles), 3, 24, 24]})  # Change batch size of input blob
        exec_rnet = ie_r.load_network(network = r_net, device_name = args.device)

        rnet_input = []
        for rectangle in rectangles:
            crop_img = origin_image[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = image_reader(crop_img, 24, 24)
            rnet_input.extend(crop_img)

        rnet_res = exec_rnet.infer(inputs={rnet_input_blob: rnet_input})

        (layer_name_roi, roi_prob), (layer_name_cls, cls_prob)  = rnet_res.items()
        rectangles = tools.filter_face_24net(cls_prob, roi_prob, rectangles, ow, oh, score_threshold[1], iou_threshold[2])

    # Start Onet
    if len(rectangles) > 0:
        log.info("Loading Onet model to the plugin")

        o_net.reshape({onet_input_blob : [len(rectangles), 3, 48, 48]})  # Change batch size of input blob
        exec_onet = ie_o.load_network(network = o_net, device_name = args.device)

        onet_input = []
        for rectangle in rectangles:
            crop_img = origin_image[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = image_reader(crop_img, 48, 48)
            onet_input.extend(crop_img)

        onet_res = exec_onet.infer(inputs={onet_input_blob: onet_input})

        (layer_name_roi, roi_prob), (layer_name_landmark, pts_prob), (layer_name_cls, cls_prob)  = onet_res.items()
        rectangles = tools.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, ow, oh, score_threshold[2], iou_threshold[3])

    for rectangle in rectangles:
        # Draw detected boxes
        cv2.putText(origin_image, 'confidence: {:.2f}'.format(rectangle[4]), (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
        cv2.rectangle(origin_image, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)
        for i in range(5, 15, 2):
            cv2.circle(origin_image, (int(rectangle[i+0]), int(rectangle[i+1])), 2, (0, 255 , 0))

    infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()  # Record infer time
    cv2.putText(origin_image, 'summary: {:.1f} FPS'.format(
        1.0 / infer_time), (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))

    if not args.no_show:
        # Show resulting image.
        cv2.imshow('test', origin_image)
        cv2.waitKey()


if __name__ == '__main__':
    sys.exit(main() or 0)
