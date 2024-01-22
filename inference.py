import copy
import glob
import random
import time
import argparse
import numpy as np
import torch.utils.data
from network import *
from util import *
from drawBiaoZhun import biaoding

def demo_diff(args):
    net = Net(Norm=args.Norm_type).to(torch.device(args.device))

    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained) if args.device == 'gpu' else torch.load(args.pretrained,
                                                                                         map_location='cpu')
        net = get_state_static(net, checkpoint)

    net.eval()

    # pre process
    input_name = args.input.split('/')[-1][:-4]

    if args.biaoding:
        biaoding(args)
        biaoding_file = "./DrawRect/biaozhun/labels/" + input_name + ".txt"
        if not os.path.exists(biaoding_file):
            print("该图片未标定！")
            biaoding(args)
        polygon, biaoding_box = get_list0(input_name)
        if len(polygon) == 4:
            x0, y0, x1, y1 = polygon[0], polygon[1], polygon[2], polygon[3]
            polygon = [x0, y0, x1, y0, x1, y1, x0, y1]
    else:
        print('使用整张图进行推理')
        if not args.biaoding:
            img1 = cv2.imread(args.input)
            biaoding_box = [0, 0, img1.shape[1], img1.shape[0]]
            polygon = [0, 0, img1.shape[1], 0, img1.shape[1], img1.shape[0], 0, img1.shape[0]]

    data, hard_mask, input_src, biaoding_box_zoom = get_demo_input_zoom(args, biaoding_box=biaoding_box, polygon=polygon)

    net = net.to(torch.device(args.device))
    data = data.to(torch.device(args.device))

    diff_map = net(data)

    # post process
    diff_map = diff_map.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    temp = data[:, 3:6, :, :].squeeze(0).permute(1, 2, 0).cpu().numpy()
    inp = data[:, 0:3, :, :].squeeze(0).permute(1, 2, 0).cpu().numpy()

    bbox_on_inp = get_bbox_from_map(args, diff_map, input_src, biaoding_box_zoom,
                                                         polygon)  # 在框出来的图片中将缺陷标注出来

    bbox_on_inp = bbox_on_inp * hard_mask

    temp = cv2.normalize(temp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    inp = cv2.normalize(inp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmapshow = None
    heatmapshow = cv2.normalize(diff_map, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    heatmapshow = cv2.addWeighted(heatmapshow, 0.7, inp.astype(np.uint8), 0.3, 0)

    if args.save_result:
        cv2.imwrite('./result/bbox_on_inp_' + input_name + '_result.png', heatmapshow)
        print('result have been saved to: ./result/' + input_name + '_result.png')
    if args.show_result:

        cv2.namedWindow('test_result', cv2.WINDOW_NORMAL)
        cv2.imshow('test_result', np.hstack((temp, inp, heatmapshow)).astype(np.uint8))

        cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', metavar='path',
                        help='path to input image')
    parser.add_argument('temp', metavar='path',
                        help='path to template image')
    parser.add_argument('--biaoding', action='store_true', help='it is necessary for high resolution images')
    parser.add_argument('--Norm_type', default='instance', choices=['batch', 'layer', 'instance', 'none'],
                        help='normalization function')
    parser.add_argument('--device', default='cpu', help='gpu or cpu')
    parser.add_argument('--save_result', action='store_true', help='write the inference result to ./result')
    parser.add_argument('--show_result', default=True, help='show the inference result')
    parser.add_argument('--diff_threshold', default=0.1, type=float, help='threshold to distinguish normal and '
                                                                          'anomaly(float number from 0.0 to 1.0)')
    parser.add_argument('--pretrained', default='./pretrained_9111.pth', metavar='N',
                        help='path to checkpoint file')

    seed_torch(3407)

    args = parser.parse_args()
    demo_diff(args)
