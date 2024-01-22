import copy
import glob
import random
import time

import numpy as np
import torch.utils.data
from tqdm import tqdm
from network import *
from util import *
import argparse

class Deep_PCB(object):
    def __init__(self, path=r'D:\DL_extend\DeepPCB-master\PCBData', is_train=False, args=None):
        super(Deep_PCB, self).__init__()
        self.path = path
        self.args = args
        if is_train:
            with open(path + '\\trainval.txt') as test_ids:
                ids_content = test_ids.read()
                test_ids.close()
        else:
            with open(path + '\\test.txt') as test_ids:
                ids_content = test_ids.read()
                test_ids.close()
        ids_content = ids_content.split()
        self.test_imgs = []
        self.temple_imgs = []
        self.labels = []
        for each in ids_content:
            if 'txt' not in each:
                self.test_imgs.append(path + '\\' + each[:-4] + '_test.jpg')
                self.temple_imgs.append(path + '\\' + each[:-4] + '_temp.jpg')
            else:
                self.labels.append(path + '\\' + each)

        self.cnt = -1
        self.biaoding_box = [0, 0, 640, 640]
        self.polygon = [0, 0, 640, 0, 640, 640, 0, 640]
        self.length = len(self.labels)
        self.ids = list(range(self.length))

    def test_next(self):
        self.cnt += 1
        data, hard_mask, img1, biaoding_box_zoom = self.get_input(self.test_imgs[self.cnt],
                                                                  self.temple_imgs[self.cnt],
                                                                  self.biaoding_box,
                                                                  self.polygon)
        self.boxes_list = []
        with open(self.labels[self.cnt]) as labels:
            labels_content = labels.read()
            labels.close()
        labels_content = labels_content.split('\n')
        img2 = copy.deepcopy(img1)
        for i, each in enumerate(labels_content[:-1]):
            x0, y0, x1, y1 = int(each.split(' ')[0]), int(each.split(' ')[1]), int(each.split(' ')[2]), int(
                each.split(' ')[3])
            xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
            x0, y0, x1, y1 = xc - 0.25 * w, yc - 0.25 * h, xc + 0.25 * w, yc + 0.25 * h

            color = (0, 255, 0)  # 边框颜色红
            thickness = 3  # 边框厚度1
            img2 = np.ascontiguousarray(img2)
            img2 = cv2.rectangle(img2, (int(each.split(' ')[0]), int(each.split(' ')[1])),
                                 (int(each.split(' ')[2]), int(each.split(' ')[3])), color, thickness)

            self.boxes_list.append([int(x0), int(y0), int(x1), int(y1)])

        gt_file = open(self.path + '.\..\evaluation\gt_v1' + self.test_imgs[self.cnt][-18:-9] + '.txt', mode='w')
        for each in self.boxes_list:
            gt_file.write(str(each[0]) + ',')
            gt_file.write(str(each[1]) + ',')
            gt_file.write(str(each[2]) + ',')
            gt_file.write(str(each[3]) + ',')
            gt_file.write('1' + '\n')
        gt_file.close()
        return data, hard_mask, img1, biaoding_box_zoom, img2

    def aug(self, src, ex, mask_out):
        # aug
        right = np.random.randint(10, 20)
        up = np.random.randint(10, 15)
        fg = np.random.randint(0, 3)
        ex = random_resize(ex, right, up, fg)
        src = random_resize(src, right, up, fg)
        mask_out = random_resize(mask_out, right, up, fg)

        if self.args.rot:
            if np.random.randint(0, 3) == 2:
                angle = np.random.randint(low=-10, high=11) / 2
                ex = rotation(img=ex, angle=angle)
                src = rotation(img=src, angle=angle)
                mask_out = rotation(img=mask_out, angle=angle)
        if self.args.jit:
            if np.random.randint(0, 3) == 2:
                dx = np.random.randint(low=-10, high=11)
                dy = np.random.randint(low=-10, high=11)
                ex = random_jitter(ex, dx, dy)
                src = random_jitter(src, dx, dy)
                mask_out = random_jitter(mask_out, dx, dy)
        if self.args.flip and (np.random.randint(0, 2) == 1):
            sd = np.random.randint(low=0, high=6)
            ex = random_flip(ex, sd).copy()
            src = random_flip(src, sd).copy()
            mask_out = random_flip(mask_out, sd).copy()
        if self.args.chan_change and (np.random.randint(0, 2) == 1):
            sd = np.random.randint(low=0, high=6)
            ex = change_channel(ex, sd=sd)
            src = change_channel(src, sd=sd)

        return src, ex, mask_out

    def train_next(self, batch_size):
        datas = []
        masks = []
        if len(self.ids) >= batch_size:
            sample_ids = random.sample(self.ids, batch_size)
        else:
            sample_ids = random.sample(self.ids, len(self.ids))

        for k in sample_ids:
            self.ids.remove(k)

        if len(self.ids) == 0:
            self.ids = list(range(self.length))

        for i in sample_ids:
            img1 = cv2.imread(self.test_imgs[i])
            img2 = cv2.imread(self.temple_imgs[i])
            mask = np.zeros((480, 640))
            img1 = cv2.resize(img1, dsize=(640, 480))  # 输入大小resize到640x480
            img2 = cv2.resize(img2, dsize=(640, 480))

            with open(self.labels[i]) as labels:
                labels_content = labels.read()
                labels.close()
            labels_content = labels_content.split('\n')
            for i, each in enumerate(labels_content[:-1]):
                x0, y0, x1, y1 = int(each.split(' ')[0]), int(each.split(' ')[1]) * 0.75, int(each.split(' ')[2]), int(
                    each.split(' ')[3]) * 0.75
                xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
                x0, y0, x1, y1 = xc - 0.25 * w, yc - 0.25 * h, xc + 0.25 * w, yc + 0.25 * h
                mask[int(y0):int(y1), int(x0):int(x1)] = 1

            img1, img2, mask = self.aug(img1, img2, mask)

            if self.args.train_vis:
                img3 = np.zeros_like(img1)
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i, j] == 1:
                            img3[i, j, :] = img1[i, j, :]
                cv2.imshow('1', img1)
                cv2.imshow('2', img2)
                cv2.imshow('m', img3)
                cv2.waitKey()

            img1 = torch.FloatTensor(img1).permute(2, 0, 1).unsqueeze(0)
            img2 = torch.FloatTensor(img2).permute(2, 0, 1).unsqueeze(0)
            datas.append(torch.cat((img1, img2), dim=1))

            masks.append(torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0))

        return torch.cat(datas, dim=0).to(torch.device(self.args.device)), torch.cat(masks, dim=0).to(torch.device(self.args.device))

    def get_input(self, defection_img_path, normal_img_path, biaoding_box, polygon):
        img1 = cv2.imread(defection_img_path)
        img2 = cv2.imread(normal_img_path)

        assert img1.shape == img2.shape
        hard_mask = np.zeros_like(img1)
        biaoding_box_zoom = copy.deepcopy(biaoding_box)
        biaoding_box_center_y = (biaoding_box[3] + biaoding_box[1]) / 2.
        biaoding_box_center_x = (biaoding_box[2] + biaoding_box[0]) / 2.
        biaoding_box_width = biaoding_box[2] - biaoding_box[0]
        biaoding_box_height = biaoding_box[3] - biaoding_box[1]
        biaoding_box_scale_y = biaoding_box_height / img1.shape[0]
        biaoding_box_scale_x = biaoding_box_width / img1.shape[1]
        zoom_scale_x = (1.0 / biaoding_box_scale_x) ** 0.5
        zoom_scale_y = (1.0 / biaoding_box_scale_y) ** 0.5

        biaoding_box_zoom[0] = biaoding_box_center_x - biaoding_box_width * (zoom_scale_x - 0.5)
        biaoding_box_zoom[2] = biaoding_box_center_x + biaoding_box_width * (zoom_scale_x - 0.5)
        biaoding_box_zoom[1] = biaoding_box_center_y - biaoding_box_height * (zoom_scale_y - 0.5)
        biaoding_box_zoom[3] = biaoding_box_center_y + biaoding_box_height * (zoom_scale_y - 0.5)

        top_size, bottom_size, left_size, right_size = biaoding_box[1] - biaoding_box_zoom[1], \
                                                       biaoding_box_zoom[3] - biaoding_box[3], \
                                                       biaoding_box[0] - biaoding_box_zoom[0], \
                                                       biaoding_box_zoom[2] - biaoding_box[2],

        for each, i in zip(biaoding_box_zoom, range(len(biaoding_box_zoom))):
            biaoding_box_zoom[i] = int(each)

        roi1 = img1[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]
        roi2 = img2[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]
        roi1_pad = cv2.copyMakeBorder(roi1, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                      cv2.BORDER_CONSTANT, value=0)
        roi2_pad = cv2.copyMakeBorder(roi2, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                      cv2.BORDER_CONSTANT, value=0)

        roi1 = cv2.resize(roi1_pad, dsize=(640, 480))  # 输入大小resize到640x480
        roi2 = cv2.resize(roi2_pad, dsize=(640, 480))

        roi1 = torch.FloatTensor(roi1).permute(2, 0, 1).unsqueeze(0)
        roi2 = torch.FloatTensor(roi2).permute(2, 0, 1).unsqueeze(0)
        data = torch.cat((roi1, roi2), dim=1).to(torch.device(self.args.device))

        hard_mask = polygon2mask(polygon=polygon, mask=hard_mask)

        return data, hard_mask, img1, biaoding_box_zoom

    def get_iou(self, bbox1, bbox2):
        """
        Calculates the intersection-over-union of two bounding boxes.
        """
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]
        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2
        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)
        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0
        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection
        return size_intersection / size_union

    def save_pred_boxes(self, result_box):
        # pre_file = open('E:\DL_Projects\mAP-master' + '/input/detection-results/' + str(self.cnt) + '.txt', mode='w')
        # print(self.cnt)
        res_file = open(self.path + '.\..\evaluation\\res_v1' + self.test_imgs[self.cnt][-18:-9] + '.txt', mode='w')
        # print(self.test_imgs[self.cnt][-18:-9])
        for each in result_box:
            iou = [self.get_iou(each, self.boxes_list[i]) for i in range(len(self.boxes_list))]
            confidence = max(iou)
            print(iou)

            res_file.write(str(each[0]) + ',')
            res_file.write(str(each[1]) + ',')
            res_file.write(str(each[2]) + ',')
            res_file.write(str(each[3]) + ',')
            res_file.write(str(confidence) + ',')
            res_file.write('1' + '\n')

        res_file.close()

    def add_gt_boxes(self, img):
        for box in self.boxes_list:
            color = (0, 255, 0)  # 边框颜色绿
            thickness = 3  # 边框厚度1
            img = np.ascontiguousarray(img)
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)

        return img


def train(args):
    net = Net(Norm=args.Norm_type).to(torch.device(args.device))

    if args.finetune:
        checkpoint = torch.load(args.pretrained) if args.device == 'gpu' else torch.load(args.pretrained,
                                                                                         map_location='cpu')
        net = get_state_static(net, checkpoint)

    net.train()
    param_groups = [{'params': net.bias_parameters()},
                    {'params': net.weight_parameters()},
                    {'params': net.other_parameters()}]

    optimizer = torch.optim.AdamW(params=param_groups, lr=args.LR, weight_decay=args.wd)
    pcb = Deep_PCB(path=args.DeepPCB_path, is_train=True if args.run_mode=='train' else False, args=args)

    for i in range(args.iters):
        data, label = pcb.train_next(batch_size=args.batch_size)
        out = net(data)
        loss_bce = nn.BCEWithLogitsLoss()(out, label)
        loss = loss_bce
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('iter:[%d/%d], lr: %.8f, loss: %.6f loss_bce: %.6f, loss_dice: %.6f,' % (i, args.iters,
                                                                                       optimizer.param_groups[0]['lr'],
                                                                                       loss, loss_bce, 0,))
        if (i + 1) % args.check_num == 0:
            torch.save(net.state_dict(), './checkpoint_' + str(i + 1) + '.pth')
    torch.save(net.state_dict(), './final_9111.pth')
    print('done')


def val(args):
    net = Net(Norm=args.Norm_type).to(torch.device(args.device))

    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained) if args.device == 'gpu' else torch.load(args.pretrained,
                                                                                         map_location='cpu')
        net = get_state_static(net, checkpoint)
    net.eval()

    pcb = Deep_PCB(path=args.DeepPCB_path, is_train=True if args.run_mode=='train' else False, args=args)

    for i in tqdm(range(pcb.length)):
        data, hard_mask, img1, biaoding_box_zoom, gt = pcb.test_next()
        biaoding_box, polygon = pcb.biaoding_box, pcb.polygon

        out = net(data)
        diff_map = out.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        temp = data[:, 3:6, :, :].squeeze(0).permute(1, 2, 0).cpu().numpy()
        inp = data[:, 0:3, :, :].squeeze(0).permute(1, 2, 0).cpu().numpy()
        bbox_on_ex, pre_boxes = get_bbox_from_mask_zoom_pcb(diff_map, img1, biaoding_box_zoom, args.diff_threshold,
                                                            polygon)  # 在框出来的图片中将缺陷标注出来
        pcb.save_pred_boxes(pre_boxes)

        bbox_on_ex = bbox_on_ex * hard_mask

        if args.show_result:
            heatmapshow = None
            heatmapshow = cv2.normalize(diff_map, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8U)
            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
            heatmapshow = cv2.addWeighted(heatmapshow, 0.7, inp.astype(np.uint8), 0.3, 0)

            cv2.imshow('result', heatmapshow)
            cv2.imshow('inp', inp / 255)
            cv2.imshow('temp', temp / 255)
            cv2.imshow('gt', gt / 255)
            cv2.imshow('bbox_on_ex', bbox_on_ex / 255)
            cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', default='val', choices=['train', 'val'], help='run mode')
    parser.add_argument('--jit', default=True, help='augmentation of random jitter')
    parser.add_argument('--rot', default=True, help='augmentation of random rotation')
    parser.add_argument('--chan_change', default=True, help='augmentation of channel swap')
    parser.add_argument('--crop', default=True, help='augmentation of random crop')
    parser.add_argument('--flip', default=True, help='augmentation of random flip')
    parser.add_argument('--noise', default=True, help='augmentation of gaussian noise')
    parser.add_argument('--label_smoothing', default=True, help='label smoothing')
    parser.add_argument('--light', default=True, help='augmentation of virtual light')
    parser.add_argument('--train_vis', default=False, help='view training data')
    parser.add_argument('--finetune', default=False, help='load pretrained model')
    parser.add_argument('--batch_size', default=16, type=int, help='learning rate')
    parser.add_argument('--LR', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--wd', default=0.001, type=float, help='weight decay')
    parser.add_argument('--iters', default=31500, type=int, help='number of steps to learn, DeepPCB has 1000 in train list, iters=epochs*1000/batch_size')
    parser.add_argument('--check_num', default=6300, type=int, help='number of steps to save checkpoint')
    parser.add_argument('--Norm_type', default='instance', choices=['batch', 'layer', 'instance', 'none'],
                        help='normalization function')
    parser.add_argument('--device', default='cpu', help='gpu or cpu')
    parser.add_argument('--save_result', action='store_true', help='write the inference result to ./result')
    parser.add_argument('--show_result', default=False, help='show the inference result')
    parser.add_argument('--diff_threshold', default=0.1, type=float, help='threshold to distinguish normal and '
                                                                          'anomaly(float number from 0.0 to 1.0)')
    parser.add_argument('--pretrained', default='./no_upload/v1.pth', metavar='N',
                        help='path to checkpoint file')
    parser.add_argument('--DeepPCB_path', default='E:\datasets\DeepPCB-master\PCBData', help='path to DeepPCB dataset')

    args = parser.parse_args()

    seed_torch(3407)

    if args.run_mode == 'train':
        train(args)
    else:
        val(args)