import copy
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import random
import torchvision.transforms
from skimage.metrics import structural_similarity
from scipy.stats import truncnorm


def seed_torch(seed=1029):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def get_state_static(model, data):
    if data is not None:
        model_dict = model.state_dict()
        print('model keys')
        print('=================================================')
        for k, v in model_dict.items():
            print(k)
        print('=================================================')

        print('data keys')
        print('=================================================')
        for k, v in data.items():
            print(k)
        print('=================================================')

        pretrained_dict = {k: v for k, v in data.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        print('load the following keys from the pretrained model')
        print('=================================================')
        for k, v in pretrained_dict.items():
            print(k)
        print('=================================================')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print('No pretrained')
    return model


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def predict_image(in_planes):
    return nn.Conv2d(in_planes, 3, kernel_size=3, stride=1, padding=1, bias=False)


def conv(Norm, in_planes, out_planes, kernel_size=3, stride=1):
    if Norm == 'instance':
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.InstanceNorm2d(out_planes),
            nn.GELU(),
        )
    elif Norm == 'batch':
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.GELU(),
        )
    elif Norm == 'layer':
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            LayerNorm(out_planes, data_format='channels_first'),
            nn.GELU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.GELU(),
        )


def predict_mask(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.GELU(),
    )


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def rgb_norm(image):
    r, g, b = cv2.split(image)
    r = (r - np.mean(r)) / np.std(r)
    g = (g - np.mean(g)) / np.std(g)
    b = (b - np.mean(b)) / np.std(b)
    return cv2.merge((r, g, b))


def random_flip(input, flag):
    if flag == 1:
        return np.fliplr(input)
    elif flag == 2:
        return np.flipud(input)
    elif flag == 3:
        return np.flipud(np.fliplr(input))
    else:
        return input


def random_jitter(input, dx, dy):
    H = np.float32([[1, 0, dx], [0, 1, dy]])  # 定义平移矩阵
    rows, cols = input.shape[:2]  # 获取图像高宽(行列数)
    res = cv2.warpAffine(input, H, (cols, rows))
    return res

def get_hull(axis_list):
    hull = cv2.convexHull(axis_list, clockwise=True, returnPoints=True)
    return hull

def random_resize(img, right, up, fg):
    if fg == 0:
        if len(img.shape) == 3:
            img_croped = img[20:460, 20:620, :]
        else:
            img_croped = img[20:460, 20:620]
        img_resized = cv2.resize(img_croped, dsize=(640, 480))
    elif fg == 1:
        img_paded = cv2.copyMakeBorder(img, int(up), int(up), int(right), int(right),
                                       cv2.BORDER_CONSTANT, value=0)
        img_resized = cv2.resize(img_paded, dsize=(640, 480))
    else:
        img_resized = img
    return img_resized.copy()


def rotation(img, angle):
    rows = img.shape[0]
    cols = img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=angle, scale=1)  # 向左旋转angle度并缩放为原来的scale倍
    img = cv2.warpAffine(img, M, (cols, rows))  # 第三个参数是输出图像的尺寸中心
    return img


def random_crop(img):
    if np.random.randint(low=1, high=101, size=None, dtype='l') % 2 == 0:
        xl, yl = np.random.randint(low=0, high=img.shape[1] // 2, size=None, dtype='l'), np.random.randint(low=0, high=
        img.shape[0] // 2, size=None, dtype='l')
        img = img[yl:yl + img.shape[0] // 2, xl:xl + img.shape[1] // 2, :]
    return img


def random_pad(img):
    right = np.random.randint(100, 200)
    up = np.random.randint(100, 200)
    img = cv2.copyMakeBorder(img, int(up), int(up), int(right), int(right),
                             cv2.BORDER_CONSTANT, value=0)
    return img


def brightness_adjustment(img):
    blank = np.zeros_like(img)
    c = (1 + np.random.randint(low=-5, high=6, size=None, dtype='l') / 10.)
    img = cv2.addWeighted(img, c, blank, 1 - c, 0)
    return img


def virtual_light_2(img):
    # 获取图像行和列
    rows, cols = img.shape[:2]
    # 设置中心点和光照半径
    centerX = np.random.randint(50, 590)
    centerY = np.random.randint(50, 430)
    radius = min(centerX, centerY)
    # 设置光照强度
    strength = 100 + np.random.randint(-20, 20)
    # 新建目标图像
    dst = np.zeros((rows, cols, 3), dtype="uint8")
    # 图像光照特效
    x = 1 if np.random.randint(0, 2) == 1 else -1
    for i in range(rows):
        for j in range(cols):
            distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
            # 获取原始图像
            B, G, R = img[i, j]
            if (distance < radius * radius):
                # 按照距离大小计算增强的光照值
                result = (int)(strength * (1.0 - math.sqrt(distance) / radius))
                result = x * result
                B = img[i, j][0] + result
                G = img[i, j][1] + result
                R = img[i, j][2] + result
                # 判断边界 防止越界
                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))
                dst[i, j] = np.uint8((B, G, R))
            else:
                dst[i, j] = np.uint8((B, G, R))
    # cv2.imshow('l', np.vstack((img, dst)))
    # cv2.waitKey()

    return dst

def random_color():
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    return (b, g, r)

def change_channel(img, sd):
    b = cv2.split(img)[0]
    g = cv2.split(img)[1]
    r = cv2.split(img)[2]
    if sd == 0:
        out = cv2.merge([b, r, g])
    elif sd == 1:
        out = cv2.merge([b, g, r])
    elif sd == 2:
        out = cv2.merge([r, g, b])
    elif sd == 3:
        out = cv2.merge([r, b, g])
    elif sd == 4:
        out = cv2.merge([g, b, r])
    elif sd == 5:
        out = cv2.merge([g, r, b])
    return out


def gasuss_noise(image, mu=0.0, sigma=0.01):
    """
	 添加高斯噪声
	:param image: 输入的图像
	:param mu: 均值
	:param sigma: 标准差
	:return: 含有高斯噪声的图像
	"""
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise
    if gauss_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    gauss_noise = np.clip(gauss_noise, low_clip, 1.0)
    gauss_noise = np.uint8(gauss_noise * 255)
    return gauss_noise


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.001
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image / 255 + gauss
        noisy = cv2.normalize(noisy, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        noisy = np.uint8(noisy * 255)
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def conv_dw(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=1, groups=in_planes,
                  bias=False),
        nn.BatchNorm2d(in_planes),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),

        nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


def polygon2mask(polygon, mask):
    points = np.array(polygon)

    # 使用多边形的点坐标创建一个包含多边形形状的路径
    path = points.reshape((-1, 1, 2))

    # 使用cv2.fillPoly()函数将路径填充到掩码中
    cv2.fillPoly(mask, [path], (1, 1, 1))

    # 返回生成的掩码
    return mask


def get_list0(gj_area_file):
    path1 = "./DrawRect/biaozhun/labels/" + gj_area_file + ".txt"
    if not os.path.exists(path1):
        print("该型号标准位置文件缺失/或输入型号与对应标准文件名称不一致")
    file1 = open(path1, 'r')
    lines = file1.readlines()

    zb0, list0 = [], []
    for i in range(len(lines)):  # 取坐标
        if lines[i] != '(pt1,pt2):\n':
            zb0.append(lines[i][:-1])
    # print(zb0)
    for i in range(0, len(zb0)):  # 转换整数,获得多边形点列表
        zb0[i] = int(zb0[i])  # 多边形点列表
    # print(zb0)

    # 获得多边形最大矩形框点列表
    x_min, y_min, x_max, y_max = zb0[0], zb0[1], zb0[0], zb0[1]
    for i in range(0, len(zb0), 2):
        if x_min > zb0[i]:
            x_min = zb0[i]
        if x_max < zb0[i]:
            x_max = zb0[i]
        if y_min > zb0[i + 1]:
            y_min = zb0[i + 1]
        if y_max < zb0[i + 1]:
            y_max = zb0[i + 1]
    list0 = [x_min, y_min, x_max, y_max]  # 多边形最大矩形框点列表

    return zb0, list0


def get_bbox_from_mask_zoom(result, ex, biaoding_box):
    # 获取mask（灰度图）
    mask = cv2.normalize(result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mask = np.uint8(mask * 255)
    # 转换成二值图
    ret, mask = cv2.threshold(mask, mask.mean() * 0.1, 255, cv2.THRESH_BINARY)

    # blocksize = 31
    # C = -mask.mean()
    # mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, C)

    def mask_find_bboxs(mask):
        retval, labels, stats, centimgds = cv2.connectedComponentsWithStats(mask,
                                                                            connectivity=8)  # connectivity参数的默认值为8
        stats = stats[stats[:, 4].argsort()]
        return stats[:-1]  # 排除最外层的连通图

    bboxs = mask_find_bboxs(mask)
    width_scale = (biaoding_box[2] - biaoding_box[0]) / 640.
    height_scale = (biaoding_box[3] - biaoding_box[1]) / 480.
    S_threshold = (biaoding_box[3] - biaoding_box[1]) * (biaoding_box[2] - biaoding_box[0]) * 0.0001
    err_threshold = 0.1
    # print('err_thrashold:{}, S_thrashold:{}'.format(err_threshold, S_threshold))
    have_diff = False

    for b in bboxs:
        x0, y0 = b[0], b[1]
        x1 = b[0] + b[2]
        y1 = b[1] + b[3]
        start_point, end_point, bbox_width, bbox_height = (x0, y0), (x1, y1), x1 - x0, y1 - y0
        start_point = (int(x0 * width_scale + biaoding_box[0]), int(y0 * height_scale + biaoding_box[1]))
        end_point = (int(x1 * width_scale + biaoding_box[0]), int(y1 * height_scale + biaoding_box[1]))
        bbox_width = (x1 * width_scale + biaoding_box[0]) - (x0 * width_scale + biaoding_box[0])
        bbox_height = (y1 * height_scale + biaoding_box[1]) - (y0 * height_scale + biaoding_box[1])

        err = 0.01378330908618641 * max(bbox_height, bbox_width)
        S_box = bbox_height * bbox_width
        # print('err:{}, S_box:{}, is_box:{}'.format(err, S_box, (bbox_width / bbox_height) >= 0.33 and (
        #         bbox_width / bbox_height) <= 3.0))

        if (bbox_width / bbox_height) >= 0.1 and (
                bbox_width / bbox_height) <= 10.0 and S_box >= S_threshold and err >= err_threshold:
            have_diff = True
            color = (0, 0, 255)  # 边框颜色红
            thickness = 3  # 边框厚度1
            ex = np.ascontiguousarray(ex)
            ex = cv2.rectangle(ex, start_point, end_point, color, thickness)
            # pixel_err = max(int(8.55 * bbox_width), int(5.34 * bbox_height))
            # ex = cv2.putText(ex, str(err), (int(start_point[0]+bbox_width/2), int(start_point[1]+bbox_height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
    if have_diff:
        print('有异物')
    else:
        print('无异物')
    return ex


from shapely.geometry import Polygon, box


def is_intersect_poly_rect(poly_coords, rect_coords):
    # 将多边形的顶点列表转换为多边形对象
    poly = Polygon([(poly_coords[i], poly_coords[i + 1]) for i in range(0, len(poly_coords), 2)])
    # 将矩形框的坐标列表转换为矩形对象
    rect = box(rect_coords[0], rect_coords[1], rect_coords[2], rect_coords[3])
    # 判断多边形和矩形是否相交
    if not poly.intersects(rect):
        return False
    # 计算它们的相交面积
    intersect_area = poly.intersection(rect).area
    # 判断相交面积是否大于矩形面积的十分之一
    if intersect_area > rect.area / 2:
        return True
    else:
        return False


def get_bbox_from_mask_zoom_pcb(result, ex, biaoding_box, diff_threshold, polygon):
    # 获取mask（灰度图）
    mask = cv2.normalize(result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mask = np.uint8(mask * 255)

    # 转换成二值图
    ret, mask = cv2.threshold(mask, 255 * diff_threshold, 255, cv2.THRESH_BINARY)    #pcb
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    width_scale = (biaoding_box[2] - biaoding_box[0]) / 640.
    height_scale = (biaoding_box[3] - biaoding_box[1]) / 480.
    S_threshold = (biaoding_box[3] - biaoding_box[1]) * (biaoding_box[2] - biaoding_box[0]) * 0.0001
    err_threshold = 0.1
    have_diff = False
    count_diff = 0
    pre_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # 获取轮廓顶点及边长
        start_point, end_point, bbox_width, bbox_height = (x, y), (x + w, y + h), w, h
        start_point = (int(x * width_scale + biaoding_box[0]), int(y * height_scale + biaoding_box[1]))
        end_point = (int((x + w) * width_scale + biaoding_box[0]), int((y + h) * height_scale + biaoding_box[1]))
        bbox_width = ((x + w) * width_scale + biaoding_box[0]) - (x * width_scale + biaoding_box[0])
        bbox_height = ((y + h) * height_scale + biaoding_box[1]) - (y * height_scale + biaoding_box[1])
        err = 0.01378330908618641 * max(bbox_height, bbox_width)
        S_box = bbox_height * bbox_width

        if (bbox_width / bbox_height) >= 0.1 and (
                bbox_width / bbox_height) <= 10.0 and S_box >= S_threshold and err >= err_threshold:
            x0, y0, x1, y1 = start_point[0], start_point[1], end_point[0], end_point[1]
            if is_intersect_poly_rect(polygon, [x0, y0, x1, y1]):
                pre_boxes.append([x0, y0, x1, y1])
                have_diff = True
                count_diff += 1
                color = (0, 0, 255)  # 边框颜色红
                thickness = 3  # 边框厚度1
                ex = np.ascontiguousarray(ex)
                ex = cv2.rectangle(ex, start_point, end_point, color, thickness)

    if have_diff:
        # print('有异物:', count_diff)
        pass
    else:
        # print('无异物')
        pass
    return ex, pre_boxes


def get_bbox_from_map(args, diff_map, ex, biaoding_box, polygon):

    # # 获取mask（灰度图）
    mask = cv2.normalize(diff_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    mask = np.uint8(mask * 255)

    # 转换成二值图
    diff_confidence = args.diff_threshold
    diff_thresh = diff_confidence * (255 - mask.mean()) + mask.mean()
    ret, mask = cv2.threshold(mask, diff_thresh, 255, cv2.THRESH_BINARY)  # diff

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    width_scale = (biaoding_box[2] - biaoding_box[0]) / 640.
    height_scale = (biaoding_box[3] - biaoding_box[1]) / 480.
    S_threshold = (biaoding_box[3] - biaoding_box[1]) * (biaoding_box[2] - biaoding_box[0]) * 0.0001
    err_threshold = 0.1
    have_diff = False
    count_diff = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # 获取轮廓顶点及边长
        start_point = (int(x * width_scale + biaoding_box[0]), int(y * height_scale + biaoding_box[1]))
        end_point = (int((x + w) * width_scale + biaoding_box[0]), int((y + h) * height_scale + biaoding_box[1]))
        bbox_width = ((x + w) * width_scale + biaoding_box[0]) - (x * width_scale + biaoding_box[0])
        bbox_height = ((y + h) * height_scale + biaoding_box[1]) - (y * height_scale + biaoding_box[1])
        err = 0.01378330908618641 * max(bbox_height, bbox_width)
        S_box = bbox_height * bbox_width

        if (bbox_width / bbox_height) >= 0.1 and (
                bbox_width / bbox_height) <= 10.0 and S_box >= S_threshold and err >= err_threshold:
            x0, y0, x1, y1 = start_point[0], start_point[1], end_point[0], end_point[1]
            if is_intersect_poly_rect(polygon, [x0, y0, x1, y1]):
                have_diff = True
                count_diff += 1
                color = (0, 0, 255)  # 边框颜色红
                thickness = 3  # 边框厚度1
                ex = np.ascontiguousarray(ex)
                ex = cv2.rectangle(ex, start_point, end_point, color, thickness)

    if have_diff:
        print('有异物:', count_diff)
    else:
        print('无异物')
    return ex

def get_demo_input_zoom(args, biaoding_box, polygon):
    img1 = cv2.imread(args.input)
    img2 = cv2.imread(args.temp)

    assert img1.shape == img2.shape
    hard_mask = np.zeros_like(img1)

    roi1 = img1[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]
    roi2 = img2[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :]

    biaoding_box_zoom = copy.deepcopy(biaoding_box)
    biaoding_box_center_y = (biaoding_box[3] + biaoding_box[1]) / 2.
    biaoding_box_center_x = (biaoding_box[2] + biaoding_box[0]) / 2.
    biaoding_box_width = biaoding_box[2] - biaoding_box[0]
    biaoding_box_height = biaoding_box[3] - biaoding_box[1]
    biaoding_box_scale_y = biaoding_box_height / img1.shape[0]
    biaoding_box_scale_x = biaoding_box_width / img1.shape[1]
    zoom_scale_x = (1.0 / biaoding_box_scale_x) ** 0.3
    zoom_scale_y = (1.0 / biaoding_box_scale_y) ** 0.3
    # zoom_scale_y, zoom_scale_x = 1, 1

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

    roi1_pad = cv2.copyMakeBorder(roi1, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                  cv2.BORDER_CONSTANT, value=0)
    roi2_pad = cv2.copyMakeBorder(roi2, int(top_size), int(bottom_size), int(left_size), int(right_size),
                                  cv2.BORDER_CONSTANT, value=0)

    roi1 = cv2.resize(roi1_pad, dsize=(640, 480))  # 输入大小resize到640x480
    roi2 = cv2.resize(roi2_pad, dsize=(640, 480))

    roi1 = torch.FloatTensor(roi1).permute(2, 0, 1).unsqueeze(0)
    roi2 = torch.FloatTensor(roi2).permute(2, 0, 1).unsqueeze(0)
    data = torch.cat((roi1, roi2), dim=1)

    hard_mask = polygon2mask(polygon, hard_mask)

    return data, hard_mask, img1, biaoding_box_zoom


def get_demo_input(defection_img_path, normal_img_path):
    img1_src = cv2.imread(defection_img_path)
    img2_src = cv2.imread(normal_img_path)

    hard_mask = np.zeros_like(img1_src)

    img1_resize = cv2.resize(img1_src, dsize=(640, 480))  # 输入大小resize到640x480
    img2_resize = cv2.resize(img2_src, dsize=(640, 480))

    img1 = torch.cuda.FloatTensor(img1_resize).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.cuda.FloatTensor(img2_resize).permute(2, 0, 1).unsqueeze(0)
    data = torch.cat((img1, img2), dim=1)

    return data, hard_mask, img1_resize


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


from bisect import bisect_right
import matplotlib.pyplot as plt


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1 / 3,
            warmup_iters=100,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score

def un_label_smooth(mask):
    un_smooth = copy.deepcopy(mask)
    un_smooth[un_smooth >= 0.5] = 1
    un_smooth[un_smooth <= 0.5] = 0
    return un_smooth


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce=None)
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def virtual_light(img):
    # 获取图像行和列
    rows, cols = img.shape[:2]
    # 设置中心点和光照半径
    centerX = np.random.randint(50, 590)
    centerY = np.random.randint(50, 430)
    radius = min(centerX, centerY)
    # 设置光照强度
    strength = 50 + np.random.randint(-20, 20)
    x = 1 if np.random.randint(0, 2) == 1 else -1
    # 新建目标图像
    distance = (centerY - np.arange(rows)[:, np.newaxis] - 0.5) ** 2 + \
               (centerX - np.arange(cols)[np.newaxis, :] - 0.5) ** 2

    # 计算结果矩阵
    result = strength * (1 - np.sqrt(distance) / radius)
    result[distance >= radius ** 2] = 0
    result = np.clip(result * x, -255, 255).astype("int32")

    # 添加结果
    dst = np.clip(img + result[..., np.newaxis], 0, 255).astype("uint8")

    return dst


def img_merge(img1, img2):
    h, w = img1.shape[0], img1.shape[1]
    img = np.zeros((h * 2, w * 2, 3))
    img[::2, ::2, :] = img1
    img[1::2, 1::2, :] = img2
    return img


# from yolov5_circle.detect import single_detect


def get_rec_mask(roi, x, y, w, h):
    # cv2.imshow('roi', roi)

    extend_h, extend_w = 20, 20
    if min(x, roi.shape[1] - x - w) < extend_w:
        extend_w = min(x, roi.shape[1] - w - x)
    if min(y, roi.shape[0] - y - h) < extend_h:
        extend_h = min(y, roi.shape[0] - h - y)

    print(extend_h, extend_w)
    extend = roi[y - extend_h:y + h + extend_h, x - extend_w:x + w + extend_w, :]
    # cv2.imshow('extend', extend)

    extend = cv2.GaussianBlur(extend, (5, 5), 0, 0)
    # cv2.imshow('Gaussian', extend)

    extend = cv2.cvtColor(extend, cv2.COLOR_BGR2HSV)
    im1Gray = extend[:, :, 2]
    # cv2.imshow('v', im1Gray)

    mask = cv2.threshold(im1Gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    number_of_white_pix = np.sum(mask == 255)
    number_of_black_pix = np.sum(mask == 0)
    if number_of_white_pix <= number_of_black_pix:
        mask = cv2.threshold(im1Gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    k = np.ones((11, 11), np.uint8)  # 创建核
    mask = cv2.erode(mask, k, 20)

    a1 = np.zeros_like(extend)
    contour1, hierarchy1 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts1 = sorted(contour1, key=cv2.contourArea, reverse=True)
    result1 = cv2.drawContours(a1, cnts1, 0, (1, 1, 1), -1)

    # cv2.namedWindow('erode', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('max_counter', cv2.WINDOW_NORMAL)
    # cv2.imshow('erode', mask)
    # cv2.imshow('max_counter', result1 * 255)
    # cv2.imshow('src_size', result1[extend_h:-extend_h, extend_w:-extend_h, :] * 255)
    # cv2.waitKey()
    if extend_w == 0 and extend_h != 0:
        return result1[extend_h:-extend_h, :, :]
    if extend_h == 0 and extend_w != 0:
        return result1[:, extend_w:-extend_w, :]
    if extend_h == 0 and extend_w == 0:
        return result1
    return result1[extend_h:-extend_h, extend_w:-extend_w, :]


def roi_dectection_by_yolo(roi1, roi2, biaoding_box, hard_mask):
    sh, sw = roi1.shape[0], roi1.shape[1]

    cls, x1, y1, w1, h1 = single_detect(roi1)
    cls, x2, y2, w2, h2 = single_detect(roi2)

    if cls == 0:

        # cv2.imshow('roi', roi1)

        smaller = 6
        if min(w1, h1) <= 120:
            smaller = 5

        roi1 = roi1[y1:y1 + h1, x1:x1 + w1, :]
        roi2 = roi2[y2:y2 + h2, x2:x2 + w2, :]
        # cv2.imshow('roi_in_box', roi1)

        circle_mask1 = np.zeros_like(roi1)
        cv2.circle(circle_mask1, (int(w1 / 2), int(h1 / 2)), int(min(w1, h1) / 2), (1, 1, 1), -1)
        cv2.circle(circle_mask1, (int(w1 / 2), int(h1 / 2)), int(min(w1, h1) / smaller), (0, 0, 0), -1)
        # cv2.imshow('circle_mask', circle_mask1*255)

        roi1 = roi1 * circle_mask1
        # cv2.imshow('masked_roi', roi1)
        circle_mask2 = np.zeros_like(roi2)
        cv2.circle(circle_mask2, (int(w2 / 2), int(h2 / 2)), int(min(w2, h2) / 2), (1, 1, 1), -1)
        cv2.circle(circle_mask2, (int(w2 / 2), int(h2 / 2)), int(min(w2, h2) / smaller), (0, 0, 0), -1)

        roi2 = roi2 * circle_mask2

        biaoding_box[0] = biaoding_box[0] + x1
        biaoding_box[1] = biaoding_box[1] + y1
        biaoding_box[2] = biaoding_box[2] - (sw - w1 - x1)
        biaoding_box[3] = biaoding_box[3] - (sh - h1 - y1)

        mask_center = (int((biaoding_box[0] + biaoding_box[2]) / 2), int((biaoding_box[1] + biaoding_box[3]) / 2))
        r1 = int(min((biaoding_box[2] - biaoding_box[0], biaoding_box[3] - biaoding_box[1])) / 2)
        r2 = int(min((biaoding_box[2] - biaoding_box[0], biaoding_box[3] - biaoding_box[1])) / smaller)
        cv2.circle(hard_mask, mask_center, r1, (1, 1, 1), -1)
        cv2.circle(hard_mask, mask_center, r2, (0, 0, 0), -1)

    elif cls == 1:
        mask1 = get_rec_mask(roi1, x1, y1, w1, h1)
        mask2 = get_rec_mask(roi2, x2, y2, w2, h2)

        roi1 = roi1[y1:y1 + h1, x1:x1 + w1, :]
        roi2 = roi2[y2:y2 + h2, x2:x2 + w2, :]
        roi1 = roi1 * mask1
        roi2 = roi2 * mask2

        biaoding_box[0] = biaoding_box[0] + x1
        biaoding_box[1] = biaoding_box[1] + y1
        biaoding_box[2] = biaoding_box[2] - (sw - w1 - x1)
        biaoding_box[3] = biaoding_box[3] - (sh - h1 - y1)

        hard_mask[biaoding_box[1]:biaoding_box[3], biaoding_box[0]:biaoding_box[2], :] = mask1

    return roi1, roi2, biaoding_box, hard_mask, cls
