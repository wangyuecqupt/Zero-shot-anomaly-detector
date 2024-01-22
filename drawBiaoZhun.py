## 1 程序默认运行是直接绘多边形，直接点击即可，
## 绘制完成后点击右上角的X或按enter即可关闭图像并保存坐标
## 2 在默认情况下，单击鼠标中键或空格即可切换为矩形模式
## 3 在绘制矩形模式下只能通过按enter关闭图像并保存坐标
## 4 在绘制矩形模式下鼠标左键取消上一步操作或重新绘制矩形
## 5 在绘制多边形时鼠标右键取消上一步操作
## 6 画多边形183行左右---查找距离较近的，删除的功能要修改运行会出错

import copy
import json
import joblib
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import imutils
from win32 import win32gui, win32print
from win32.lib import win32con

WIN_NAME = 'draw_rect'


def get_list0(path):
    if not os.path.exists(path):
        print("记录该型号标准位置的文件缺失/或输入型号与其对应标准文件名称不一致")
    file1 = open(path, 'r')
    lines = file1.readlines()
    # for line in lines:
    #     if (any(kw in line for kw in kws)):
    #         SeriousFix.write(line + '\n')
    zb0, list0 = [], []
    for i in range(len(lines)):  # 取坐标
        if lines[i] != '(pt1,pt2):\n':
            zb0.append(lines[i][:-1])
    # print(zb0)
    for i in range(0, len(zb0)):  # 转换整数
        zb0[i] = int(zb0[i])
    # print(zb0)

    for i in range(0, len(zb0), 4):  # 每四个取一次，加入列表
        x0, y0, x1, y1 = zb0[i: i + 4]

        # 使点设为左上至右下
        if y1<=y0:
            temp = y0
            y0 = y1
            y1 = temp

        # print(x0,y0,x1,y1)
        list0.append([x0, y0, x1, y1])
    print("list0:", list0)
    file1.close()
    return list0


'''
        初始校验文件，文件名代表类型，检验时读取文件名作为类型判断标准
        打开sourse文件夹，读取标准件原始图片，保存标准位置到biaozhun/labels，保存画有标准位置的图片到biaozhun/imgs
'''
POLYLINES = False  # 多边形向矩形切换


def define_start(img_name, img_path, type):
    pts = []  # 用于存放点

    def draw_roi(event, x, y, flags, param):

        img2 = img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击，选择点
            pts.append((x, y))
            cv2.circle(img2, pts[-1], 3, (0, 255, 0), -1)
        #
        # if event == cv2.EVENT_MOUSEMOVE:  # 画圆
        #     if len(pts) >= 1:
        #         radius = np.sqrt(pow(x-pts[0][0],2) + pow(y-pts[0][1],2))
        #         radius = int(radius)
        #         rs.append(radius)
        #         cv2.circle(img2, pts[0], rs[-1], (0, 0, 255), 2)  # x ,y 为鼠标点击地方的坐标
        #


        if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击，取消最近一次选择的点
            if len(pts) >= 1:
                pts.pop()


        if event == cv2.EVENT_MBUTTONDOWN:  # 中键绘制轮廓
            global POLYLINES
            # print("MBUTTONDOWN:  # 中键绘制轮廓")
            POLYLINES = True

        if len(pts) > 0:
            # 将pts中的最后一点画出来
            cv2.circle(img2, pts[-1], 3, (0, 255, 0), -1)


        if len(pts) > 1:

            # 画线
            for i in range(len(pts) - 1):
                cv2.circle(img2, pts[i], 5, (0, 255, 0), -1)  # x ,y 为鼠标点击地方的坐标
                cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(0, 0, 255), thickness=2)
            cv2.line(img=img2, pt1=pts[0], pt2=pts[-1], color=(0, 0, 255), thickness=2)

        cv2.imshow(WIN_NAME, img2)

    def set_ratio(image):
        if image is None:
            return 0, 0, 0
        # print(image.shape)
        img_h, img_w = image.shape[:2]
        """获取真实的分辨率"""
        hDC = win32gui.GetDC(0)
        screen_w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)  # 横向分辨率
        screen_h = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)  # 纵向分辨率
        # print(img_w,img_h)

        num_wh = 1
        if img_w * img_h > 1.9e7:  # 两千万像素
            num_wh = 4
        elif img_w * img_h > 1.0e7:  # 一千万像素
            num_wh = 3
        elif min(img_w, img_h) >= min(screen_w, screen_h) or \
                max(img_w, img_h) >= max(screen_w, screen_h):
            num_wh = 2
        else:
            num_wh = 1

        ratio_h = int(img_h / num_wh)
        ratio_w = int(img_w / num_wh)

        return ratio_h, ratio_w, num_wh

    (filepath, file) = os.path.split(img_path)

    # file = 'r.jpg'      # 需要用户选择图片，传入图片的名称

    if file.endswith(".jpg") or file.endswith(".png"):  # 如果file以jpg结尾
        # img_dir = os.path.join(file_dir, file)
        image = cv2.imread(img_path)

        ratio_h, ratio_w, num_wh = set_ratio(image)
        if ratio_h == 0 and ratio_w == 0 and num_wh == 0:
            print("No image")

        txt_path = "./DrawRect/biaozhun/labels/%s.txt" % (img_name)
        open(txt_path, 'w').close()  # 清空文件数据
        f = open(txt_path, mode='a+')
        txt_save = []

        img = imutils.resize(image, width = ratio_w)
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, ratio_w, ratio_h)
        cv2.imshow(WIN_NAME, img)

        # 默认直接执行画多边形
        cv2.setMouseCallback(WIN_NAME, draw_roi)

        while True:
            w_key = cv2.waitKey(1)
            # enter 或回车键:

            if w_key == 13 or cv2.getWindowProperty(WIN_NAME, 0) == -1:
                for i in range(len(pts)):
                    if i == 0:
                      txt_save.append("(pt1,pt2):")
                    txt_save.append(str(pts[i][0] * num_wh))
                    txt_save.append(str(pts[i][1] * num_wh))

                num_txt_i = 0
                for txt_i in range(len(txt_save)):
                    txt_i = txt_i - num_txt_i
                    if txt_save[txt_i] == 'delete':
                        for j in ran7ge(6):
                            del txt_save[txt_i - j]
                        num_txt_i += 6
                for txt_i in txt_save:
                    f.write(str(txt_i) + '\n')
                print("txt_save:", txt_save)
                break
                f.close()

            # 空格切换至矩形
            if POLYLINES == True or w_key == 32:
                roi = cv2.selectROI(windowName=WIN_NAME, img=img, showCrosshair=False, fromCenter=False)
                x, y, w, h = roi
                cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)

                print('pt1: x = %d, y = %d' % (x, y))
                txt_save.append("(pt1,pt2):")
                txt_save.append(str(x * num_wh))
                txt_save.append(str(y * num_wh))
                txt_save.append(str((x + w) * num_wh))
                txt_save.append(str((y + h) * num_wh))

                # 用红色框显示ROI
                # cv2.imshow(WIN_NAME, img)
                # cv2.waitKey(0)

                # 保存txt坐标
                num_txt_i = 0
                for txt_i in range(len(txt_save)):
                    txt_i = txt_i - num_txt_i
                    if txt_save[txt_i] == 'delete':
                        for j in range(6):
                            del txt_save[txt_i - j]
                        num_txt_i += 6
                for txt_i in txt_save:
                    f.write(str(txt_i) + '\n')
                print("txt_save:", txt_save)
                # break
                f.close()

                # 查找距离较近的，删除
                points_list = get_list0(txt_path)
                new_points_list = []
                for i in points_list:
                    x0, y0, x1, y1 = i[0], i[1], i[2], i[3]
                    if abs(x1 - x0) > 5 and abs(y1 - y0) > 5:
                        new_points_list.append('(pt1,pt2):')
                        new_points_list.append(x0)
                        new_points_list.append(y0)
                        new_points_list.append(x1)
                        new_points_list.append(y1)
                print(new_points_list)
                file2 = open(txt_path, 'w')
                for i in new_points_list:
                    file2.write(str(i) + '\n')
                file2.close()
                break

        cv2.destroyAllWindows()

    else:
        print("输入图片类型错误！请输入JPG/PNG格式的图片！")

def biaoding(args):
    img_path = args.input
    img_name = img_path.split('/')[-1][:-4]
    define_start(img_name, img_path, 0)