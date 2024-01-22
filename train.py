import argparse
import torch.utils.data
from network import *
from util import *

def make_datasets(batchsize=8, imgs=[], L=0):
    data = []
    labels = []
    for a in range(batchsize):
        num_diff = np.random.randint(low=12, high=20, size=None, dtype='l')  # 生成差异的数量
        mask = np.zeros((480, 640))  # 生成固定大小的空白图片（全黑）
        i = np.random.randint(low=0, high=L)
        src = cv2.imread(imgs[i])

        if args.crop and np.random.randint(0, 2) == 1:
            src = random_crop(src)

        src = cv2.resize(src, dsize=(640, 480))  # 随机选择背景图片resize到640x480
        H, W, C = src.shape
        ex = copy.deepcopy(src)  # 将背景图复制一份用来添加差异
        sd = np.random.randint(low=0, high=6)
        if args.chan_change and (np.random.randint(0, 2) == 1):
            ex = change_channel(ex, sd=sd)
            src = change_channel(src, sd=sd)
        if args.flip and (np.random.randint(0, 2) == 1):
            ex = random_flip(ex, sd)
            src = random_flip(src, sd)
        #########################################################
        for b in range(num_diff):
            # 随机确定左上角的点和长宽
            x_l, y_l, w, h = np.random.randint(low=39, high=W - 40, size=None, dtype='l'), \
                             np.random.randint(low=39, high=H - 40, size=None, dtype='l'), \
                             np.random.randint(low=10, high=25, size=None, dtype='l'), \
                             np.random.randint(low=10, high=25, size=None, dtype='l')
            # 在此方形区域内生成差异， 同时将空mask图片中的对应区域变为1（白色）作为标签
            if np.random.randint(0, 1) == 1:
                diff_patch = cv2.resize(cv2.imread(imgs[np.random.randint(low=0, high=L)]), dsize=(640, 480)) \
                    [y_l:y_l + h, x_l:x_l + w, :]
                ex[y_l:y_l + h, x_l:x_l + w, :] = diff_patch
                mask[y_l:y_l + h, x_l:x_l + w] = 1
            elif np.random.randint(0, 1) == 0:
                points = [[x_l + w // 2, y_l + h // 2]]
                for i in range(x_l, x_l + w):
                    for j in range(y_l, y_l + h):
                        if np.random.randint(0, 10) == 1:
                            points.append([i, j])
                random.shuffle(points)
                pts = np.asarray([points], dtype=np.int32)
                hull = get_hull(pts).transpose(1, 0, 2)
                mask = cv2.fillPoly(mask.copy(), hull, color=1)
                # ex_show = cv2.fillPoly(ex_show, hull, color=random_color())
                ex = cv2.fillPoly(ex.copy(), hull, color=(128, 128, 128))
                mask_filled = np.stack((mask, mask, mask), axis=2).astype(np.uint8)
                diff_img = cv2.imread(imgs[np.random.randint(0, L)])
                diff_img = cv2.resize(diff_img, dsize=(640, 480))
                mask_filled = mask_filled * diff_img
                ex += mask_filled

            else:
                diff_img = cv2.imread(imgs[np.random.randint(0, L)])
                diff_img = cv2.resize(diff_img, dsize=(w, h))
                diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
                diff_mask = cv2.threshold(diff_gray, diff_gray.mean() * 0.75, 1, cv2.THRESH_BINARY)[1]
                for i in range(x_l, x_l + w):
                    for j in range(y_l, y_l + h):
                        if diff_mask[j - y_l, i - x_l] == 0:
                            ex[j, i, :] = diff_img[j - y_l, i - x_l, :]
                            mask[j, i] = 1

        if args.label_smoothing:
            mask[mask == 1] = 0.95
            mask[mask == 0] = 0.05
        #########################################################
        if args.light:
            if np.random.randint(0, 3) == 2:
                ex = virtual_light(ex)
            elif np.random.randint(0, 3) == 1:
                src = virtual_light(src)
        if args.rot:
            if np.random.randint(0, 3) == 2:
                angle = np.random.randint(low=-5, high=6) / 2
                ex = rotation(img=ex, angle=angle)
                mask = rotation(img=mask, angle=angle)
            elif np.random.randint(0, 3) == 1:
                angle = np.random.randint(low=-5, high=6) / 2
                src = rotation(img=src, angle=angle)
        if args.jit:
            if np.random.randint(0, 3) == 2:
                dx = np.random.randint(low=-10, high=11)
                dy = np.random.randint(low=-10, high=11)
                ex = random_jitter(ex, dx, dy)
                mask = random_jitter(mask, dx, dy)
            elif np.random.randint(0, 3) == 1:
                dx = np.random.randint(low=-10, high=11)
                dy = np.random.randint(low=-10, high=11)
                src = random_jitter(src, dx, dy)
        if args.noise:
            if np.random.randint(0, 3) == 2:
                src = noisy(noise_typ='gauss', image=src)
                ex = noisy(noise_typ='gauss', image=ex)
            elif np.random.randint(0, 3) == 1:
                ex = noisy(noise_typ='gauss', image=ex)

        fg = np.random.randint(0, 3)
        right = np.random.randint(40, 80)
        up = np.random.randint(30, 60)
        ex = random_resize(ex, right, up, fg)
        src = random_resize(src, right, up, fg)
        mask = random_resize(mask, right, up, fg)

        if args.train_vis:
            cv2.imshow('mask', mask)
            cv2.imshow('src', src)
            cv2.imshow('ex', ex)
            cv2.waitKey()

        data1 = torch.FloatTensor(ex).permute(2, 0, 1).unsqueeze(0)
        data2 = torch.FloatTensor(src).permute(2, 0, 1).unsqueeze(0)
        mask_label = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0)

        data.append(torch.cat((data1, data2), dim=1))
        labels.append(mask_label)

    return torch.cat(data, dim=0).to(torch.device(args.device)), torch.cat(labels, dim=0).to(torch.device(args.device))


def train(args, imgs, L):

    net = Net(Norm=args.norm_type).to(torch.device(args.device))

    if args.finetune:
        checkpoint = torch.load(args.pretrained) if args.device == 'gpu' else torch.load(args.pretrained,
                                                                                         map_location='cpu')
        net = get_state_static(net, checkpoint)

    net.train()
    param_groups = [{'params': net.bias_parameters()},
                    {'params': net.weight_parameters()},
                    {'params': net.other_parameters()}]

    optimizer = torch.optim.AdamW(params=param_groups, lr=args.LR, weight_decay=args.wd)

    for i in range(args.iters):
        data, label = make_datasets(batchsize=args.batch_size, imgs=imgs, L=L)
        out = net(data)

        loss_bce = nn.BCEWithLogitsLoss()(out, label)
        loss = loss_bce
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print('iter:[%d/%d], lr: %.8f, weight_decay: %.6f,loss: %.6f loss_bce: %.6f loss_dice: %.6f, ' % (i, args.iters,
                                                                                               optimizer.param_groups[
                                                                                                   0]['lr'],
                                                                                               optimizer.param_groups[
                                                                                                   0]['weight_decay'],
                                                                                               loss, loss_bce, 0))
        if (i + 1) % args.check_num == 0:
            torch.save(net.state_dict(), './checkpoint_' + str(i + 1) + '.pth')
    torch.save(net.state_dict(), './final.pth')
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jit', default=True, help='augmentation of random jitter')
    parser.add_argument('--rot', default=True, help='augmentation of random rotation')
    parser.add_argument('--chan_change', default=True, help='augmentation of channel swap')
    parser.add_argument('--crop', default=True, help='augmentation of random crop')
    parser.add_argument('--flip', default=True, help='augmentation of random flip')
    parser.add_argument('--noise', default=True, help='augmentation of gaussian noise')
    parser.add_argument('--label_smoothing', default=True, help='label smoothing')
    parser.add_argument('--light', default=True, help='augmentation of virtual light')
    parser.add_argument('--train_vis', default=True, help='view training data')
    parser.add_argument('--pretrained', default='./final_4.pth', help='path to pretrained model')
    parser.add_argument('--finetune', default=False, help='load pretrained model')
    parser.add_argument('--norm_type', default='instance', choices=['batch', 'layer', 'instance', 'none'],
                        help='normalization function')
    parser.add_argument('--device', default='cpu', help='gpu or cpu')
    parser.add_argument('--batch_size', default=1, type=int, help='learning rate')
    parser.add_argument('--LR', default=0.001, type=float, help='learning rate')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--iters', default=10000, type=int, help='number of steps to learn')
    parser.add_argument('--check_num', default=1000, type=int, help='number of steps to save checkpoint')
    parser.add_argument('--bg_data', default='', type=str, help='path to the dictionary of background images')
    args = parser.parse_args()

    imgs = glob.glob(args.bg_data + '*.jpg')
    imgs.extend(glob.glob(args.bg_data + '*/' + '*.png'))
    imgs.extend(glob.glob(args.bg_data + '*/' + '*.bmp'))
    L = len(imgs)

    seed_torch(3407)

    train(args, imgs, L)