import os
import numpy as np
import torch
from PIL import Image
import time
from torchvision import transforms as T
import glob
import tqdm
import random


class LoadData(object):
    """

    """

    def __init__(self, mode='train', seq=2, w=1280, h=720):  # ss 表示跟踪txt 文件中未记录的帧数
        self.seq = seq
        self.w = w
        self.h = h
        self.in_net_size = (64, 64)
        if mode == 'train':
            flow_path = 'E:/cxy/datasets/hevi/hevi_val1/flow/'
            image_path = 'E:/cxy/datasets/hevi/hevi_val1/image/*'
        else:
            flow_path = 'E:/cxy/datasets/hevi/hevi_val1/flow/'
            image_path = 'E:/cxy/datasets/hevi/hevi_val1/image/*'

        self.all_flow_image = []
        image_files = glob.glob(image_path)
        image_files.sort()

        for image_file in image_files:
            video_seq = image_file
            flow_p = flow_path + video_seq[-3:] + '/'
            image_p = video_seq + '/'
            frame = os.listdir(image_p)
            frame.sort()
            for i in tqdm.tqdm(frame[:-1]):  # i+ss 即表示帧号
                flo_pths = [flow_p + '%06d' % (xx + int(i[:-4])) + '.png' for xx in range(1, seq)]
                img_paths = [image_p + '%06d' % (xx + int(i[:-4])) + '.jpg' for xx in range(seq)]
                # print(flo_pths)
                # print(img_paths)
                self.all_flow_image.append([flo_pths, img_paths])

        self.transforms = T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,), std=(0.5,))])

    # toTensor 将把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    # normalize 将取值变换为-1，1

    def __getitem__(self, index):
        flow, image = self.all_flow_image[index][0], self.all_flow_image[index][1]

        flow = self.read_imgs(flow)
        image = self.read_imgs(image)
        flow_tensor, image_tensor = [], []
        for i in range(len(image)):
            image_tensor.append(self.transforms(image[i]))
        for i in range(len(flow)):
            flow_tensor.append(self.transforms(flow[i]))
        # input_flow = torch.stack(flow_tensor, 0)
        #
        input_flow = flow_tensor[-1]
        input_image = image_tensor[:-1][0]
        #
        # print('flow',input_flow.size())

        target_image = image_tensor[-1]

        return input_image, input_flow, target_image

    def __len__(self):
        return len(self.all_flow_image)

    def read_imgs(self, pths):
        ims = []
        for pth in pths:
            try:
                ims.append(Image.open(pth).resize(self.in_net_size))
            except:
                print(pth)
        return ims

    def read_flows(self, pths):
        ims = []
        for pth in pths:
            try:
                ims.append(Image.open(pth).resize((64, 64)))
            except:
                print(pth)
        return ims
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import tqdm

    train_dataset = LoadData()
    # print(train_dataset.__len__())
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=False, )
    a = None
    for epoch in range(10):
        # print(train_loader.__len__())
        for input_image, input_flow in train_loader:
            # for (x1, x2) in train_loader:
            end_time = time.time()
