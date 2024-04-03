import os

import tqdm
from torchvision.utils import save_image

from configs.config import args
from nets import MyGenerator, Linear_de, Linear_en
import numpy as np
import torch
from loss_function import intensity_loss, gradient_loss, psnr_error, My_iou
import pickle
import glob
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as T
import tqdm
from sklearn.metrics import roc_curve, auc, average_precision_score
import math
import numpy

from ssim import ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize generator
generator = MyGenerator(args).to(device)


# ----------
# Testing
# ----------

class Test(object):
    def __init__(self, generator, mode='test', seq=5, w=1280, h=720):  # ss 表示跟踪txt 文件中未记录的帧数
        self.seq = seq
        self.w = w
        self.h = h
        self.in_net_size = (256, 256)
        self.flow_path = 'F:/test/DADA/flow'
        self.image_path = 'F:/test/DADA/image/*'
        self.image_files = glob.glob(self.image_path)
        self.image_files.sort()
        self.generator = generator
        self.transforms = T.Compose([T.ToTensor(), T.Normalize(mean=(0.5,), std=(0.5,))])

    def gen_data(self):
        with torch.no_grad():
            for image_file in self.image_files:
                psnr = []
                sim = []
                video_seq = image_file
                flow_p = self.flow_path + '/'+video_seq[-3:] + '/'
                # flow_p = self.flow_path +video_seq[-3:] + '/'
                image_p = video_seq + '/'
                frame = os.listdir(image_p)
                frame.sort()
                print('starting test : ...')

                for iii in tqdm.tqdm(frame[:-4]):  # i+ss 即表示帧号

                    img_paths = [image_p + '%04d' % (xx + int(iii[:-4])) + '.png' for xx in range(self.seq)]
                    flo_pths = [flow_p + '%04d' % (xx + int(iii[:-4])) + '.png' for xx in range(1, self.seq-1)]
                    self.all_flow_image = ([flo_pths, img_paths])
                    flow, image, ddpm_flow  = self.all_flow_image[0], self.all_flow_image[1], self.all_flow_image[0]
                    flow = self.read_imgs(flow)
                    ddpm_flow = self.read_flows(ddpm_flow)
                    image = self.read_imgs(image)
                    flow_tensor, image_tensor, ddpm_flow_tensor = [], [], []
                    for i in range(len(image)):
                        image_tensor.append(self.transforms(image[i]))
                    for i in range(len(flow)):
                        flow_tensor.append(self.transforms(flow[i]))
                    for i in range(len(ddpm_flow)):
                            ddpm_flow_tensor.append(self.transforms(ddpm_flow[i]))
                    input_flow = torch.stack(flow_tensor, 0)
                    ddpm_flow = ddpm_flow_tensor[-1]
                    input_image = torch.cat(image_tensor[:-1], 0)

                    target_image = image_tensor[-1]

                    #  开始测试：
                    imgs, flow, ddpm_flow, gt_img = input_image, input_flow, ddpm_flow, target_image
                    imgs, flow, ddpm_flow, gt_img = imgs.unsqueeze(0), flow.unsqueeze(0), ddpm_flow.unsqueeze(0) , gt_img.unsqueeze(0)
                    img_inf = video_seq + '_' + iii

                    imgs, flow,ddpm_flow, gt_img = imgs.to(device), flow.to(device),ddpm_flow.to(device), gt_img.to(device)
                    gen_img = self.generator(imgs, flow, ddpm_flow)
                    real_outputs = gen_img* 0.5 + 0.5
                    fake_outputs = gt_img* 0.5 + 0.5
                    # print(float(ssim(real_outputs, fake_outputs)))
                    # # tensor = ((torch.cosine_similarity(real_outputs, fake_outputs))[0])
                    # # print(float(tensor[0][-1]))
                    # # normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
                    # # print(float(normalized_tensor[0][0]))
                    # print(float(psnr_error(real_outputs, fake_outputs)))
                    # print(float(psnr_error(real_outputs, fake_outputs)))
                    # save_image(real_outputs, os.path.join(
                    #      "F:/test/DADA/gen/{}".format(video_seq[-4:]),"gen_frame_{}.png".format(iii[:-4])), nrow=8)
                    # save_image(fake_outputs, os.path.join(
                    #      "E:/cxy/xuexi/models/result", "gt_frame_{}.png".format(iii)), nrow=8)
                    # sim.append([img_inf,float(ssim(real_outputs,fake_outputs))])  # torch.norm(real_outputs -  fake_outputs)
            #         psnr.append([img_inf, float(psnr_error(real_outputs, fake_outputs))])
            # # #  每一个序列存储结果
            #     with open('F:/test/DADA/{}_11.pkl'.format(video_seq[-4:]), 'wb') as f:
            #         pickle.dump([psnr], f)


    def read_imgs(self,pths):
        ims = []
        for pth in pths:
            # try:
                ims.append(Image.open(pth).resize(self.in_net_size))
            # except:
            #     print(pth)
        return ims

    def read_flows(self,pths):
        ims = []
        for pth in pths:
            # try:
                ims.append(Image.open(pth).resize((64, 64)))
            # except:
            #     print(pth)
        return ims

if __name__ == '__main__':
    generator.eval()
    generator.load_state_dict(torch.load('./models/generator/vae.pth'), False)
    test_11 = Test(generator)
    test_11.gen_data()
