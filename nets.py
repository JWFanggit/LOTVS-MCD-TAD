import os

import torch
from tensorboard import summary
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.ops import DeformConv2d
from torchvision.utils import save_image
from tqdm import tqdm

from ConvLSTM import ConvLSTM
from Diffusion import UNet, GaussianDiffusionSampler
# from Diffusion.Model import UNet
from configs.config import args
from Diffusion.dataProcessing import LoadData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear_en(nn.Module):
    def __init__(self):
        super(Linear_en, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),

            nn.Linear(128, 64),
            nn.Tanh(),

            nn.Linear(64, 32),
            nn.Tanh(),

            nn.Linear(32, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.encoder(x)
        return out

class Linear_en1(nn.Module):
    def __init__(self):
        super(Linear_en1, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),

            nn.Linear(128, 64),
            nn.Tanh(),

            nn.Linear(64, 32),
            nn.Tanh(),

            nn.Linear(32, 4),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.encoder(x)
        return out

class Linear_de(nn.Module):
    def __init__(self):
        super(Linear_de, self).__init__()

        self.device = device
        self.decoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),

            nn.Linear(32, 64),
            nn.Tanh(),

            nn.Linear(64, 128),
            nn.Tanh(),

            nn.Linear(128, 256),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.decoder(x)
        return out

# 生成未来帧, 可以先训练， 效果不好的话，可以考虑使用 EfficientNet 做特征提取
class MyGenerator(nn.Module):
    def __init__(self, args):
        super(MyGenerator, self).__init__()

        self.args = args
        self.convlstm = ConvLSTM(256, 256, (3, 3), 1, bias=False)  # 对光流进行计算
        self.Linear_en = Linear_en()
        self.Linear_de = Linear_de()
        self.Linear_en1 = Linear_en1()

        self.UNet = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)

        #  ------------------12 指的是 4张图  3通道×4 编码 解码过程 -----------------------------
        self.conv0_1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )  # 光流输入

        self.conv0_2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )  # RGB 图像输入

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2), )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2), )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2), )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(), )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(), )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(), )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(), )


        #  ------------------12 指的是 4张图  3通道×4 编码 解码过程 -----------------------------

    def forward(self, x, y, z):  # z 表示坐标框([N, T, M, 5]) M 表示一张图上有M个框
        '''
        x 表示图片， [B, 3*T, H, W]
        y 表示光流([B,T,h,w,2])
        '''

        # encode 过程 rgb

        x_0 = self.conv0_2(x)
        x_1 = self.conv1(x_0)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)

        # 光流 encode 过程
        y_1 = self.flow_encode(y)  # 光流的特征 [N, T, C, H, W]
        y_2 = self.convlstm(y_1)

        #扩散模块
        d_1 = x_3.permute(0, 2, 3, 1)
        d_1 = self.Linear_en(d_1)
        d_1 = d_1.permute(0, 3, 1, 2)

        self.Linear_en.load_state_dict(torch.load('./models/generator/encode.pth'), False)
        self.Linear_de.load_state_dict(torch.load('./models/generator/decode.pth'), False)
        model = UNet(T=1000, ch=128, ch_mult=[1, 2, 3, 4],
                         attn=[2],
                         num_res_blocks=2, dropout=0.1)
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        ckpt = torch.load(os.path.join(
                "./models/generator/", "DDPM.pt"),
                map_location=device)
        model.load_state_dict(ckpt, False)
        model.eval()
        sampler = GaussianDiffusionSampler(
                model, 1e-4, 0.02, 1000).to(device)
        # Sampled from standard normal distribution
        noisyImage = d_1.to(device)

        f = y_2[:, -1, :].permute(0, 2, 3, 1)
        f = self.Linear_en1(f)
        f = f.permute(0, 3, 1, 2)
        d_2 = sampler(noisyImage, f)

        d_2 = d_2.permute(0, 2, 3, 1)
        d_2 = d_2.float()
        d_2 = self.Linear_de(d_2)
        d_2 = d_2.permute(0, 3, 1, 2)


        # decode 过程 rgb
        x_4 = self.conv4(y_2[:, -1, :] + d_2)
        x_5 = self.conv5(x_2 +x_4)
        x_6 = self.conv6(x_1 +x_5)
        rgb_out = self.conv7(x_6)

        return rgb_out

    def flow_encode(self, inputs):
        t = inputs.shape[1]
        outs = []
        for i in range(t):
            out = self.conv0_1(inputs[:, i])
            out = self.conv1(out)
            out = self.conv2(out)
            out = self.conv3(out)
            outs.append(out)
        return torch.stack(outs, 1)

    # def encode(self,inputs):


if __name__ == '__main__':
    from configs.config import args

    x = torch.zeros([1, 12, 256, 256]).to(device)
    y = torch.rand([1, 4, 3, 256, 256]).to(device)

    net = MyGenerator(args).to(device)

    rr = net(x, y)
    print(rr[0].shape)
    summary(net, (8, 3))