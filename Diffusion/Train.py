import os
from typing import Dict

import numpy
import numpy as np
import skimage
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from Diffusion.Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Diffusion.dataProcessing import LoadData
# from Diffusion.nets import encoder, Linear_en
from Scheduler import GradualWarmupScheduler
from diffusers import DDIMScheduler


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])

    # dataset = CIFAR10( root='./CIFAR10', train=True, download=True, transform=transforms.Compose([
    # transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,
    # 0.5)), ])) dataloader = DataLoader( dataset, batch_size=modelConfig["batch_size"], shuffle=True,
    # num_workers=4, drop_last=True, pin_memory=True)
    # dataset = ImageFolder(
    #     root='E:\\cxy\\datasets\\hevi\\hevi_val\\image\\',
    #     transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]))
    dataset = LoadData()
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True,
        pin_memory=False)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.Adam(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for input_image, input_flow in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                input_image = input_image.to(device)
                x_0 = input_image.to(device)
                flow = input_flow.to(device)
                # trainer = GaussianDiffusionTrainer(
                #     net_model(input_image, t, flow),modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
                loss = trainer(x_0, flow).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


from PIL import Image
from einops import rearrange



device = torch.device("cuda:0")
# Linear_en = Linear_en().to(device)
# encoder = encoder().to(device)

def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        dataset = LoadData()
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True,
            pin_memory=False)
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt, False)
        print("model load weight done.")
        model.eval()
        # flowencoder.eval()
        # flowencoder.load_state_dict(torch.load('E:/cxy/xuexi/models/generator/vae.pth'), False)
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for input_image, input_flow, target_image in tqdmDataLoader:
                flow = input_flow.to(device)
                noisyImage = input_image.to(device)
                fake_outputs = target_image.to(device)
                # noisyImage = torch.randn(
                #     size=[modelConfig["batch_size"], 3, 64, 64], device=device)
                saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
                save_image(saveNoisy, os.path.join(
                    modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
                sampledImgs = sampler(noisyImage, flow)
                sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
                save_image(sampledImgs, os.path.join(
                    modelConfig["sampled_dir"], modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
                # print(float(ssim(sampledImgs, fake_outputs)))







