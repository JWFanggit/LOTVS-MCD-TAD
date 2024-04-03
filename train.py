import tqdm
from configs.config import args
from nets import MyGenerator
import numpy as np
import torch
from loss_function import intensity_loss, gradient_loss, psnr_error
from dataProcessing1 import LoadData
import glob
from torch.utils.data import DataLoader

# ---------------------------------------- start training ---------------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')

# Configure data loader


train_dataset = LoadData()
print('train data length:', train_dataset.__len__())
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=1)

val_dataset = LoadData(mode='val')
print('val data length:', val_dataset.__len__())
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1)

# Loss functions
adversarial_loss = torch.nn.BCELoss().to(device)
auxiliary_loss = torch.nn.CrossEntropyLoss().to(device)
mes_loss = torch.nn.MSELoss().to(device)
css_loss = torch.nn.CosineSimilarity().to(device)
# Initialize generator
generator = MyGenerator(args).to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr)

# ----------
#  Training
# ----------

if __name__ == '__main__':

    changes = 0
    print('start training ...')
    RESUME = True

    start_epoch = 2
    if RESUME:
        path_checkpoint = "./models/generator/epoch_2_psnr_14.66217.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        generator.load_state_dict(checkpoint['generator'])  # 加载模型可学习参数

        optimizer_G.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        # start_epoch = 4  # 设置开始的epoch

    for epoch in range(start_epoch + 1, args.n_epochs):
        # training ...
        generator.train().to(device)

        for step, (imgs, flow, gt_img) in enumerate(train_loader):
            gt_img = gt_img.to(device)
            imgs, flow = imgs.to(device), flow.to(device)
            gen_img = generator(imgs, flow).to(device)
            # 计算 图片生成的 psn r
            train_psnr_error = psnr_error(gen_frames=gen_img, gt_frames=gt_img).to(device)
            # ---------------------------------------training generator------------------------------------
            # define intensity loss
            if args.lam_lp != 0:
                lp_loss = (intensity_loss(gen_frames=gen_img, gt_frames=gt_img, l_num=2)).to(device)
            else:
                lp_loss = 0.

            # define gdl loss
            if args.lam_gdl != 0:
                gdl_loss = (gradient_loss(gen_frames=gen_img, gt_frames=gt_img, alpha=args.alpha_num)).to(device)
            else:
                gdl_loss = 0.

            g_loss = lp_loss * args.lam_lp + gdl_loss * args.lam_gdl
            all_loss = g_loss.item()
            # print(all_loss )

            if changes < 200:
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                changes += 1

            if (step + 1) % 10 == 0:
                print(
                    'epoch:[{}/{}] -> step:[{}/{}] -- train_psnr_error: {:.5f}, generator loss : {} '.format(
                        epoch, args.n_epochs, step, train_loader.__len__(),
                        train_psnr_error, g_loss.item()))

        # start val
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                generator.eval()
                mean_val_loss = 0
                val_psnr_error = 0

                count = 0

                for i, (imgs, flow, gt_img) in enumerate(val_loader):
                    count += 1
                    imgs, flow, gt_img = \
                        imgs.to(device), flow.to(device), gt_img.to(device)
                    with torch.no_grad():
                        gen_img = generator(imgs, flow)
                        val_psnr_error += float(psnr_error(gen_frames=gen_img, gt_frames=gt_img))
                val_psnr_error = val_psnr_error / (count + 1)
                # 保存模型参数
                # state_dict = generator.state_dict()
                # checkpoint = {
                #     'generator': generator.state_dict(),
                #     'optimizer': optimizer_G.state_dict(),
                #     "epoch": epoch
                # }
                torch.save(generator.state_dict(),
                           # './models/generator/epoch_{}_psnr_{:.5f}.pth'.format(epoch + 1, val_psnr_error))
                           './models/generator/epoch_{}_psnr_{:.5f}.pth'.format(epoch + 1, val_psnr_error))
