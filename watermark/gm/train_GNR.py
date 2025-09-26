# modify from the official implementation of GaussMarker. To check it, see https://github.com/SunnierLee/GaussMarker
# run python train_GNR.py --train_steps 50000 --r 180 --s_min 1.0 --s_max 1.2 --fp 0.35 --neg_p 0.5 --model_nf 128 --batch_size 32 --num_workers 16 --w_info_path w1_256.pth
# After training, the model weight will be saved as `./ckpts/model_final.pth`

import os
import argparse
import logging
from tqdm import tqdm
import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image

from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, nf=64, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, nf))
        self.down1 = (Down(nf, nf*2))
        self.down2 = (Down(nf*2, nf*4))
        self.down3 = (Down(nf*4, nf*8))
        factor = 2 if bilinear else 1
        # self.down4 = (Down(512, 1024 // factor))
        # self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(nf*8, nf*4 // factor, bilinear))
        self.up3 = (Up(nf*4, nf*2 // factor, bilinear))
        self.up4 = (Up(nf*2, nf, bilinear))
        self.outc = (OutConv(nf, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # print(x5.shape)
        # x = self.up1(x5, x4)
        # print(x4.shape)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class Gaussian_Shading_chacha:
    def __init__(self, ch_factor, w_factor, h_factor, fpr, user_number, watermark=None, key=None, nonce=None, m=None):
        self.ch = ch_factor
        self.w = w_factor
        self.h = h_factor
        self.nonce = nonce
        self.key = key
        self.watermark = watermark
        self.m = m
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.w * self.h)

        self.threshold = 1 if self.h == 1 and self.w == 1 and self.ch == 1 else self.ch * self.w * self.h // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z

    def create_watermark_and_return_w(self):
        if self.watermark is None:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.w, 64 // self.h])
            sd = self.watermark.repeat(1,self.ch,self.w,self.h)
            m = self.stream_key_encrypt(sd.flatten().numpy())
            self.m = torch.from_numpy(m).reshape(1, 4, 64, 64)
        w = self.truncSampling(self.m)
        return w
    
    # def create_watermark_and_return_w_sd(self):
    #     self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw])
    #     sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
    #     m = self.stream_key_encrypt(sd.flatten().numpy())
    #     w = self.truncSampling(m)
    #     return w, sd

    def create_watermark_and_return_w_m(self):
        if self.watermark is None:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.w, 64 // self.h])
            sd = self.watermark.repeat(1, self.ch, self.w, self.h)
            self.m = self.stream_key_encrypt(sd.flatten().numpy())
        w = self.truncSampling(self.m)
        return w, torch.from_numpy(self.m).reshape(1, 4, 64, 64)
    
    def stream_key_encrypt(self, sd):
        if self.key is None or self.nonce is None:
            self.key = get_random_bytes(32)
            self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        return sd_tensor
    
    # def stream_key_encrypt(self, sd):
    #     return sd

    # def stream_key_decrypt(self, reversed_m):
    #     return torch.from_numpy(reversed_m).reshape(1, 4, 64, 64).to(torch.uint8)

    def diffusion_inverse(self,watermark_r):
        ch_stride = 4 // self.ch
        w_stride = 64 // self.w
        h_stride = 64 // self.h
        ch_list = [ch_stride] * self.ch
        w_list = [w_stride] * self.w
        h_list = [h_stride] * self.h
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(w_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(h_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote
    
    def pred_m_from_latent(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        return reversed_m
    
    def pred_w_from_latent(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        return reversed_watermark
    
    def pred_w_from_m(self, reversed_m):
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        return reversed_watermark

def flip_tensor(tensor, flip_prob):
    random_tensor = torch.rand(tensor.size())
    flipped_tensor = tensor.clone()
    flipped_tensor[random_tensor < flip_prob] = 1 - flipped_tensor[random_tensor < flip_prob]
    return flipped_tensor

def Affine_random(latent, r, t, s_min, s_max, sh):
    config = dict(degrees=(-r, r), translate=(t, t), scale_ranges=(s_min, s_max), shears=(-sh, sh), img_size=latent.shape[-2:])
    r, (tx, ty), s, (shx, shy) = transforms.RandomAffine.get_params(**config)
    
    b, c, w, h = latent.shape
    new_latent = transforms.functional.affine(latent.view(b*c, 1, w, h), angle=r, translate=(tx, ty), scale=s, shear=(shx, shy), fill=999999)
    new_latent = new_latent.view(b, c, w, h)

    mask = (new_latent[:, :1, ...] < 999998).float()
    new_latent = new_latent * mask + torch.randint_like(new_latent, low=0, high=2) * (1-mask)
    
    return new_latent, (r, tx, ty, s)

class LatentDataset_m(IterableDataset):
    def __init__(self, watermark, args):
        super(LatentDataset_m, self).__init__()
        self.watermark = watermark
        self.args = args
        if self.args.num_watermarks > 1:
            t_m = torch.from_numpy(self.watermark.m).reshape(1, 4, 64, 64)
            o_m = torch.randint(low=0, high=2, size=(self.args.num_watermarks-1, 4, 64, 64))
            self.m = torch.cat([t_m, o_m])
        else:
            self.m = torch.from_numpy(self.watermark.m).reshape(1, 4, 64, 64)
        self.args.neg_p = 1 / (1 + self.args.num_watermarks)
    
    def __iter__(self):
        while True:
            random_index = torch.randint(0, self.args.num_watermarks, (1,)).item()
            latents_m = self.m[random_index:random_index+1]
            false_latents_m = torch.randint_like(latents_m, low=0, high=2)
            # latents_m = latents_m[:, :1, ...]
            # false_latents_m = false_latents_m[:, :1, ...]
            if np.random.rand() > self.args.neg_p:
                aug_latents_m, params = Affine_random(latents_m.float(), self.args.r, self.args.t, self.args.s_min, self.args.s_max, self.args.sh)
                aug_latents_m = flip_tensor(aug_latents_m, self.args.fp)
                yield aug_latents_m.squeeze(0).float(), latents_m.squeeze(0).float()
            else:
                aug_false_latents_m, params = Affine_random(false_latents_m.float(), self.args.r, self.args.t, self.args.s_min, self.args.s_max, self.args.sh)
                aug_false_latents_m = flip_tensor(aug_false_latents_m, self.args.fp)
                yield aug_false_latents_m.squeeze(0).float(), aug_false_latents_m.squeeze(0).float()


def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    gfile_stream = open(os.path.join(args.output_path, 'log.txt'), 'a')
    set_logger(gfile_stream)
    logging.info(args)

    num_steps = args.train_steps
    bs = args.batch_size

    model = UNet(4, 4, nf=args.model_nf).cuda()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters in model: %d' % n_params)
    logging.info('Number of trainable parameters in model: %d' % n_params)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if os.path.exists(args.w_info_path):
        w_info = torch.load(args.w_info_path)
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.w_copy, args.h_copy, args.fpr, args.user_number, watermark=w_info["w"], m=w_info["m"], key=w_info["key"], nonce=w_info["nonce"])
    else:
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.w_copy, args.h_copy, args.fpr, args.user_number)
        _ = watermark.create_watermark_and_return_w_m()
        torch.save({"w": watermark.watermark, "m": watermark.m, "key": watermark.key, "nonce": watermark.nonce}, args.w_info_path)

    if args.sample_type == "m":
        dataset = LatentDataset_m(watermark, args)
    else:
        raise NotImplementedError
    
    data_loader = DataLoader(dataset, batch_size=bs, num_workers=args.num_workers)

    for i, batch in tqdm(enumerate(data_loader)):
        x, y = batch
        # print(x[0, 0])
        x = x.cuda()
        y = y.cuda().float()

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % 2000 == 0:
        #     torch.save(model.state_dict(), os.path.join(args.output_path, "model_{}.pth".format(i)))
        if i % 2000 == 0:
            # torch.save(model.state_dict(), os.path.join(args.output_path, "model_{}.pth".format(i)))
            pred = F.sigmoid(pred)
            save_imgs = torch.cat([x[:, :1, ...].unsqueeze(0), pred[:, :1, ...].unsqueeze(0), y[:, :1, ...].unsqueeze(0)]).permute(1, 0, 2, 3, 4).contiguous()
            save_imgs = save_imgs.view(-1, save_imgs.shape[2], save_imgs.shape[3], save_imgs.shape[4])[:64]
            save_image(save_imgs, os.path.join(args.output_path, "sample_{}.png".format(i)), nrow=6)
        if i % 200 == 0:
            print(loss.item())
            torch.save(model.state_dict(), os.path.join(args.output_path, "checkpoint.pth".format(i)))
            logging.info("Iter {} Loss {}".format(i, loss.item()))

        if i > num_steps:
            break
    
    torch.save(model.state_dict(), os.path.join(args.output_path, "model_final.pth"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--w_copy', default=8, type=int)
    parser.add_argument('--h_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--output_path', default='./ckpts')
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--w_info_path', default='./w1.pth')

    parser.add_argument('--train_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sample_type', default="m")
    parser.add_argument('--r', type=float, default=8)
    parser.add_argument('--t', type=float, default=0)
    parser.add_argument('--s_min', type=float, default=0.5)
    parser.add_argument('--s_max', type=float, default=2.0)
    parser.add_argument('--sh', type=float, default=0)
    parser.add_argument('--fp', type=float, default=0.00)
    parser.add_argument('--neg_p', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_watermarks', type=int, default=1)

    parser.add_argument('--model_nf', type=int, default=64)
    parser.add_argument('--exp_description', '-ed', default="")

    args = parser.parse_args()

    # multiprocessing.set_start_method("spawn")
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # args.output_path = args.output_path + 'r{}_t{}_s_{}_{}_sh{}_fp{}_np{}_{}_{}'.format(args.r, args.t, args.s_min, args.s_max, args.sh, args.fp, args.neg_p, args.exp_description, nowTime)
    args.output_path = args.output_path + '_' + args.exp_description

    main(args)
