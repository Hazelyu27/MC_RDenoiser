import torch
import torch.nn as nn
from torch.nn import functional as func
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
__all__ = ['SMAPELoss','l1loss','SMAPELosstemp']

def LoG(img):
    weight = [
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
    ]
    weight = np.array(weight)

    weight_np = np.zeros((1, 1, 5, 5))
    weight_np[0, 0, :, :] = weight
    weight_np = np.repeat(weight_np, img.shape[1], axis=1)
    weight_np = np.repeat(weight_np, img.shape[0], axis=0)

    weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')

    return func.conv2d(img, weight, padding=1)

class SMAPELoss_kd(nn.Module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps

    def forward(self,student_outputs, teacher_outputs, targets):

        loss1 = torch.mean(torch.abs(student_outputs - targets) / (student_outputs.abs() + targets.abs() + self.eps))
        loss2 = torch.mean(torch.abs(student_outputs - teacher_outputs) / (student_outputs.abs() + teacher_outputs.abs() + self.eps))
        return loss1+loss2
class SMAPELoss(nn.Module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps

    def forward(self,outputs, targets):

        loss = torch.mean(torch.abs(outputs - targets) / (outputs.abs() + targets.abs() + self.eps))

        return loss
class SMAPELosstemp(nn.Module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps

    def forward(self,student_outputs, teacher_outputs, targets):
        meanloss = 0

        for i in range(len(targets)):

            so = student_outputs[:,i*3:(i+1)*3,:,:]
            to = teacher_outputs[:,i*3:(i+1)*3,:,:]
            tg = targets[i]

            tg = tg.to('cuda')
            # print(so.shape)
            # print(tg.shape)
            # print(to.shape)
            loss_1 = torch.mean(torch.abs(so - tg) / (so.abs() + tg.abs() + self.eps))
            loss_2 = torch.mean(torch.abs(so - to) / (so.abs() + to.abs() + self.eps))
            meanloss += (loss_1 + loss_2)
        meanloss = meanloss/len(targets)

        return meanloss

def HFEN(output, target):
    return torch.sum(torch.pow(LoG(output) - LoG(target), 2)) / torch.sum(torch.pow(LoG(target), 2))

class l1loss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, outputs, targets):
        l1loss = torch.sum(torch.abs(outputs - targets)) / torch.numel(outputs)
        HFENloss = HFEN(outputs, targets)
        return 0.8* l1loss + 0.2 * HFENloss

class Reeloss_kd(nn.Module):
    def __init__(self):
        super().__init__()

    # 知识蒸馏
    def forward(self, student_outputs, teacher_outputs, targets):
        l1loss = torch.sum(torch.abs(student_outputs - targets)) / torch.numel(student_outputs)
        HFENloss = HFEN(student_outputs, targets)
        loss1 = 0.8* l1loss + 0.2 * HFENloss
        l1loss = torch.sum(torch.abs(student_outputs - teacher_outputs)) / torch.numel(student_outputs)
        HFENloss = HFEN(student_outputs, teacher_outputs)
        loss2 = 0.8* l1loss + 0.2 * HFENloss
        return loss2 + loss1


def calculate_rae(image1, image2):
    # 将图像转换为numpy数组
    # image1 = np.array(image1)
    # image2 = np.array(image2)
    #
    # # 确保图像尺寸相同
    # assert image1.shape == image2.shape, "图像尺寸不匹配"

    # 计算RAE
    diff = np.abs(image1 - image2)
    rae = np.sum(diff) / np.sum(image1)

    return rae


class RAEloss_kd(nn.Module):
    def __init__(self):
        super(RAEloss_kd, self).__init__()

    def forward(self, student_outputs, teacher_outputs, targets):


        diff = torch.abs(student_outputs - targets)
        rae1 = torch.sum(diff) / torch.sum(targets)

        diff = torch.abs(teacher_outputs - targets)
        rae2 = torch.sum(diff) / torch.sum(targets)

        return rae1+rae2

class RAEloss(nn.Module):
    def __init__(self):
        super(RAEloss, self).__init__()

    def forward(self, outputs, targets):
        diff = torch.abs(outputs - targets)
        rae1 = torch.sum(diff) / torch.sum(targets)


        return rae1

class RAEloss_temp(nn.Module):
    def __init__(self):
        super(RAEloss_temp, self).__init__()

    def forward(self, outputs, targets):
        targets = targets.cuda()
        targets = targets[:,-1,:,:,:]
        diff = torch.abs(outputs - targets)
        rae1 = torch.sum(diff) / torch.sum(targets)


        return rae1
class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        # print(y.shape)
        # print(x.shape)
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean()



# class MS_SSIM_L1_LOSS_temp(nn.Module):
#     # Have to use cuda, otherwise the speed is too slow.
#     def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
#                  data_range = 1.0,
#                  K=(0.01, 0.03),
#                  alpha=0.025,
#                  compensation=200.0,
#                  cuda_dev=0,):
#         super(MS_SSIM_L1_LOSS_temp, self).__init__()
#         self.DR = data_range
#         self.C1 = (K[0] * data_range) ** 2
#         self.C2 = (K[1] * data_range) ** 2
#         self.pad = int(2 * gaussian_sigmas[-1])
#         self.alpha = alpha
#         self.compensation=compensation
#         filter_size = int(4 * gaussian_sigmas[-1] + 1)
#         g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
#         for idx, sigma in enumerate(gaussian_sigmas):
#             # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
#             g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
#             g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
#             g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
#         self.g_masks = g_masks.cuda(cuda_dev)
#
#     def _fspecial_gauss_1d(self, size, sigma):
#         """Create 1-D gauss kernel
#         Args:
#             size (int): the size of gauss kernel
#             sigma (float): sigma of normal distribution
#
#         Returns:
#             torch.Tensor: 1D kernel (size)
#         """
#         coords = torch.arange(size).to(dtype=torch.float)
#         coords -= size // 2
#         g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#         g /= g.sum()
#         return g.reshape(-1)
#
#     def _fspecial_gauss_2d(self, size, sigma):
#         """Create 2-D gauss kernel
#         Args:
#             size (int): the size of gauss kernel
#             sigma (float): sigma of normal distribution
#
#         Returns:
#             torch.Tensor: 2D kernel (size x size)
#         """
#         gaussian_vec = self._fspecial_gauss_1d(size, sigma)
#         return torch.outer(gaussian_vec, gaussian_vec)
#
#     def forward(self, x, y):
#         # x:output, y:target
#         y = y[:,-1,:,:,:]
#         print(y.shape)
#         print(x.shape)
#         b, c, h, w = x.shape
#         mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
#         muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)
#
#         mux2 = mux * mux
#         muy2 = muy * muy
#         muxy = mux * muy
#
#         sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
#         sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
#         sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy
#
#         # l(j), cs(j) in MS-SSIM
#         l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
#         cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
#
#         lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
#         PIcs = cs.prod(dim=1)
#
#         loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]
#
#         loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
#         # average l1 loss in 3 channels
#         gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
#                                groups=3, padding=self.pad).mean(1)  # [B, H, W]
#
#         loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
#         loss_mix = self.compensation*loss_mix
#
#         return loss_mix.mean()