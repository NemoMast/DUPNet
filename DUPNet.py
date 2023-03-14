# -*- coding: utf-8 -*-
"""
Created on 2021/4/5 14:26

@author: JiaHao Du
"""
import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F



def upsample(x, h, w):
    # return F.interpolate(x, size=[h,w], mode='bilinear', align_corners=True)
    return F.interpolate(x, size=[h,w], mode='bicubic', align_corners=True)

class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels,out_channels, 3, padding=1, bias=False)
        self.relu  = nn.ReLU(True)

    def forward(self, x):
        x = x+self.conv2(self.relu(self.conv1(x)))
        return x

class BasicUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(BasicUnit, self).__init__()
        p = kernel_size//2
        self.basic_unit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.ReLU(True),
            #nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=p, bias=False),
            #nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False)
            )

    def forward(self, input):
        return self.basic_unit(input)

# LRBlock is called MSBlock in our paper
class LRBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 n_feat):
        super(LRBlock, self).__init__()
        self.get_N1 = BasicUnit(ms_channels, n_feat, ms_channels)
        self.get_N2 = BasicUnit(ms_channels, n_feat, ms_channels)
        self.prox_N = BasicUnit(ms_channels, n_feat, ms_channels)

    def forward(self, MS, lrms, Dt, Ct):
        _,_,M,N = MS.shape
        _,_,m,n = lrms.shape
        HR = self.get_N1(MS + Dt)                   # 获得HRMS
        LR_hat = upsample(HR, m, n)                 # 下采�?�?
        LR_Residual = self.get_N2(LR_hat - lrms)    # 获得残差�?
        R = upsample(LR_Residual, M, N)             # 获得数据保真度R
        Dt = self.prox_N(Dt - R - Ct)                    # 更新Dt
        return Dt, LR_Residual

class PANBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat,
                 kernel_size):
        super(PANBlock, self).__init__()
        self.get_N3 = BasicUnit(ms_channels, n_feat, ms_channels, kernel_size)
        self.get_N4 = BasicUnit(pan_channels, n_feat, pan_channels, kernel_size)
        self.get_C = BasicUnit(ms_channels, n_feat, pan_channels, kernel_size)  # 线型组合
        self.get_CT = BasicUnit(pan_channels, n_feat, ms_channels, kernel_size) # 线型组合转置
        self.prox_N = BasicUnit(ms_channels, n_feat, ms_channels, kernel_size)

    def forward(self, MS, PAN, Dt, Ct):
        # HR = self.get_N3(MS + Dt)   # 获得HRMS
        HR = MS + Dt
        PAN_hat = self.get_C(HR)  # 对HRMS线型组合
        # PAN_Residual = self.get_N4(PAN - PAN_hat)    # 获得残差�?
        PAN_Residual = PAN - PAN_hat
        R = self.get_CT(PAN_Residual)
        Dt = self.prox_N(Dt - R + Ct)
        return Dt, PAN_Residual

class PMFNet4(nn.Module):
    def __init__(self,
                 dataset='wv2',
                 n_feat=128,
                 n_layer=8):
        super(PMFNet4, self).__init__()
        ms_channels = 4 if dataset in ['qb', 'gf2'] else 8
        self.ms_channels = ms_channels
        pan_channels = 1
        self.lr_blocks = nn.ModuleList([LRBlock(ms_channels, n_feat) for i in range(n_layer)])
        self.pan_blocks = nn.ModuleList([PANBlock(ms_channels, pan_channels, n_feat, 1) for i in range(n_layer)])
        
        self.eta = nn.Conv2d(ms_channels, ms_channels, kernel_size=3, stride=1, padding=1, groups=ms_channels)
        print('hello', n_layer)

    def forward(self, *inputs):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        lrms, pan = inputs[0], inputs[1]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = lrms.shape
        _, _, M, N = pan.shape
        MS = upsample(lrms, M, N)
        Dt = torch.zeros(16, self.ms_channels, 128, 128).to('cuda')
        Zt = torch.zeros(16, self.ms_channels, 128, 128).to('cuda')
        Wt = torch.zeros(16, self.ms_channels, 128, 128).to('cuda')

        E1_ls, E2_ls, stage_ls = list(), list(), list()
        for i in range(len(self.lr_blocks)):
            Wt = Dt - Zt + Wt 
            
            Ct = self.eta(Dt - Zt + Wt)
            Dt, E1 = self.lr_blocks[i](MS, lrms, Dt, Ct)

            Ct = self.eta(Dt - Zt + Wt)
            Zt, E2 = self.pan_blocks[i](MS, pan, Zt, Ct)

            E1_ls.append(E1)
            E2_ls.append(E2)
            stage_ls.append(MS + 0.5*(Dt+Zt))
        HR = MS + 0.5*(Dt+Zt)
        E1 = reduce(lambda x,y:x+y, E1_ls)/len(E1_ls)
        E2 = reduce(lambda x,y:x+y, E2_ls)/len(E2_ls)
        return HR, E1, E2, stage_ls

if __name__ == '__main__':
    # Z, Y
    inputs = [
        torch.randn(16, 8, 32, 32),  # LRMS
        torch.randn(16, 1, 128, 128),  # PAN
        torch.randn(16, 8, 128, 128)
    ]
    # X = torch.randn(8, 8, 128, 128)
    # Y = torch.randn(8, 8, 1, 1)
    # Z = torch.mul(X, Y)
    # print(Z.shape)
    # out_dim�?闁捐绉撮幃搴ㄥ炊閹冨壖闁哄牆顦ˇ璺ㄤ焊閹存繄婀?
    # upRank = out_dim - 1
    model = PMFNet4()
    out = model(*inputs)
    # print(model)
    print(out[0].shape)
