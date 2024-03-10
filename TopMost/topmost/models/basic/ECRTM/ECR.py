import torch
from torch import nn


class ECR(nn.Module):
    '''
        Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023

        Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.
    '''
    def __init__(self, weight_loss_ECR, sinkhorn_alpha, OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_ECR = weight_loss_ECR
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def forward(self, M):  # M:Cjk = ∥wj − tk∥2
        # M: KxV
        # a: Kx1
        # b: Vx1
        device = M.device

        # Sinkhorn's algorithm  计算两个分布之间的最优输运方案
        # 使得所有元素的和为 1，表示每个元素的初始概率。
        # 这样得到的向量即为一个均匀分布的概率向量，用于初始化 Sinkhorn 算法中的概率向量。
        a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device) # 将一维张量转换为二维张量，形状为 (M.shape[0], 1)
        b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)

        u = (torch.ones_like(a) / a.size()[0]).to(device) # Kx1

        K = torch.exp(-M * self.sinkhorn_alpha)
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:   # 处理理解如果err打出提前错误或者小于max循环
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        transp = u * (K * v.T)

        loss_ECR = torch.sum(transp * M)
        loss_ECR *= self.weight_loss_ECR

        return loss_ECR
