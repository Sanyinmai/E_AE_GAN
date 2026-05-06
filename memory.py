import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


def hard_shrink_relu(x, lambd=0, epsilon=1e-12):
    ''' Hard Shrinking '''
    return (F.relu(x - lambd) * x) / (torch.abs(x - lambd) + epsilon)


class MemoryModule(nn.Module):
    ''' Memory Module '''

    def __init__(self, mem_dim, fea_dim, shrink_thres, entropy_weight=0.002):
        super().__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        # attention
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # [M, C]
        self.shrink_thres = shrink_thres
        self.entropy_weight = entropy_weight
        self.reset_parameters()
        self.cosine_similarity = nn.CosineSimilarity(dim=2,)
        self.gate = nn.Conv2d(1024, 512, 1)
        nn.init.normal_(self.gate.weight, 0, 0.01)
        nn.init.constant_(self.gate.bias, 2)

    def reset_parameters(self):
        ''' init memory elements : Very Important !! '''
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.weight)
    
    def calculate_entropy_loss(self, att_weight):
        ''' 计算熵损失
        Args:
            att_weight: [N, M] 注意力权重（N=BxHxW, M=mem_dim）
        Returns:
            entropy_loss: 标量，熵损失值
        '''
        epsilon = 1e-8  
        entropy = -torch.mean(att_weight * torch.log(att_weight + epsilon))
        
        sparse_w_loss = self.entropy_weight * entropy
        return sparse_w_loss

    def forward(self, x):
        loss = torch.nn.TripletMarginLoss(margin=1.0)
        loss_mse = torch.nn.MSELoss()
        ''' x [B,C,H,W] : latent code Z'''
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).flatten(end_dim=2)  # Fea : [NxC]  N=BxHxW
        # calculate attention weight
        x_nor = F.normalize(x, dim=1)
        weight_nor = F.normalize(self.weight, dim=1)
        att_weight = F.linear(x_nor, weight_nor)  # Fea*Mem^T : [NxC] x [CxM] = [N, M]

        # score_query = F.softmax(att_weight, dim=0)
        score_memory = F.softmax(att_weight, dim=1)
        att_weight = F.softmax(att_weight, dim=1)  # [N, M]

        _, gathering_indices = torch.topk(score_memory, 2, dim=1)
        pos = weight_nor[gathering_indices[:, 0]]
        neg = weight_nor[gathering_indices[:, 1]]
        top1_loss = loss_mse(x, pos)
        gathering_loss = loss(x, pos, neg)
        # top1_loss = loss_mse(x, pos.detach())
        # gathering_loss = loss(x, pos.detach(), neg.detach())

        if self.shrink_thres > 0:
            # hard shrink
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1) 
        #     # 和上面一行换：att_weight = F.normalize(att_weight, dim=1)  # [N, M]
        else:
            att_weight = att_weight

        sparse_w_loss = self.calculate_entropy_loss(att_weight)

        # generate code z'
        mem_T = self.weight.permute(1, 0)
        output = F.linear(att_weight, mem_T)  # Fea*Mem^T^T : [N, M] x [M, C] = [N, C]
        #直接concat
        output = output.view(B, H, W, C).permute(0, 3, 1, 2)
        x_feature = x_nor.view(B, H, W, C).permute(0, 3, 1, 2)
        output = torch.cat((output, x_feature), dim=1)
        return att_weight, output, top1_loss, gathering_loss, sparse_w_loss

        #concat+fusion
        # output = output.view(B, H, W, C).permute(0, 3, 1, 2)
        # x_feature = x_nor.view(B, H, W, C).permute(0, 3, 1, 2)
        # cat_feature = torch.cat((output, x_feature), dim=1)
        # output = self.gate(cat_feature)
        # return att_weight, output, top1_loss, gathering_loss, sparse_w_loss

        
        #gate fusion
        # output = output.view(B, H, W, C).permute(0, 3, 1, 2)
        # x_feature = x_nor.view(B, H, W, C).permute(0, 3, 1, 2)
        # alpha = torch.sigmoid(self.gate(torch.cat((output, x_feature), dim=1)))
        # output = alpha * output + x_feature
        
        # # return att_weight, output, top1_loss, gathering_loss, alpha
        # return att_weight, output, top1_loss, gathering_loss, sparse_w_loss, alpha