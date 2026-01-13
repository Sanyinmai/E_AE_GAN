import torch.nn as nn
class MHSAWrapper(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8): # 定义8组QKV
        super(MHSAWrapper, self).__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, z):
        B, C, H, W = z.size()

        z = z.view(B, C, H * W)

        z = z.permute(2, 0, 1)

        # MHSA
        att_z, att_weight = self.mhsa(z, z, z)

        att_z = att_z.permute(1, 2, 0)

        att_z = att_z.view(B, C, H, W)

        return att_z, att_weight
