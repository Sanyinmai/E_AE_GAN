import torch
import torch.nn as nn

class AdaptiveWeightedSkipConnection(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveWeightedSkipConnection, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, in_channels, 1, 1))
        nn.init.constant_(self.weight, 0.5)
        self.threshold = nn.Parameter(torch.Tensor([0.2]))  # 用于异常检测的阈值

    def forward(self, original_features, memory_output):
        feature_diff = torch.abs(original_features - memory_output)
        anomaly_mask = torch.maximum(torch.zeros_like(feature_diff), (feature_diff - self.threshold) / self.threshold)
        adaptive_weight = self.weight * (1 - anomaly_mask)
        fused_features = adaptive_weight * memory_output + (1 - adaptive_weight) * original_features
        return fused_features

class AdaptiveWeightedSkipConnection_Test(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveWeightedSkipConnection_Test, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, in_channels, 1, 1))
        nn.init.constant_(self.weight, 0.5)
        self.threshold = nn.Parameter(torch.Tensor([0.2]))  # 用于异常检测的阈值

    def forward(self, original_features, memory_output):
        feature_diff = torch.abs(original_features - memory_output)
        anomaly_mask = torch.maximum(torch.zeros_like(feature_diff), (feature_diff - self.threshold) / self.threshold)
        adaptive_weight = self.weight * (1 - anomaly_mask)
        fused_features = adaptive_weight * original_features + (1 - adaptive_weight) * memory_output
        return fused_features

class WeightedSkipConnection(nn.Module):
    def __init__(self, in_channels):
        super(WeightedSkipConnection, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, in_channels, 1, 1))
        nn.init.constant_(self.weight, 0.5)  # 初始化为0.5，表示初始时两者权重相等

    def forward(self, original_features, memory_output):
        fused_features = self.weight * original_features + (1 - self.weight) * memory_output
        return fused_features

