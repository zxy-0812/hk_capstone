from .modeling import deeplabv3plus_resnet101
import torch
from torch import nn
from torch.nn import functional as F

class PredictRegressionModel(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # 初始化 DeepLabV3+ 模型
        self.deeplabv3plus = deeplabv3plus_resnet101(
            num_classes=num_classes,
            output_stride=16,
            pretrained_backbone=True  # 可选是否加载 ImageNet 预训练权重
        )

        # 获取 backbone 和 decoder
        self.feature_extractor = self.deeplabv3plus.backbone  # ResNet-101 + IntermediateLayerGetter
        self.decoder = self.deeplabv3plus.classifier  # DeepLabHeadV3Plus

        # 确定融合后的通道数
        self.fused_dim = 256 + 48  # layer1 (256 -> project to 48) + ASPP 输出 (256)

        # 自定义回归层
        self.regression_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 压缩空间维度
            nn.Flatten(),
            nn.Linear(self.fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # 输出单个预测值
        )

    def forward(self, x):
        # 提取多级特征
        features = self.feature_extractor(x)
        low_level_features = features['low_level']  # layer1 输出 [B, 256, H', W']
        high_level_features = features['out']       # layer4 输出 [B, 2048, H, W]

        # 使用 decoder 的前向函数提取融合特征（不经过 classifier）
        low_level_projected = self.decoder.project(low_level_features)
        aspp_output = self.decoder.aspp(high_level_features)

        # 上采样对齐尺寸
        aspp_output = F.interpolate(aspp_output,
                                    size=low_level_projected.shape[2:],
                                    mode='bilinear',
                                    align_corners=False)

        # 特征拼接
        fused_features = torch.cat([low_level_projected, aspp_output], dim=1)  # [B, 304, H', W']

        # 回归预测
        return self.regression_layer(fused_features).squeeze()