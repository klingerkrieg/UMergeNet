import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3


class MobileNetBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        mobilenet = mobilenet_v2(weights='DEFAULT')

        if in_channels != 3:
            first_conv = mobilenet.features[0][0]
            mobilenet.features[0][0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
        
        self.features = mobilenet.features
        self.out_channels = 1280

    def forward(self, x):
        x = self.features(x)
        return {"out": x}


def getDeepLabV3_MobileNetV2(num_classes, in_channels=3):
    backbone = MobileNetBackbone(in_channels=in_channels)
    return DeepLabV3(
        backbone=backbone,
        classifier=DeepLabHead(backbone.out_channels, num_classes),
        aux_classifier=None
    )


class DeepLabV3MobilenetV2Wrapper(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.model = getDeepLabV3_MobileNetV2(num_classes=out_channels, in_channels=in_channels)
    
    def forward(self, x):
        out = self.model(x)
        return out['out']