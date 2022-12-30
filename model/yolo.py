import torch
import math
from torch import nn
import torch.nn.functional as F
from model.utils import *
from model.utils import CSPDarknet


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, image):
        # fpn output content features of [dark3, dark4, dark5]

        fpn_outs = self.backbone(image)

        outputs = self.head(fpn_outs)

        return outputs


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        params,
        in_dim,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        expand_ratio=0.5,
    ):
        super().__init__()
        fix_expand = params.fix_expand
        self.is_deploy = params.is_BNC

        self.backbone = CSPDarknet(in_dim, depth, width, depthwise=depthwise, act=act)

        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        if params.upsample_type == 'bilinear':  # SNPE 不支持 nearest
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        else:  # default
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            expansion=expand_ratio,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            expansion=1 if depthwise or fix_expand else expand_ratio,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            expansion=expand_ratio,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            expansion=expand_ratio,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        #  backbone
        out_features = self.backbone(input)
        [x2, x1, x0] = [out_features[f] for f in self.in_features]

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = pan_out0
        return outputs


class YOLOXHead(nn.Module):
    def __init__(
        self,
        params,
        num_classes,
        head_width=1.0,
        yolo_width=1.0,
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        super().__init__()
        out_ch = params.perturbed_points * 2

        self.seq = nn.Sequential(
            nn.Conv2d(int(in_channels[-1] * yolo_width), int(256 * head_width), 1),
            nn.BatchNorm2d(int(256 * head_width)),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(256 * head_width), 2 * int(256 * head_width), 1),
            nn.Conv2d(2 * int(256 * head_width), out_ch, 1),
            nn.Flatten(1),
            nn.Tanh(),
        )

    def forward(self, x):
        y = self.seq(x)
        return y


def get_model(params):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    backbone = YOLOPAFPN(
        params,
        params.in_dim,
        params.yolo_depth,
        params.yolo_width,
        in_channels=params.in_channels,
        act=params.yolo_act,
        depthwise=params.dwconv,
        expand_ratio=params.expand_ratio,
    )
    head = YOLOXHead(
        params,
        params.num_classes,
        in_channels=params.in_channels,
        head_width=params.head_width,
        yolo_width=params.yolo_width,
        act=params.yolo_act,
        depthwise=params.dwconv,
    )
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    return model
