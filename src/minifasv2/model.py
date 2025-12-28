import torch
from torch import nn
import torch.nn.functional as F


class MultiFTNet(nn.Module):
    def __init__(
        self, num_channels=3, num_classes=2, embedding_size=128, conv6_kernel=(5, 5)
    ):
        super(MultiFTNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.model = MiniFASNetV2SE(
            embedding_size=embedding_size,
            conv6_kernel=conv6_kernel,
            num_classes=num_classes,
            num_channels=num_channels,
        )
        self.FTGenerator = FTGenerator(in_channels=128)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x1 = self.model.dropout(x1)
        classifier_output = self.model.logits(x1)

        if self.training:
            fourier_transform = self.FTGenerator(x)
            return classifier_output, fourier_transform
        else:
            return classifier_output


class FTGenerator(nn.Module):
    def __init__(self, in_channels=48, out_channels=1):
        super(FTGenerator, self).__init__()

        self.fourier_transform = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fourier_transform(x)


class L2Norm(nn.Module):
    def forward(self, input):
        return F.normalize(input)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        groups=1,
    ):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        groups=1,
    ):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        c3,
        residual=False,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
    ):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(
            c1_in, out_channels=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.conv_dw = Conv_block(
            c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride
        )
        self.project = Linear_block(
            c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        c3,
        num_block,
        groups,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    ):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(
                Depth_Wise(
                    c1_tuple,
                    c2_tuple,
                    c3_tuple,
                    residual=True,
                    kernel=kernel,
                    padding=padding,
                    stride=stride,
                    groups=groups,
                )
            )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return module_input * x


class ResidualSE(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        c3,
        num_block,
        groups,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        se_reduction=4,
    ):
        super(ResidualSE, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            if i == num_block - 1:
                modules.append(
                    Depth_Wise_SE(
                        c1_tuple,
                        c2_tuple,
                        c3_tuple,
                        residual=True,
                        kernel=kernel,
                        padding=padding,
                        stride=stride,
                        groups=groups,
                        se_reduction=se_reduction,
                    )
                )
            else:
                modules.append(
                    Depth_Wise(
                        c1_tuple,
                        c2_tuple,
                        c3_tuple,
                        residual=True,
                        kernel=kernel,
                        padding=padding,
                        stride=stride,
                        groups=groups,
                    )
                )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Depth_Wise_SE(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        c3,
        residual=False,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
        se_reduction=8,
    ):
        super(Depth_Wise_SE, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(
            c1_in, out_channels=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.conv_dw = Conv_block(
            c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride
        )
        self.project = Linear_block(
            c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.residual = residual
        self.se_module = SEModule(c3_out, se_reduction)

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = self.se_module(x)
            output = short_cut + x
        else:
            output = x
        return output


class MiniFASNet(nn.Module):
    def __init__(
        self,
        channel_config,
        embedding_size,
        conv6_kernel=(7, 7),
        dropout_prob=0.0,
        num_classes=2,
        num_channels=3,
    ):
        super(MiniFASNet, self).__init__()
        self.embedding_size = embedding_size

        self.conv1 = Conv_block(
            num_channels,
            channel_config[0],
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv2_dw = Conv_block(
            channel_config[0],
            channel_config[1],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=channel_config[1],
        )

        c1 = [(channel_config[1], channel_config[2])]
        c2 = [(channel_config[2], channel_config[3])]
        c3 = [(channel_config[3], channel_config[4])]

        self.conv_23 = Depth_Wise(
            c1[0],
            c2[0],
            c3[0],
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=channel_config[3],
        )

        c1 = [
            (channel_config[4], channel_config[5]),
            (channel_config[7], channel_config[8]),
            (channel_config[10], channel_config[11]),
            (channel_config[13], channel_config[14]),
        ]
        c2 = [
            (channel_config[5], channel_config[6]),
            (channel_config[8], channel_config[9]),
            (channel_config[11], channel_config[12]),
            (channel_config[14], channel_config[15]),
        ]
        c3 = [
            (channel_config[6], channel_config[7]),
            (channel_config[9], channel_config[10]),
            (channel_config[12], channel_config[13]),
            (channel_config[15], channel_config[16]),
        ]

        self.conv_3 = Residual(
            c1,
            c2,
            c3,
            num_block=4,
            groups=channel_config[4],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        c1 = [(channel_config[16], channel_config[17])]
        c2 = [(channel_config[17], channel_config[18])]
        c3 = [(channel_config[18], channel_config[19])]

        self.conv_34 = Depth_Wise(
            c1[0],
            c2[0],
            c3[0],
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=channel_config[19],
        )

        c1 = [
            (channel_config[19], channel_config[20]),
            (channel_config[22], channel_config[23]),
            (channel_config[25], channel_config[26]),
            (channel_config[28], channel_config[29]),
            (channel_config[31], channel_config[32]),
            (channel_config[34], channel_config[35]),
        ]
        c2 = [
            (channel_config[20], channel_config[21]),
            (channel_config[23], channel_config[24]),
            (channel_config[26], channel_config[27]),
            (channel_config[29], channel_config[30]),
            (channel_config[32], channel_config[33]),
            (channel_config[35], channel_config[36]),
        ]
        c3 = [
            (channel_config[21], channel_config[22]),
            (channel_config[24], channel_config[25]),
            (channel_config[27], channel_config[28]),
            (channel_config[30], channel_config[31]),
            (channel_config[33], channel_config[34]),
            (channel_config[36], channel_config[37]),
        ]

        self.conv_4 = Residual(
            c1,
            c2,
            c3,
            num_block=6,
            groups=channel_config[19],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        c1 = [(channel_config[37], channel_config[38])]
        c2 = [(channel_config[38], channel_config[39])]
        c3 = [(channel_config[39], channel_config[40])]

        self.conv_45 = Depth_Wise(
            c1[0],
            c2[0],
            c3[0],
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=channel_config[40],
        )

        c1 = [
            (channel_config[40], channel_config[41]),
            (channel_config[43], channel_config[44]),
        ]
        c2 = [
            (channel_config[41], channel_config[42]),
            (channel_config[44], channel_config[45]),
        ]
        c3 = [
            (channel_config[42], channel_config[43]),
            (channel_config[45], channel_config[46]),
        ]

        self.conv_5 = Residual(
            c1,
            c2,
            c3,
            num_block=2,
            groups=channel_config[40],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv_6_sep = Conv_block(
            channel_config[46],
            channel_config[47],
            kernel=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.conv_6_dw = Linear_block(
            channel_config[47],
            channel_config[48],
            groups=channel_config[48],
            kernel=conv6_kernel,
            stride=(1, 1),
            padding=(0, 0),
        )
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.logits = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.dropout(out)
        out = self.logits(out)
        return out


class MiniFASNetSE(MiniFASNet):
    def __init__(
        self,
        channel_config,
        embedding_size,
        conv6_kernel=(7, 7),
        dropout_prob=0.75,
        num_classes=2,
        num_channels=3,
    ):
        super(MiniFASNetSE, self).__init__(
            channel_config=channel_config,
            embedding_size=embedding_size,
            conv6_kernel=conv6_kernel,
            dropout_prob=dropout_prob,
            num_classes=num_classes,
            num_channels=num_channels,
        )

        c1 = [
            (channel_config[4], channel_config[5]),
            (channel_config[7], channel_config[8]),
            (channel_config[10], channel_config[11]),
            (channel_config[13], channel_config[14]),
        ]
        c2 = [
            (channel_config[5], channel_config[6]),
            (channel_config[8], channel_config[9]),
            (channel_config[11], channel_config[12]),
            (channel_config[14], channel_config[15]),
        ]
        c3 = [
            (channel_config[6], channel_config[7]),
            (channel_config[9], channel_config[10]),
            (channel_config[12], channel_config[13]),
            (channel_config[15], channel_config[16]),
        ]

        self.conv_3 = ResidualSE(
            c1,
            c2,
            c3,
            num_block=4,
            groups=channel_config[4],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        c1 = [
            (channel_config[19], channel_config[20]),
            (channel_config[22], channel_config[23]),
            (channel_config[25], channel_config[26]),
            (channel_config[28], channel_config[29]),
            (channel_config[31], channel_config[32]),
            (channel_config[34], channel_config[35]),
        ]
        c2 = [
            (channel_config[20], channel_config[21]),
            (channel_config[23], channel_config[24]),
            (channel_config[26], channel_config[27]),
            (channel_config[29], channel_config[30]),
            (channel_config[32], channel_config[33]),
            (channel_config[35], channel_config[36]),
        ]
        c3 = [
            (channel_config[21], channel_config[22]),
            (channel_config[24], channel_config[25]),
            (channel_config[27], channel_config[28]),
            (channel_config[30], channel_config[31]),
            (channel_config[33], channel_config[34]),
            (channel_config[36], channel_config[37]),
        ]

        self.conv_4 = ResidualSE(
            c1,
            c2,
            c3,
            num_block=6,
            groups=channel_config[19],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        c1 = [
            (channel_config[40], channel_config[41]),
            (channel_config[43], channel_config[44]),
        ]
        c2 = [
            (channel_config[41], channel_config[42]),
            (channel_config[44], channel_config[45]),
        ]
        c3 = [
            (channel_config[42], channel_config[43]),
            (channel_config[45], channel_config[46]),
        ]
        self.conv_5 = ResidualSE(
            c1,
            c2,
            c3,
            num_block=2,
            groups=channel_config[40],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )


channel_config_dict = {
    "1.8M_": [
        32,
        32,
        103,
        103,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        231,
        231,
        128,
        231,
        231,
        128,
        52,
        52,
        128,
        26,
        26,
        128,
        77,
        77,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        308,
        308,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        512,
        512,
    ]
}


def MiniFASNetV2SE(
    embedding_size=128,
    conv6_kernel=(7, 7),
    dropout_prob=0.75,
    num_classes=2,
    num_channels=3,
):
    return MiniFASNetSE(
        channel_config_dict["1.8M_"],
        embedding_size,
        conv6_kernel,
        dropout_prob,
        num_classes,
        num_channels,
    )
