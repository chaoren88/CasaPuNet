import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU(0.2, True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Beta(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Beta, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du1 = nn.Sequential(
            BasicConv(channel, channel // 16, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True),

            nn.ReLU(inplace=True),
            BasicConv(channel // reduction, channel, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True),
            nn.Sigmoid()
        )
        self.conv_du2 = nn.Sequential(
            BasicConv(channel, channel // 16, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True),

            nn.ReLU(inplace=True),
            BasicConv(channel // reduction, channel, 1, stride=1, padding=(1 - 1) // 2, relu=False, bias=True),
        )

    def forward(self, x):
        y = []
        for i in x:
            y.append(self.avg_pool(i).squeeze(dim=3))
        z = torch.cat(y,2)
        weight = torch.sum(nn.Softmax(dim=2)(z)*z, dim=2).unsqueeze(dim=2).unsqueeze(dim=3)
        beta = self.conv_du1(weight)
        gamma = self.conv_du2(weight)
        U_list = []
        for i in x:
            U_list.append(beta*i+gamma)
        return U_list


class line(nn.Module):

    def __init__(self):
        super(line, self).__init__()
        self.delta = nn.Parameter(torch.randn(1, 1))

    def forward(self, x, y):
        return torch.mul((1 - self.delta), x) + torch.mul(self.delta, y)


class ConvLayer1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer1, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, stride=stride)

        nn.init.xavier_normal_(self.conv2d.weight.data)

    def forward(self, x):
        return self.conv2d(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        reflection_padding = padding
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, groups=8, dilation=dilation),
            nn.ReLU(inplace=True)
        )
        nn.init.xavier_normal_(self.op[0].weight.data)

    def forward(self, x):
        return self.op(x)



Operations = [
    'dil_2_conv_3x3',
    'dil_4_conv_3x3',
    'dil_8_conv_3x3',
]

OPS = {
    'dil_2_conv_3x3': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 2, 2, affine=affine),
    'dil_4_conv_3x3': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 4, 4, affine=affine),
    'dil_8_conv_3x3': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 8, 8, affine=affine),
}


class muti_scale(nn.Module):
    def __init__(self, C_in, C_out, stride=1):
        super(muti_scale, self).__init__()
        self._ops = nn.ModuleList()
        for o in Operations:
            op = OPS[o](C_in, C_out, stride, False)
            self._ops.append(op)

        self._out = nn.Sequential(nn.Conv2d(C_out * len(Operations) + C_in, C_in, 1, stride=1), nn.ReLU(inplace=True))
        nn.init.kaiming_normal_(self._out[0].weight.data, a=0, mode='fan_in')

    def forward(self, x):
        _input = x
        states = []
        for i in range(len(Operations)):
            states.append(self._ops[i](x))
        states.append(_input)

        return self._out(torch.cat(states[:], dim=1)) + _input


class Decoding_block2(nn.Module):
    def __init__(self, base_filter, n_convblock):
        super(Decoding_block2, self).__init__()
        self.n_convblock = n_convblock
        self.upsample = upsample1(base_filter)
        modules_body = []
        for i in range(self.n_convblock - 1):
            modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, y):
        x = self.upsample(x, y)
        for i in range(self.n_convblock):
            x = self.body[i](x)
        return x


class Decoding_block1(nn.Module):
    def __init__(self, base_filter, n_convblock):
        super(Decoding_block1, self).__init__()
        self.n_convblock = n_convblock
        self.upsample = upsample2(base_filter)
        modules_body = []
        for i in range(self.n_convblock - 1):
            modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, y, z):
        y = self.upsample(x, y, z)
        for i in range(self.n_convblock):
            y = self.body[i](y)
        return y


class upsample2(nn.Module):
    def __init__(self, base_filter):
        super(upsample2, self).__init__()
        self.conv1 = ConvLayer(base_filter, base_filter, 3, stride=1)
        self.down = ConvLayer(base_filter, base_filter, 3, stride=2)
        self.ConvTranspose = UpsampleConvLayer(base_filter, base_filter, kernel_size=3, stride=1, upsample=2)
        self.cat = ConvLayer(base_filter * 3, base_filter, kernel_size=1, stride=1)

    def forward(self, x, y, z):
        state = []
        state.append(self.down(x))
        state.append(self.conv1(y))
        # state.append(y)
        state.append(self.ConvTranspose(z))
        return self.cat(torch.cat(state, dim=1))


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.conv2d = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, scale_factor=self.upsample)

        out = self.conv2d(x_in)
        return out


class upsample1(nn.Module):
    def __init__(self, base_filter):
        super(upsample1, self).__init__()
        self.conv1 = ConvLayer(base_filter, base_filter, 3, stride=1)
        self.ConvTranspose = UpsampleConvLayer(base_filter, base_filter, kernel_size=3, stride=1, upsample=2)
        self.cat = ConvLayer(base_filter * 2, base_filter, kernel_size=1, stride=1)

    def forward(self, x, y):
        y = self.ConvTranspose(y)
        x = self.conv1(x)

        return self.cat(torch.cat((x, y), dim=1))


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True)
        )

        nn.init.xavier_normal_(self.block[0].weight.data)

    def forward(self, x):
        return self.block(x)


class Encoding_block(nn.Module):
    def __init__(self, base_filter, n_convblock):
        super(Encoding_block, self).__init__()
        self.n_convblock = n_convblock
        modules_body = []
        for i in range(self.n_convblock - 1):
            modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=2))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        for i in range(self.n_convblock - 1):
            x = self.body[i](x)
        ecode = x
        x = self.body[self.n_convblock - 1](x)
        return ecode, x


class ziwangluo(nn.Module):
    def __init__(self, base_filter, n_convblock):
        super(ziwangluo, self).__init__()
        self.e1 = Encoding_block(base_filter, n_convblock)
        self.e2 = Encoding_block(base_filter, n_convblock)
        self.e3 = Encoding_block(base_filter, n_convblock)
        self.e4 = Encoding_block(base_filter, n_convblock)
        self.upsample1 = upsample1(base_filter)
        self.upsample2 = upsample2(base_filter)
        self.upsample3 = upsample2(base_filter)
        self.upsample4 = upsample1(base_filter)
        self.upsample5 = upsample2(base_filter)
        self.upsample6 = upsample1(base_filter)
        self.de4 = Decoding_block1(base_filter, n_convblock)
        self.de3 = Decoding_block1(base_filter, n_convblock)
        self.de2 = Decoding_block1(base_filter, n_convblock)
        self.de1 = Decoding_block2(base_filter, n_convblock)
        self.multi = muti_scale(64, 32, stride=1)
        self.de_end = ConvLayer1(base_filter, base_filter, 3, stride=1)

    def forward(self, x, x_in):
        encode0, down0 = self.e1(x)
        encode1, down1 = self.e2(down0)
        encode2, down2 = self.e3(down1)
        encode3, down3 = self.e4(down2)
        media_end = self.multi(down3)
        encode0 = self.upsample1(encode0, encode1)
        encode1 = self.upsample2(encode0, encode1, encode2)
        encode2 = self.upsample3(encode1, encode2, encode3)
        encode0 = self.upsample4(encode0, encode1)
        encode1 = self.upsample5(encode0, encode1, encode2)
        encode0 = self.upsample6(encode0, encode1)
        decode3 = self.de4(encode2, encode3, media_end)
        decode2 = self.de3(encode1, encode2, decode3)
        decode1 = self.de2(encode0, encode1, decode2)
        decode0 = self.de1(encode0, decode1)
        decode0 = self.de_end(decode0)

        return x_in + decode0


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fcn(x)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.T = 4
        self.est = FCN()

        self.net = ziwangluo(64, 3)
        self.lines = nn.ModuleList([line() for _ in range(self.T)])

        self.trans_3t64 = torch.nn.Sequential(
            ConvLayer1(3, 64, 3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(64, 64, 3, stride=1)
        )
        self.trans_1t64 = torch.nn.Sequential(
            ConvLayer1(1, 64, 3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(64, 64, 3, stride=1)
        )
        self.trans_64t3 = torch.nn.Sequential(
            ConvLayer1(64, 64, 3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(64, 3, 3, stride=1)
        )
        self.trans_128t64 = nn.Sequential(
            ConvLayer1(128, 64, 3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(64, 64, 3, stride=1)
        )

        G0 = 64
        kSize = 3
        self.RNNF = nn.ModuleList([nn.Sequential(*[
            nn.Conv2d((i + 2) * G0, G0, 1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(G0, 64, kSize, padding=(kSize - 1) // 2, stride=1)
        ]) for i in range(self.T - 1)])

        self.beta = Beta(64, 16)

    def forward(self, x):
        sigma_prediction = self.est(x)
        x = self.trans_3t64(x)
        sigma64 = self.trans_1t64(sigma_prediction)

        mid = x
        V_list = []
        for i in range(self.T):
            x_in = x
            input_all = torch.cat([x, sigma64], dim=1)
            x = self.trans_128t64(input_all)
            x = self.net(x, x_in)
            V_list.append(x)
            if i != 0:
                U_list = self.beta(V_list)
                x = self.RNNF[i - 1](torch.cat(U_list, 1))
            x = self.lines[i](x, mid)
        x = self.trans_64t3(x)

        return sigma_prediction, x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                #torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.xavier_uniform_(m.weight.data)
                #torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
