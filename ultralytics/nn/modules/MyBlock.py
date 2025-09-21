
import torch
import torch.nn as nn


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class DWConv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=0):
        super(DWConv,self).__init__()
        self.depthwise = nn.Conv2d(in_channel,in_channel,groups=in_channel,kernel_size=kernel_size,stride=stride,padding=padding)
        self.pointwise = nn.Conv2d(in_channel,out_channel,kernel_size=1)

    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Localperception(nn.Module):

    def __init__(self,in_channel,out_channel,scale=0.01):
        super(Localperception,self).__init__()

        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=in_channel//2,kernel_size=1),
            nn.BatchNorm2d(in_channel//2),
            nn.SiLU(),
            DWConv(in_channel=in_channel//2,out_channel=in_channel,kernel_size=5,padding=2),
            nn.BatchNorm2d(in_channel // 2),
            nn.SiLU()

        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=in_channel//2,kernel_size=1),
            nn.BatchNorm2d(in_channel // 2),
            nn.SiLU(),
            DWConv(in_channel=in_channel//2, out_channel=in_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channel // 2),
            nn.SiLU()
        )
        self.finalconv = nn.Conv2d(in_channels=in_channel*4,out_channels=out_channel,kernel_size=1)
        # self.scale = scale


    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat([x0,x1],dim=1)
        out = self.finalconv(out)
        out = out + x
        out = nn.functional.silu(out)
        return out



# x = torch.rand(8,256,80,80)
# FEM = FEM(256,256)
# x = FEM(x)
# print(x.size())

x = torch.rand(8,128,80,80)
LP = Localperception(128,128)
x = LP(x)
print(x.size())