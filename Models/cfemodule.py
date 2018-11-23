import torch
import torch.nn as nn

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class CFEM(nn.Module):
    """
    INPUT:2, 512, 20, 10
    OUTPUT:2, 512, 20, 10
    """

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, groups=8, thinning=2, k=7, dilation=1):
        super(CFEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        second_in_planes = in_planes // thinning

        p = (k-1)//2
        self.cfem_a = list()
        self.cfem_a += [BasicConv(in_planes, in_planes, kernel_size = (1,k), stride = 1, padding = (0,p), groups = groups, relu = False)]
        self.cfem_a += [BasicConv(in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size=3, stride=stride, padding=dilation, groups = 4, dilation=dilation)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size = (k, 1), stride = 1, padding = (p, 0), groups = groups, relu = False)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_a = nn.ModuleList(self.cfem_a)

        self.cfem_b = list()
        self.cfem_b += [BasicConv(in_planes, in_planes, kernel_size=(k, 1), stride=1, padding = (p,0), groups = groups, relu = False)]
        self.cfem_b += [BasicConv(in_planes, second_in_planes, kernel_size=1, stride=1)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size=3, stride=stride, padding=dilation,groups =4,dilation=dilation)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size=(1, k), stride=1, padding = (0, p), groups = groups, relu = False)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size=1, stride=1)]
        self.cfem_b = nn.ModuleList(self.cfem_b)


        self.ConvLinear = BasicConv(2 * second_in_planes, out_planes, kernel_size = 1, stride = 1, relu = False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size = 1, stride = stride, relu = False)
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):

        x1 = self.cfem_a[0](x)
        x1 = self.cfem_a[1](x1)
        x1 = self.cfem_a[2](x1)
        x1 = self.cfem_a[3](x1)
        x1 = self.cfem_a[4](x1)

        x2 = self.cfem_b[0](x)
        x2 = self.cfem_b[1](x2)
        x2 = self.cfem_b[2](x2)
        x2 = self.cfem_b[3](x2)
        x2 = self.cfem_b[4](x2)

        out = torch.cat([x1, x2], 1)

        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

def get_CFEM(cfe_type='large', in_planes=512, out_planes=512, stride=1, scale=1, groups=8, dilation=1):
    assert cfe_type in ['large', 'normal', 'light'], 'no that type of CFEM'
    if cfe_type == 'large':
        return CFEM(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=2)
    elif cfe_type == 'normal':
        return CFEM(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=4)
    else:
        return CFEM(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=8)


if __name__=='__main__':
    # print(CFEM(512, 512, stride=1, scale=1, groups=8, dilation=1, thinning=2))
    net = CFEM(32, 32, stride=1, scale=1, groups=8, dilation=1, thinning=2)
    input_data = torch.randn(1, 32, 800, 700)
    
    out = net(input_data)
    print(out.shape)