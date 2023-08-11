import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.nn.parameter import Parameter



class BasicConv1d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Inception1d(nn.Module):
    def __init__(self, in_dim=1536):
        super(Inception1d, self).__init__()
        self.branch0 = BasicConv1d(in_dim, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv1d(in_dim, 512, kernel_size=1, stride=1),
            BasicConv1d(512, 256, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv1d(in_dim, 512, kernel_size=1, stride=1),
            BasicConv1d(512, 224, kernel_size=3, stride=1, padding=1),
            BasicConv1d(224, 256, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv1d(in_dim, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Inception2d(nn.Module):

    def __init__(self, in_dim=3072):
        super(Inception2d, self).__init__()

        self.branch0 = BasicConv2d(in_dim, 640, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_dim, 512, kernel_size=1, stride=1),
            BasicConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_dim, 512, kernel_size=1, stride=1),
            BasicConv2d(512, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 512, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_dim, 384, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        n_dim = len(x.size())
        if n_dim == 3:
            if self.p == 1:
                return x.mean(dim=[-1])
            elif self.p == float('inf'):
                return torch.flatten(F.adaptive_max_pool1d(x, 1), start_dim=1)
            else:
                return torch.flatten(F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), x.size(-1)).pow(1./self.p), start_dim=1)
        elif n_dim == 4:
            if self.p == 1:
                return x.mean(dim=[-1, -2])
            elif self.p == float('inf'):
                return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
            #return LF.gem(x, p=self.p, eps=self.eps)
            else:
                return torch.flatten(F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p), start_dim=1)