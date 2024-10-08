"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import math



class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.linear =  nn.Linear(c, 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax =nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

    def forward(self, x):
        b, c, _, _ = x.size()

        x = self.avg_pool(x)

        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        x = self.relu_2(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x = self.dropout(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        x = self.linear(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        x = self.relu(x)
        out = self.softmax(x)


        
        return out



class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function

        # residual function

        if stride ==2:
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                space_to_depth(),
                nn.BatchNorm2d(4*out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(4*out_channels, out_channels * 2, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(2*out_channels, out_channels , kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            ]
        else:
            layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            ]

        #shortcut
        self.residual_function = torch.nn.Sequential(*layers)
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))







class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        layers = [
                  nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = False),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True),
        ]
        self.strde = stride
        self.relu = nn.ReLU(inplace=True)
        if stride ==2:

            layers2 = [
            nn.Conv2d(out_channels, out_channels,stride= 1, kernel_size=3, padding=1, bias= False),
            space_to_depth(),   # the output of this will result in 4*out_channels
            nn.BatchNorm2d(4*out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(4*out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(BottleNeck.expansion*out_channels, BottleNeck.expansion*out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels* BottleNeck.expansion),
            nn.ReLU(inplace=True)
                      
            ]

        else:

            layers2 = [
            nn.Conv2d(out_channels, out_channels,stride= stride, kernel_size=3, padding=1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels* BottleNeck.expansion, kernel_size=1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),                       
            ]

        layers.extend(layers2)


        self.residual_function = torch.nn.Sequential(*layers)

		
        # self.residual_function = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False),
		# 	space_to_depth(),   # the output of this will result in 4*out_channels
        #     nn.BatchNorm2d(4*out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(4*out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        # )
        self.eca = EfficientChannelAttention(out_channels * BottleNeck.expansion*2)
        self.shortcut = nn.Sequential()

        if stride != 1 :
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
            #     nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            # )
            layers_3 = [
                  nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, 3,2, 2,dilation=2, bias=False),
                  nn.BatchNorm2d(out_channels * BottleNeck.expansion),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(BottleNeck.expansion*out_channels, BottleNeck.expansion*out_channels, kernel_size=1, bias=False),
                  nn.BatchNorm2d(out_channels* BottleNeck.expansion),
                  nn.ReLU(inplace=True)
            ]
            self.shortcut = nn.Sequential(*layers_3)
        if  in_channels != out_channels * BottleNeck.expansion: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )  

    def forward(self, x):

        out = self.residual_function(x)

        residual = self.shortcut(x)

        if self.strde != 1 :
            fr_feature = torch.cat([out, residual], dim=1)
            # return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
            fusion_score = self.eca(fr_feature)
            out = fusion_score[:,0,:,:].unsqueeze(-1)*out+fusion_score[:,1,:,:].unsqueeze(-1)*residual
            out = self.relu(out)
            return out
        else :
            return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=200):
        super().__init__()

        self.in_channels = 64

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))

        self.conv1 = Focus(3, 64, k=1,s=1)


		
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)    # Here in_channels = 64, and num_block[0] = 64 and s = 1 
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])


def resnet18():
    """
    return a ResNet 18 object
    """
    return ResNet(BottleNeck,[3,4,6,3])


#if __name__ =="__main__":
#	net = resnet50()
#	x = torch.empty((2,3,112,112)).normal_()
#	print(net(x).shape)



