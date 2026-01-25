"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
import math



class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super(space_to_depth,self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)










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
        
        # out =out.view(b, -1,1,1)
        
        return out









class mcdc_block(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inchannels, outchannels):
        super(mcdc_block, self).__init__()

        self.conv2_dila =  nn.Conv2d(inchannels, outchannels, 3, 2, 2,dilation=2, bias=False)
        # self.conv3_dila =  nn.Conv2d(inchannels, outchannels, 3, 2, 3,dilation=3, bias=False)
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False)        
        self.spd = space_to_depth()
        self.bn1 = nn.BatchNorm2d(4*outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4*outchannels, 2*outchannels, kernel_size=3, padding=1, bias=False)
        # self.conv2_2 = nn.Conv2d(2*outchannels, outchannels, kernel_size=3, padding=1, bias=False)
        # self.conv2_dila = nn.Conv2d(4*outchannels, outchannels, kernel_size=3, padding=2,dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(2*outchannels)

        self.relu_2 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(2*outchannels, outchannels, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(outchannels)
        self.relu_2_2 = nn.ReLU(inplace=True)
        self.conv2_1X1 = nn.Conv2d(outchannels, outchannels, kernel_size=1)
         
        self.conv3= nn.Conv2d(inchannels,outchannels,3, 2, 1, bias=False)                                    
        self.bn3 =  nn.BatchNorm2d(outchannels)


        # self.SE = SELayer(reduction=16)
        self.eca = EfficientChannelAttention(outchannels*2)

                                    

    def forward(self, x):
        residual = x
        # residual_2 = x

        out = self.conv1(x)
        out = self.spd(out)
        out = self.bn1(out)
        out = self.relu(out)
        # out, att_c = self.channel_layer(out)
        # out, att_c = self.channel_layer(out)
        # out, att_s = self.spatial_layer(out)
        # out = self.conv2_1(out)
        # out = self.conv2_2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.conv2_2(out)
        out = self.bn2_2(out)
        # out, att_c = self.channel_layer(out)
        # out, att_s = self.spatial_layer(out)
        # out = self.ECA(out)


        residual = self.conv2_dila(residual)
        residual = self.conv2_1X1(residual)
        # residual_2 = self.conv3_dila(residual_2)


        fr_feature = torch.cat([out, residual], dim=1)

        fusion_score = self.eca(fr_feature)

        out = fusion_score[:,0,:,:].unsqueeze(-1)*out+fusion_score[:,1,:,:].unsqueeze(-1)*residual
        out = self.relu(out)



                                     # nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, kernel_size=3, stride=1, padding=1, bias=False),
                                    # space_to_depth(),##还可加无干扰的、附加的、有用的信息
                                    # nn.BatchNorm2d(4*num_outchannels_conv3x3),
                                    # nn.ReLU(inplace=True),
                                    # nn.Conv2d(4*num_outchannels_conv3x3, num_outchannels_conv3x3, kernel_size=3, padding=1, bias=False),##还可改为更加尽可能保留信息的维度变换操作！！！！！！！！
                                    # nn.BatchNorm2d(num_outchannels_conv3x3)


        return out















def resnet18():
    """
    return a ResNet 18 object
    """
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])


#if __name__ =="__main__":
#	net = resnet50()
#	x = torch.empty((2,3,112,112)).normal_()
#	print(net(x).shape)

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    # nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                    mcdc_block(dims[i], dims[i+1]),

            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x




def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x