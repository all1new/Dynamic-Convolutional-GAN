import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
from Dynamic_Conv.drconv import DRConv2d
from Dynamic_Conv.condConv import CondConv2D
import math

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, 
                 groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Basic_dcd(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dowansample=None):
        super(Basic_dcd, self).__init__()
        self.inplanes = inplanes
        
        self.conv1 = conv_dy(inplanes, planes, 1, 1, 0)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = conv_dy(planes, planes, 3, stride, 1)
        self.bn2 = nn.InstanceNorm2d(planes)

        # self.conv4 = conv_dy(planes*2, planes, 1, 1, 0)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = dowansample
        self.stride = stride

        # self.Conv_size = nn.Conv2d(planes * 4, planes * 4, 3, stride, 1)

    def forward(self, x):
        residual = x    #[2, 64, 64, 64]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)   # [2, 64, 64, 64]
        out = self.bn2(out)
        # out = self.relu(out)    
        
        # out = self.conv3(out)   # [2, 128, 128, 128]
        # out = self.bn3(out)

        # out = self.conv4(out)
       
        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out

class conv_dy(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(conv_dy, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.dim = int(math.sqrt(inplanes))
        squeeze = max(inplanes * 4, self.dim ** 2) // 16

        self.q = nn.Conv2d(inplanes, self.dim, 1, stride, 0, bias=False)   # 降维

        self.p = nn.Conv2d(self.dim, planes, 1, 1, 0, bias=False)          # 升维
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(inplanes, squeeze, bias=False),
            SEModule_small(squeeze),                    # 加了一个小的注意力模块
        )
        self.fc_phi = nn.Linear(squeeze, self.dim**2, bias=False)   # 论文中的phi矩阵,phi即φ，通道融合矩阵
        self.fc_scale = nn.Linear(squeeze, planes, bias=False)
        self.hs = Hsigmoid()

    def forward(self, x):
        r = self.conv(x)                    # 先进行卷积，后对卷积结果加权
        b, c, _, _ = x.size()               # 获得batchsize和通道数
        y = self.avg_pool(x).view(b, c)     # 平均池化，self.avg_pool(x)尺寸为[b, c, 1, 1], 故加view, se模块操作
        y = self.fc(y)                      # fc层 这是SE模块，用于提取重要信息，剔除不重要的信息
        phi = self.fc_phi(y).view(b, self.dim, self.dim)    # phi即φ，这一步即为Φ(x)，通道融合矩阵
        scale = self.hs(self.fc_scale(y)).view(b, -1, 1, 1) # hs即Hsigmoid()激活函数 这一步表示权值参数就准备好了
        r = scale.expand_as(r) * r                          # 这里就是加权操作，从公式来看这里应该是A * W0
                                                            # 实际上这里是 A*W0*x,即把参数的获取和参数计算融合到一块，fc_scale实现了A*W0

        out = self.bn1(self.q(x))                           # q的话就是压缩通道
        _, _, h, w = out.size()                             # 这里操作的顺序和示意图不太一样

        out = out.view(b, self.dim, -1)
        out = self.bn2(torch.matmul(phi, out)) + out        # 加out做类似残差的处理
        out = out.view(b, -1, h, w)
        out = self.p(out) + r                               # p是把通道维数进行升维
        return out

class Basic_dcd_sa(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dowansample=None):
        super(Basic_dcd_sa, self).__init__()
        self.inplanes = inplanes
        
        self.conv1 = conv_dy(inplanes, planes*2, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(planes*2)
        # self.bn1 = nn.InstanceNorm2d(planes*2)
        self.conv2 = conv_dy(planes*2, planes*4, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(planes*4)
        # self.bn2 = nn.InstanceNorm2d(planes*4)
        self.conv3 = conv_dy(planes*4, planes, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(planes)
        # self.bn3 = nn.InstanceNorm2d(planes)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.aconv1 = Conv2dBlock(inplanes, planes * 2, kernel_size=7, stride=1,padding=3, norm='in', use_sn=True)
        # self.aconv2 = Conv2dBlock(planes * 2, inplanes, kernel_size=3, stride=1,padding=1, norm='in', use_sn=True)
        self.aconv2 = Conv2dBlock(planes * 2, inplanes, kernel_size=7, stride=1,padding=3, norm='in', use_sn=True)
        self.upsample = nn.Upsample(scale_factor=2)

        # self.conv4 = conv_dy(planes*2, planes, 1, 1, 0)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = dowansample
        self.stride = stride

        # self.Conv_size = nn.Conv2d(planes * 4, planes * 4, 3, stride, 1)

    def forward(self, x):
        residual = x            #[2, 64, 64, 64]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)   # [2, 64, 64, 64]
        out = self.bn2(out)
        out = self.relu(out)    
        
        out = self.conv3(out)   # [2, 128, 128, 128]
        out = self.bn3(out)

        # out = self.conv4(out)
       
        scale = self.avg_pool(x)
        scale = self.aconv1(scale)
        scale = self.aconv2(scale)
        scale = self.upsample(scale)
        residual = scale.expand_as(residual) * residual

        if self.downsample is not None:
            residual = self.downsample(residual)
        # if self.stride == 1:            # 即没有下采样
        #     scale = self.upsample(scale)
        # else:
        #     residual = self.downsample(residual)
        
        # residual = scale.expand_as(residual) * residual

        out += residual
        out = self.relu(out)

        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1, 
                 use_bias=True, use_sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if use_sn:
            self.conv = spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)           
        if self.activation:
            x = self.activation(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, use_sn=use_sn)]
        self.model = nn.Sequential(*model)
        self.se = SELayer(dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.model(x)
        out = self.se(out)
        out += residual

        out = self.relu(out)

        return out

class Get_image(nn.Module):
    def __init__(self, input_dim, output_dim, activation='tanh'):
        super(Get_image, self).__init__()
        self.conv = Conv2dBlock(input_dim, output_dim, kernel_size=3, stride=1, 
                                padding=1, pad_type='reflect', activation=activation)
        
    def forward(self, x):
        return self.conv(x)

class DownsampleResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activate='relu', pad_type='zero', use_sn=False):
        super(DownsampleResBlock, self).__init__()
        self.conv_1 = nn.ModuleList()
        self.conv_2 = nn.ModuleList()

        self.conv_1.append(Conv2dBlock(input_dim,input_dim,3,1,1, 'none', activate, pad_type,use_sn=use_sn))
        self.conv_1.append(Conv2dBlock(input_dim,output_dim,3,1,1,'none', activate, pad_type,use_sn=use_sn))
        self.conv_1.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv_1 = nn.Sequential(*self.conv_1)


        self.conv_2.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv_2.append(Conv2dBlock(input_dim,output_dim,1,1,0,'none', activate, pad_type,use_sn=use_sn))
        self.conv_2 = nn.Sequential(*self.conv_2)

    def forward(self, x):
        out = self.conv_1(x) + self.conv_2(x)
        return out

class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
          self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
          self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
          return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
          return F.layer_norm(x, normalized_shape) 

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'         

# 多尺度鉴别器        
class Discriminator(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_layers=3, norm='none',
                 activ='lrelu', pad_type='reflect', use_sn=True):
        super(Discriminator, self).__init__()

        self.model = nn.ModuleList()
        self.model.append(Conv2dBlock(input_dim, dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))
        dim_in = dim
        for i in range(n_layers-1):
            dim_out = min(dim*8, dim_in*2)
            self.model.append(DownsampleResBlock(dim_in, dim_out, norm, activ, pad_type, use_sn))
            dim_in = dim_out

        self.model.append(Conv2dBlock(dim_in, 1, 1, 1, activation='none', use_bias=False, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
        
class MultiDiscriminator(nn.Module):
    def __init__(self, **parameter_dic):
        super(MultiDiscriminator, self).__init__()
        self.model_1 = Discriminator(**parameter_dic)
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.mdoel_2 = Discriminator(**parameter_dic)

    def forward(self, x):
        pre1 = self.model_1(x)
        pre2 = self.mdoel_2(self.down(x))
        return [pre1, pre2]

# 结构生成器网络
class StructureGen(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_res=4, activ='relu', 
                 norm='in', pad_type='reflect', use_sn=True):
        super(StructureGen, self).__init__()

        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        self.content_param = nn.ModuleList()

        # 输入有4个通道
        self.input_layer = Conv2dBlock(input_dim+1, dim, 7, 1, 3, norm, activ,  pad_type, use_sn=use_sn)
        self.down_sample += [nn.Sequential(
            Conv2dBlock(dim, 2*dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(2*dim, 2*dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
        )]

        self.down_sample += [nn.Sequential(
            Conv2dBlock(2*dim, 4*dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(4*dim, 4*dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
        )]

        self.down_sample += [nn.Sequential(
            Conv2dBlock(4*dim, 8*dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn)
        )]
        dim = 8*dim

        # content decoder
        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim//2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
        ))]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//2, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//2, dim//4, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//4, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//4, dim//8, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]  

        self.content_param += [Conv2dBlock(dim//2, dim//2, 5, 1, 2, norm, activ, pad_type)]
        self.content_param += [Conv2dBlock(dim//4, dim//4, 5, 1, 2, norm, activ, pad_type)]
        self.content_param += [Conv2dBlock(dim//8, dim//8, 5, 1, 2, norm, activ, pad_type)]

        self.image_net = Get_image(dim//8, input_dim)   # 得到结果前再做一次卷积

    def forward(self, inputs):
        # 整体上还是一个U-Net架构
        x0 = self.input_layer(inputs)
        x1 = self.down_sample[0](x0)
        x2 = self.down_sample[1](x1)
        x3 = self.down_sample[2](x2)

        u1 = self.up_sample[0](x3) + self.content_param[0](x2)
        u2 = self.up_sample[1](u1) + self.content_param[1](x1)
        u3 = self.up_sample[2](u2) + self.content_param[2](x0)

        images_out = self.image_net(u3)
        return images_out

class StructureGen_dcd(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_res=4, activ='relu', 
                 norm='in', pad_type='reflect', use_sn=True):
        super(StructureGen_dcd, self).__init__()

        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        self.content_param = nn.ModuleList()
        
        # 输入有4个通道
        self.input_layer = Conv2dBlock(input_dim+1, dim, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn)

        downsample1 = Conv2dBlock(dim, dim * 2, 4, 2, 1, activation='none', use_sn=use_sn)
        # self.down_sample += [nn.Sequential(
        #     Conv2dBlock(dim, 2*dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
        #     Conv2dBlock(2*dim, 2*dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
        # )]
        self.down_sample += [nn.Sequential(Basic_dcd(dim, 2 * dim, stride=2, dowansample=downsample1))]
        dim = 2 * dim

        # self.down_sample += [nn.Sequential(
        #     Conv2dBlock(2*dim, 4*dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
        #     Conv2dBlock(4*dim, 4*dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
        # )]
        
        downsample2 = Conv2dBlock(dim, dim * 2, 4, 2, 1, activation='none', use_sn=use_sn)
        self.down_sample += [nn.Sequential(Basic_dcd(dim, 2 * dim, stride=2, dowansample=downsample2))]
        dim = 2 * dim
        
        self.down_sample += [nn.Sequential(
            Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn)
        )]
        dim = 2*dim

        # content decoder
        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim//2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
        ))]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//2, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//2, dim//4, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//4, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//4, dim//8, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]  

        self.content_param += [Conv2dBlock(dim//2, dim//2, 5, 1, 2, norm, activ, pad_type)]
        self.content_param += [Conv2dBlock(dim//4, dim//4, 5, 1, 2, norm, activ, pad_type)]
        self.content_param += [Conv2dBlock(dim//8, dim//8, 5, 1, 2, norm, activ, pad_type)]

        self.image_net = Get_image(dim//8, input_dim)   # 得到结果前再做一次卷积

    def forward(self, inputs):
        # 整体上还是一个U-Net架构
        x0 = self.input_layer(inputs)
        x1 = self.down_sample[0](x0)
        x2 = self.down_sample[1](x1)
        x3 = self.down_sample[2](x2)

        u1 = self.up_sample[0](x3) + self.content_param[0](x2)
        u2 = self.up_sample[1](u1) + self.content_param[1](x1)
        u3 = self.up_sample[2](u2) + self.content_param[2](x0)

        images_out = self.image_net(u3)
        return images_out

class StructureGen_dcd_v2(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_res=4, activ='relu', 
                 norm='in', pad_type='reflect', use_sn=True):
        super(StructureGen_dcd_v2, self).__init__()

        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        self.content_param = nn.ModuleList()
        
        # 输入有4个通道
        # self.input_layer = Conv2dBlock(input_dim+1, dim, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn)
        self.input_layer = DRConv2d(in_channels=4, out_channels=dim, kernel_size=3, region_num=8)

        downsample1 = Conv2dBlock(dim, dim * 2, 4, 2, 1, activation='none', use_sn=use_sn)
        self.down_sample += [nn.Sequential(Basic_dcd(dim, 2 * dim, stride=2, dowansample=downsample1))]
        dim = 2 * dim
        
        downsample2 = Conv2dBlock(dim, dim * 2, 4, 2, 1, activation='none', use_sn=use_sn)
        self.down_sample += [nn.Sequential(Basic_dcd(dim, 2 * dim, stride=2, dowansample=downsample2))]
        dim = 2 * dim
        
        self.down_sample += [nn.Sequential(
            Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn)
        )]
        dim = 2*dim

        # content decoder
        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim//2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
        ))]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//2, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//2, dim//4, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//4, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//4, dim//8, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]  

        # self.content_param += [Conv2dBlock(dim//2, dim//2, 5, 1, 2, norm, activ, pad_type)]
        # self.content_param += [Conv2dBlock(dim//4, dim//4, 5, 1, 2, norm, activ, pad_type)]
        # self.content_param += [Conv2dBlock(dim//8, dim//8, 5, 1, 2, norm, activ, pad_type)]
        self.content_param += [Basic_dcd(dim//2, dim//2)]
        self.content_param += [Basic_dcd(dim//4, dim//4)]
        self.content_param += [Basic_dcd(dim//8, dim//8)]

        self.image_net = Get_image(dim//8, input_dim)   # 得到结果前再做一次卷积

    def forward(self, inputs):
        # 整体上还是一个U-Net架构
        x0 = self.input_layer(inputs)

        x1 = self.down_sample[0](x0)
        x2 = self.down_sample[1](x1)
        x3 = self.down_sample[2](x2)

        u1 = self.up_sample[0](x3) + self.content_param[0](x2)
        u2 = self.up_sample[1](u1) + self.content_param[1](x1)
        u3 = self.up_sample[2](u2) + self.content_param[2](x0)

        images_out = self.image_net(u3)
        return images_out

class StructureGen_dcd_v3(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_res=4, activ='relu', 
                 norm='in', pad_type='reflect', use_sn=True):
        super(StructureGen_dcd_v3, self).__init__()

        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        self.content_param = nn.ModuleList()
        
        # 输入有4个通道
        # self.input_layer = Conv2dBlock(input_dim+1, dim, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn)
        # self.input_layer = DRConv2d(in_channels=4, out_channels=dim, kernel_size=3, region_num=8)
        self.input_layer = conv_dy(inplanes=4, planes=dim, kernel_size=7, stride=1, padding=3)
        

        downsample1 = Conv2dBlock(dim, dim * 2, 4, 2, 1, activation='none', use_sn=use_sn)
        self.down_sample += [nn.Sequential(Basic_dcd_sa(dim, 2 * dim, stride=2, dowansample=downsample1))]
        dim = 2 * dim
        
        downsample2 = Conv2dBlock(dim, dim * 2, 4, 2, 1, activation='none', use_sn=use_sn)
        self.down_sample += [nn.Sequential(Basic_dcd_sa(dim, 2 * dim, stride=2, dowansample=downsample2))]
        dim = 2 * dim
        
        downsample3 = Conv2dBlock(dim, dim * 2, 4, 2, 1, activation='none', use_sn=use_sn)
        self.down_sample += [nn.Sequential(
            Basic_dcd_sa(dim, 2 * dim, stride=2, dowansample=downsample3)
        )]
        dim = 2*dim

        # content decoder
        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim//2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
        ))]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//2, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//2, dim//4, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//4, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//4, dim//8, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]  

        # self.content_param += [Conv2dBlock(dim//2, dim//2, 5, 1, 2, norm, activ, pad_type)]
        # self.content_param += [Conv2dBlock(dim//4, dim//4, 5, 1, 2, norm, activ, pad_type)]
        # self.content_param += [Conv2dBlock(dim//8, dim//8, 5, 1, 2, norm, activ, pad_type)]
        self.content_param += [Basic_dcd(dim//2, dim//2)]
        self.content_param += [Basic_dcd(dim//4, dim//4)]
        self.content_param += [Basic_dcd(dim//8, dim//8)]

        self.image_net = Get_image(dim//8, input_dim)   # 得到结果前再做一次卷积

    def forward(self, inputs):
        # 整体上还是一个U-Net架构
        x0 = self.input_layer(inputs)   # 没有改变维度

        x1 = self.down_sample[0](x0)
        x2 = self.down_sample[1](x1)
        x3 = self.down_sample[2](x2)

        u1 = self.up_sample[0](x3) + self.content_param[0](x2)
        u2 = self.up_sample[1](u1) + self.content_param[1](x1)
        u3 = self.up_sample[2](u2) + self.content_param[2](x0)

        images_out = self.image_net(u3)
        return images_out

class StructureGen_drconv(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_res=4, activ='relu', 
                 norm='in', pad_type='reflect', use_sn=True):
        super(StructureGen_drconv, self).__init__()

        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        self.content_param = nn.ModuleList()

        # 输入有4个通道
        self.input_layer = DRConv2d(in_channels=4, out_channels=dim, kernel_size=3, region_num=8)
        self.down_sample += [nn.Sequential(
            Conv2dBlock(dim, 2*dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            # Conv2dBlock(2*dim, 2*dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
            DRConv2d(2*dim, 2*dim, kernel_size=3, region_num=8)
        )]

        self.down_sample += [nn.Sequential(
            Conv2dBlock(2*dim, 4*dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            DRConv2d(4*dim, 4*dim, kernel_size=3, region_num=8)
        )]

        self.down_sample += [nn.Sequential(
            Conv2dBlock(4*dim, 8*dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            DRConv2d(8*dim, 8*dim, kernel_size=3, region_num=8)
        )]
        dim = 8*dim

        # content decoder
        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim//2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
        ))]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//2, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//2, dim//4, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//4, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//4, dim//8, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]  

        self.content_param += [Conv2dBlock(dim//2, dim//2, 5, 1, 2, norm, activ, pad_type)]
        self.content_param += [Conv2dBlock(dim//4, dim//4, 5, 1, 2, norm, activ, pad_type)]
        self.content_param += [Conv2dBlock(dim//8, dim//8, 5, 1, 2, norm, activ, pad_type)]


        self.image_net = Get_image(dim//8, input_dim)   # 得到结果前再做一次卷积

    def forward(self, inputs):
        # 整体上还是一个U-Net架构
        x0 = self.input_layer(inputs)
        x1 = self.down_sample[0](x0)
        x2 = self.down_sample[1](x1)
        x3 = self.down_sample[2](x2)

        u1 = self.up_sample[0](x3) + self.content_param[0](x2)
        u2 = self.up_sample[1](u1) + self.content_param[1](x1)
        u3 = self.up_sample[2](u2) + self.content_param[2](x0)

        images_out = self.image_net(u3)
        return images_out

class StructureGen_condconv(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_res=4, activ='relu', 
                 norm='in', pad_type='reflect', use_sn=True):
        super(StructureGen_condconv, self).__init__()

        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        self.content_param = nn.ModuleList()

        # 输入有4个通道
        self.input_layer = Conv2dBlock(input_dim+1, dim, 7, 1, 3, norm, activ,  pad_type, use_sn=use_sn)
        self.down_sample += [nn.Sequential(
            Conv2dBlock(dim, 2*dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            # Conv2dBlock(2*dim, 2*dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
            CondConv2D(2*dim, 2*dim, kernel_size=5, stride=1, padding=2)
        )]

        self.down_sample += [nn.Sequential(
            Conv2dBlock(2*dim, 4*dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            # Conv2dBlock(4*dim, 4*dim, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
            CondConv2D(4*dim, 4*dim, kernel_size=5, stride=1, padding=2)
        )]

        self.down_sample += [nn.Sequential(
            Conv2dBlock(4*dim, 8*dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn),
            CondConv2D(8*dim, 8*dim, kernel_size=5, stride=1, padding=2)
        )]
        dim = 8*dim

        # content decoder
        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim//2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn)
        ))]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//2, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//2, dim//4, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]

        self.up_sample += [(nn.Sequential(
            ResBlocks(n_res, dim//4, norm, activ, pad_type=pad_type),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim//4, dim//8, 5, 1, 2,norm, activ, pad_type, use_sn=use_sn)) )]  

        self.content_param += [Conv2dBlock(dim//2, dim//2, 5, 1, 2, norm, activ, pad_type)]
        self.content_param += [Conv2dBlock(dim//4, dim//4, 5, 1, 2, norm, activ, pad_type)]
        self.content_param += [Conv2dBlock(dim//8, dim//8, 5, 1, 2, norm, activ, pad_type)]

        self.image_net = Get_image(dim//8, input_dim)   # 得到结果前再做一次卷积

    def forward(self, inputs):
        # 整体上还是一个U-Net架构
        x0 = self.input_layer(inputs)
        x1 = self.down_sample[0](x0)
        x2 = self.down_sample[1](x1)
        x3 = self.down_sample[2](x2)

        u1 = self.up_sample[0](x3) + self.content_param[0](x2)
        u2 = self.up_sample[1](u1) + self.content_param[1](x1)
        u3 = self.up_sample[2](u2) + self.content_param[2](x0)

        images_out = self.image_net(u3)
        return images_out

