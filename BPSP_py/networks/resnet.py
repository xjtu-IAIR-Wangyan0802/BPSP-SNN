import torch
import torch.nn as nn
import layers.conv as conv
import layers.pooling as pooling
import layers.dropout as dropout
import layers.linear as linear
from torch.cuda.amp import custom_fwd, custom_bwd
import global_v as glv


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    config = {'in_channels': in_planes, 'out_channels': out_planes,
              'kernel_size': 3, 'padding': 1, 'stride': stride, 'dilation': dilation, 'threshold': 1}
    return conv.ConvLayer(config=config)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    config = {'in_channels': in_planes, 'out_channels': out_planes,
              'kernel_size': 1, 'padding': 0, 'stride': stride, 'threshold': 1}
    return conv.ConvLayer(config=config)

#实现了一个基本的残差块(Residual Block)结构,是 ResNet 等深度卷积神经网络中的基本组成部分
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, **kwargs):
        #dilation: 卷积核的膨胀率,这里被设置为 1 表示普通卷积
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # may need custom backward 自定义反向传播函数
        # out = out + identity
        out = AddFunc.apply(out, identity)

        return out

# 加法运算并支持反向传播
class AddFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    @custom_bwd
    #backward() 方法定义了反向传播的操作。它接受一个梯度 grad,并根据保存的 a 和 b 计算关于 a 和 b 的梯度。
    # 这里有一个特殊的处理,当 a + b 等于 0 时,将其设置为 1,以避免除以 0 的错误。最后,返回关于 a 和 b 的梯度。
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        s = a + b
        s[s == 0] = 1
        return grad * a / s, grad * b / s


class SpikingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, groups=1, width_per_group=64, norm_layer=None, **kwargs):
        super(SpikingResNet, self).__init__()
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        config = {'in_channels': 3, 'out_channels': self.inplanes,
                  'kernel_size': 5, 'padding': 2, 'stride': 1, 'dilation': 1, 'threshold': 1}
        self.conv1 = conv.ConvLayer(config=config)

        # self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], stride=2, **kwargs)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, **kwargs)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, **kwargs)
        config = {'kernel_size': 32 // 2 ** 4}
        self.pool = pooling.PoolLayer(config=config)
        config = {'n_inputs': 512 * 4 * block.expansion, 'n_outputs': num_classes, 'threshold': 1}
        self.fc = linear.LinearLayer(config=config)

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, **kwargs))

        layers.append(dropout.DropoutLayer(config={'p':0.1}))
        return nn.Sequential(*layers)

    def forward(self, x, labels, epoch, is_train):
        assert (is_train or labels == None)
        # See note [TorchScript super()]
        x = self.conv1(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.pool(x)
        x = self.fc(x, labels)

        return x


class Network(SpikingResNet):
    def __init__(self, input_shape=None):
        super(Network, self).__init__(BasicBlock, [2, 2, 2, 2], glv.network_config['n_class'])
        print("-----------------------------------------")

