"""
VIB Module
"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.common.initializer as init
from mindspore.common.initializer import Initializer, _assignment, random_normal
import numpy as np


def weights_init_kaiming(module_):
    """
    weight init
    """
    classname = module_.__class__.__name__
    if classname.find('Conv') != -1:
        module_.weight.set_data(init.initializer(init.HeNormal(negative_slope=0, mode='fan_in'),\
            module_.weight.shape, module_.weight.dtype))
    elif classname.find('Linear') != -1:
        module_.weight.set_data(init.initializer(init.HeNormal(negative_slope=0, mode='fan_out'),\
        module_.weight.shape, module_.weight.dtype))
        module_.bias.set_data(init.initializer(init.Zero(), module_.bias.shape, module_.bias.dtype))
    elif classname.find('BatchNorm1d') != -1:
        module_.gamma.set_data(init.initializer(NormalWithMean(mu=1, sigma=0.01),\
        module_.gamma.shape, module_.gamma.dtype))
        module_.beta.set_data(init.initializer(init.Zero(), module_.beta.shape, module_.beta.dtype))


def weights_init_classifier(module_):
    """
    weight init
    """
    classname = module_.__class__.__name__
    if classname.find('Linear') != -1:
        module_.gamma.set_data(init.initializer(init.Normal(sigma=0.001),\
            module_.gamma.shape, module_.gamma.dtype))
        if module_.bias:
            module_.bias.set_data(init.initializer(init.Zero(),\
                module_.bias.shape, module_.bias.dtype))


class NormalWithMean(Initializer):
    """
    Initialize a normal array, and obtain values N(0, sigma) from the uniform distribution
    to fill the input tensor.

    Args:
        sigma (float): The sigma of the array. Default: 0.01.

    Returns:
        Array, normal array.
    """
    def __init__(self, mu=0, sigma=0.01):
        super(NormalWithMean, self).__init__(sigma=sigma)
        self.miu = mu
        self.sigma = sigma

    def _initialize(self, arr):
        """
        init
        """
        seed, seed2 = self.seed
        output_tensor = ms.Tensor(np.zeros(arr.shape, dtype=np.float32) + \
            np.ones(arr.shape, dtype=np.float32) * self.miu)
        random_normal(arr.shape, seed, seed2, output_tensor)
        output_data = output_tensor.asnumpy()
        output_data *= self.sigma
        _assignment(arr, output_data)

########################################################################
# Variational Distillation
########################################################################
class ChannelCompress(nn.Cell):
    """
    ChannelCompress
    """
    def __init__(self, in_ch=2048, out_ch=256):
        super(ChannelCompress, self).__init__()
        num_bottleneck = 1000
        add_block = []
        add_block += [nn.Dense(in_ch, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_features=num_bottleneck)]
        add_block += [nn.ReLU()]

        add_block += [nn.Dense(num_bottleneck, 500)]
        add_block += [nn.BatchNorm1d(500)]
        add_block += [nn.ReLU()]

        add_block += [nn.Dense(500, out_ch)]

        add_block = nn.SequentialCell(add_block)

        weights_init_kaiming(add_block)

        self.model = add_block

    def construct(self, x):
        """
        construct
        """
        x = self.model(x)
        return x

########################################################################
# Variational Distillation
########################################################################
class VIB(nn.Cell):
    """
    VIB module
    """
    def __init__(self, in_ch=2048, z_dim=256, num_class=395):
        super(VIB, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
        # classifier of VIB
        classifier = []
        classifier += [nn.Dense(self.out_ch, self.out_ch // 2)]
        classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(0.5)]
        classifier += [nn.Dense(self.out_ch // 2, self.num_class)]
        classifier = nn.SequentialCell(classifier)
        weights_init_classifier(classifier)
        self.classifier = classifier

    def construct(self, v):
        """
        construct
        """
        z_given_v = self.bottleneck(v)
        logits_given_z = self.classifier(z_given_v)
        return z_given_v, logits_given_z
