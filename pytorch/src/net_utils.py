'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Xiaohan Fei <feixh@cs.ucla.edu>
If you use this code, please cite the following paper:
A. Wong, X. Fei, S. Tsuei, and S. Soatto. Unsupervised Depth Completion from Visual Inertial Odometry.
https://arxiv.org/pdf/1905.08616.pdf

@article{wong2020unsupervised,
    title={Unsupervised Depth Completion From Visual Inertial Odometry},
    author={Wong, Alex and Fei, Xiaohan and Tsuei, Stephanie and Soatto, Stefano},
    journal={IEEE Robotics and Automation Letters},
    volume={5},
    number={2},
    pages={1899--1906},
    year={2020},
    publisher={IEEE}
}
'''
import torch
import numpy as np


EPSILON = 1e-10


def activation_func(activation_fn):
    '''
    Select activation function

    Arg(s):
        activation_fn : str
            name of activation function
    Returns:
        torch.nn.Module : activation function
    '''

    if 'linear' in activation_fn:
        return None
    elif 'leaky_relu' in activation_fn:
        return torch.nn.LeakyReLU(negative_slope=0.20, inplace=True)
    elif 'relu' in activation_fn:
        return torch.nn.ReLU()
    elif 'elu' in activation_fn:
        return torch.nn.ELU()
    elif 'sigmoid' in activation_fn:
        return torch.nn.Sigmoid()
    else:
        raise ValueError('Unsupported activation function: {}'.format(activation_fn))


'''
Network layers
'''
class Conv2d(torch.nn.Module):
    '''
    2D convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(Conv2d, self).__init__()

        padding = kernel_size // 2

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through a convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)

        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class DepthwiseSeparableConv2d(torch.nn.Module):
    '''
    Depthwise separable convolution class
    Performs
    1. separate k x k convolution per channel (depth-wise)
    2. 1 x 1 convolution across all channels (point-wise)

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel (k x k)
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(DepthwiseSeparableConv2d, self).__init__()

        padding = kernel_size // 2

        self.conv_depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=in_channels)

        self.conv_pointwise = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv_depthwise.weight)
            torch.nn.init.kaiming_normal_(self.conv_pointwise.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv_depthwise.weight)
            torch.nn.init.xavier_normal_(self.conv_pointwise.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv_depthwise.weight)
            torch.nn.init.xavier_uniform_(self.conv_pointwise.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.conv = torch.nn.Sequential(
            self.conv_depthwise,
            self.conv_pointwise)

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

        self.activation_func = activation_func

    def forward(self, x):
        '''
        Forward input x through a depthwise convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)

        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class AtrousConv2d(torch.nn.Module):
    '''
    2D atrous convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        dilation : int
            dilation of convolution (skips rate - 1 pixels)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(AtrousConv2d, self).__init__()

        padding = dilation

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through an atrous convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)

        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class TransposeConv2d(torch.nn.Module):
    '''
    Transpose convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(TransposeConv2d, self).__init__()

        padding = kernel_size // 2

        self.deconv = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.deconv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through a transposed convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        deconv = self.deconv(x)

        if self.use_batch_norm:
            deconv = self.batch_norm(deconv)
        elif self.use_instance_norm:
            deconv = self.instance_norm(deconv)

        if self.activation_func is not None:
            return self.activation_func(deconv)
        else:
            return deconv


class UpConv2d(torch.nn.Module):
    '''
    Up-convolution (upsample + convolution) block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        shape : list[int]
            two element tuple of ints (height, width)
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(UpConv2d, self).__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x, shape):
        '''
        Forward input x through an up convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        upsample = torch.nn.functional.interpolate(x, size=shape, mode='nearest')
        conv = self.conv(upsample)
        return conv


'''
Network encoder blocks
'''
'''
Network encoder blocks
'''
class ResNetBlock(torch.nn.Module):
    '''
    Basic ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(ResNetBlock, self).__init__()

        self.activation_func = activation_func

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.projection = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through a basic ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv2 + X)


class ResNetBottleneckBlock(torch.nn.Module):
    '''
    ResNet bottleneck block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(ResNetBottleneckBlock, self).__init__()

        self.activation_func = activation_func

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv3 = conv2d(
            out_channels,
            4 * out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.projection = Conv2d(
            in_channels,
            4 * out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through a ResNet bottleneck block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv3 + X)


class AtrousResNetBlock(torch.nn.Module):
    '''
    Basic atrous ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dilation : int
            dilation of convolution (skips rate - 1 pixels)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=2,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(AtrousResNetBlock, self).__init__()

        self.activation_func = activation_func

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv1 = AtrousConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.projection = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through an atrous ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)

        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv2 + X)


class VGGNetBlock(torch.nn.Module):
    '''
    VGGNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_convolution : int
            number of convolution layers
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(VGGNetBlock, self).__init__()

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        layers = []
        for n in range(n_convolution - 1):
            conv = conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            layers.append(conv)
            in_channels = out_channels

        conv = conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward input x through a VGG block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        return self.conv_block(x)


class AtrousVGGNetBlock(torch.nn.Module):
    '''
    Atrous VGGNet block class
    (last block performs atrous convolution instead of convolution with stride)

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_convolution : int
            number of convolution layers
        dilation : int
            dilation of atrous convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 dilation=2,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(AtrousVGGNetBlock, self).__init__()

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        layers = []
        for n in range(n_convolution - 1):
            conv = conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            layers.append(conv)
            in_channels = out_channels

        conv = AtrousConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward input x through an atrous VGG block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        return self.conv_block(x)


class AtrousSpatialPyramidPooling(torch.nn.Module):
    '''
    Atrous Spatial Pyramid Pooling class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dilations : list[int]
            dilations for different atrous convolution of each branch
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations=[6, 12, 18],
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(AtrousSpatialPyramidPooling, self).__init__()

        output_channels = out_channels // (len(dilations) + 1)

        # Point-wise 1 by 1 convolution branch
        self.conv1 = Conv2d(
            in_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # 3 by 3 convolutions with different dilation rate
        self.atrous_convs = torch.nn.ModuleList()

        for dilation in dilations:
            atrous_conv = AtrousConv2d(
                in_channels,
                output_channels,
                kernel_size=3,
                dilation=dilation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            self.atrous_convs.append(atrous_conv)

        # Global pooling
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.global_pool_conv = Conv2d(
            in_channels,
            output_channels,
            kernel_size=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        # Fuse point-wise (1 by 1) convolution, atrous convolutions, global pooling
        self.conv_fuse = Conv2d(
            (len(dilations) + 2) * output_channels,
            out_channels,
            kernel_size=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=False,
            use_instance_norm=False)

    def forward(self, x):
        '''
        Forward input x through a ASPP block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # List to hold all branches
        branches = []

        # Point-wise (1 by 1) convolution branch
        branches.append(self.conv1(x))

        # Atrous branches
        for atrous_conv in self.atrous_convs:
            branches.append(atrous_conv(x))

        # Global pooling branch
        global_pool = self.global_pool(x)
        global_pool = self.global_pool_conv(global_pool)
        global_pool = torch.nn.functional.interpolate(
            global_pool,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=True)
        branches.append(global_pool)

        return self.conv_fuse(torch.cat(branches, dim=1))


class SpatialPyramidPooling(torch.nn.Module):
    '''
    Spatial Pyramid Pooling class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_sizes : list[int]
            pooling kernel size of each branch
        pool_func : str
            max, average
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 pool_func='max',
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(SpatialPyramidPooling, self).__init__()

        output_channels = out_channels // len(kernel_sizes)

        # List of pooling kernel sizes
        self.kernel_sizes = kernel_sizes

        if pool_func == 'max':
            self.pool_func = torch.nn.functional.max_pool2d
        elif pool_func == 'average':
            self.pool_func = torch.nn.functional.avg_pool2d
        else:
            raise ValueError('Unsupported pooling function: {}'.format(pool_func))

        # List of convolutions to compress feature maps
        self.convs = torch.nn.ModuleList()

        for n in kernel_sizes:
            conv = Conv2d(
                in_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            self.convs.append(conv)

        self.conv_fuse = torch.nn.Sequential(
            Conv2d(
                2 * len(kernel_sizes) * output_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                weight_initializer=weight_initializer,
                activation_func=None,
                use_batch_norm=False,
                use_instance_norm=False))

    def forward(self, x):
        '''
        Forward input x through SPP block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        # List to hold all branches
        branches = [x]

        # Pyramid pooling branches
        for kernel_size, conv in zip(self.kernel_sizes, self.convs):
            pool = self.pool_func(
                x,
                kernel_size=(kernel_size, kernel_size),
                stride=(kernel_size, kernel_size))

            pool = torch.nn.functional.interpolate(
                pool,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=True)

            branches.append(conv(pool))

        return self.conv_fuse(torch.cat(branches, dim=1))


'''
Network decoder blocks
'''
class DecoderBlock(torch.nn.Module):
    '''
    Decoder block with skip connection

    Arg(s):
        in_channels : int
            number of input channels
        skip_channels : int
            number of skip connection channels
        out_channels : int
            number of output channels
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        deconv_type : str
            deconvolution types: transpose, up
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 deconv_type='up',
                 use_depthwise_separable=False):
        super(DecoderBlock, self).__init__()

        self.skip_channels = skip_channels
        self.deconv_type = deconv_type

        if deconv_type == 'transpose':
            self.deconv = TransposeConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
        elif deconv_type == 'up':
            self.deconv = UpConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)

        concat_channels = skip_channels + out_channels

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        self.conv = conv2d(
            concat_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x, skip=None, shape=None):
        '''
        Forward input x through a decoder block and fuse with skip connection

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            skip : torch.Tensor[float32]
                N x F x H x W skip connection
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        if self.deconv_type == 'transpose':
            deconv = self.deconv(x)
        elif self.deconv_type == 'up':

            if skip is not None:
                shape = skip.shape[2:4]
            elif shape is not None:
                pass
            else:
                n_height, n_width = x.shape[2:4]
                shape = (int(2 * n_height), int(2 * n_width))

            deconv = self.deconv(x, shape=shape)

        if self.skip_channels > 0:
            concat = torch.cat([deconv, skip], dim=1)
        else:
            concat = deconv

        return self.conv(concat)


'''
Pose regression layer
'''
def pose_matrix(v, rotation_parameterization='exponential'):
    '''
    Convert 6 DoF parameters to transformation matrix

    Arg(s):
        v : torch.Tensor[float32]
            N x 6 vector in the order of tx, ty, tz, rx, ry, rz
        rotation_parameterization : str
            euler, exponential
    Returns:
        torch.Tensor[float32] : N x 4 x 4 homogeneous transformation matrix
    '''

    # Select N x 3 element rotation vector
    r = v[..., :3]
    # Select N x 3 element translation vector
    t = v[..., 3:]

    if rotation_parameterization in ['euler', 'exponential']:
        # Convert N x 3 element rotation vector to N x 3 x 3 rotation matrix
        if rotation_parameterization == 'euler':
            R = euler_angles(r)
        elif rotation_parameterization == 'exponential':
            R = batch_exp_map(r)

        # Convert N x 3 element translation vector to N x 3 x 1
        t = torch.unsqueeze(t, dim=-1)

        # Concatenate rotation and translation to get N x 3 x 4 transformation matrix
        Rt = torch.cat([R, t], dim=-1)

        # Convert to homogenous coordinate N x 4 x 4
        n_batch = v.shape[0]
        h = torch.tensor([0.0, 0.0, 0.0, 1.0], device=v.device) \
            .view(1, 1, 4) \
            .repeat(n_batch, 1, 1)

        Rt = torch.cat([Rt, h], dim=1)
    else:
        raise ValueError('Unsupported rotation parameterization: {}'.format(rotation_parameterization))

    return Rt


'''
Utility functions for rotation
'''
def tilde(v):
    '''
    Tilde (hat) operation

    Arg(s):
        v : torch.Tensor[float32]
            3 element rotation vector
    Returns:
        torch.Tensor[float32] : 3 x 3 skew matrix
    '''

    v1, v2, v3 = v[0], v[1], v[2]
    zero = torch.tensor(0.0, device=v.device)
    r1 = torch.stack([zero,  -v3,   v2], dim=0)
    r2 = torch.stack([  v3, zero,  -v1], dim=0)
    r3 = torch.stack([ -v2,   v1, zero], dim=0)

    return torch.stack([r1, r2, r3], dim=0)

def tilde_inv(R):
    '''
    Inverse of the tilde operation

    Arg(s):
        R : torch.Tensor[float32]
            3 x 3 inverse skew matrix
    Returns:
        torch.Tensor[float32] : 3 element rotation vector
    '''

    return 0.5 * torch.stack([R[2, 1] - R[1, 2],
                              R[0, 2] - R[2, 0],
                              R[1, 0] - R[0, 1]], dim=0)

def log_map(R):
    '''
    Logarithm map of rotation matrix element

    Arg(s):
        R : torch.Tensor[float32]
            3 x 3 rotation matrix
    Returns:
        torch.Tensor[float32] : 3 element rotation vector
    '''

    trR = 0.5 * (torch.trace(R) - 1.0)

    if trR >= 1.0:
        return tilde_inv(R)
    else:
        th = torch.acos(trR)
        return tilde_inv(R) * (th / torch.sin(th))

def exp_map(v):
    '''
    Exponential map of rotation vector using Rodrigues

    Arg(s):
        v : torch.Tensor[float32]
            3 element rotation vector
    Returns:
        torch.Tensor[float32] : 3 x 3 rotation matrix
    '''

    device = v.device
    th = torch.norm(v, p='fro')

    if th < EPSILON:
        return tilde(v) + torch.eye(3, device=device)
    else:
        sin_th = torch.sin(th)
        cos_th = torch.cos(th)
        W = tilde(v / th)
        WW = torch.matmul(W, W)
        R = sin_th * W + (1.0 - cos_th) * WW
        return R + torch.eye(3, device=device)

def batch_log_map(R):
    '''
    Applies logarithmic map for entire batch

    Arg(s):
        R : torch.Tensor[float32]
            N x 3 x 3 rotation matrices
    Returns:
        torch.Tensor[float32] : N x 3 vectors
    '''

    outputs = []
    n_batch = R.shape[0]
    for n in range(n_batch):
        outputs.append(log_map(R[n, :, :]))

    return torch.stack(outputs, dim=0)

def batch_exp_map(v):
    '''
    Applies exponential map for entire batch

    Arg(s):
        v : torch.Tensor[float32]
            N x 3 vectors
    Returns:
        torch.Tensor[float32] : N x 3 x 3 rotation matrices
    '''

    outputs = []
    n_batch = v.shape[0]
    for n in range(n_batch):
        outputs.append(exp_map(v[n, :]))

    return torch.stack(outputs, dim=0)

def euler_angles(v):
    '''
    Converts euler angles to rotation matrix

    Arg(s):
        v : torch.Tensor[float32]
            N x 3 vector of rotation angles along x, y, z axis
    Returns:
        torch.Tensor[float32] : N x 3 x 3 rotation matrix corresponding to the euler angles
    '''

    n_batch = v.shape[0]
    # Expand to B x 1 x 1
    x = torch.clamp(v[:, 0], min=-np.pi, max=np.pi).view(n_batch, 1, 1)
    y = torch.clamp(v[:, 1], min=-np.pi, max=np.pi).view(n_batch, 1, 1)
    z = torch.clamp(v[:, 2], min=-np.pi, max=np.pi).view(n_batch, 1, 1)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)

    # Rotation about x-axis
    cos_x = torch.cos(x)
    sin_x = torch.sin(x)
    rot_mat_x_row1 = torch.cat([ ones, zeros,  zeros], dim=2)
    rot_mat_x_row2 = torch.cat([zeros, cos_x, -sin_x], dim=2)
    rot_mat_x_row3 = torch.cat([zeros, sin_x,  cos_x], dim=2)
    rot_mat_x = torch.cat([rot_mat_x_row1, rot_mat_x_row2, rot_mat_x_row3], dim=1)

    # Rotation about y-axis
    cos_y = torch.cos(y)
    sin_y = torch.sin(y)
    rot_mat_y_row1 = torch.cat([ cos_y, zeros, sin_y], dim=2)
    rot_mat_y_row2 = torch.cat([ zeros,  ones, zeros], dim=2)
    rot_mat_y_row3 = torch.cat([-sin_y, zeros, cos_y], dim=2)
    rot_mat_y = torch.cat([rot_mat_y_row1, rot_mat_y_row2, rot_mat_y_row3], dim=1)

    # Rotation about z-axis
    cos_z = torch.cos(z)
    sin_z = torch.sin(z)
    rot_mat_z_row1 = torch.cat([cos_z, -sin_z, zeros], dim=2)
    rot_mat_z_row2 = torch.cat([sin_z,  cos_z, zeros], dim=2)
    rot_mat_z_row3 = torch.cat([zeros,  zeros,  ones], dim=2)
    rot_mat_z = torch.cat([rot_mat_z_row1, rot_mat_z_row2, rot_mat_z_row3], dim=1)

    return torch.matmul(torch.matmul(rot_mat_x, rot_mat_y), rot_mat_z)


'''
Utility function to pre-process sparse depth
'''
class OutlierRemoval(object):
    '''
    Class to perform outlier removal based on depth difference in local neighborhood

    Arg(s):
        kernel_size : int
            local neighborhood to consider
        threshold : float
            depth difference threshold
    '''

    def __init__(self, kernel_size=7, threshold=1.5):

        self.kernel_size = kernel_size
        self.threshold = threshold

    def remove_outliers(self, sparse_depth, validity_map):
        '''
        Removes erroneous measurements from sparse depth and validity map

        Arg(s):
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W tensor sparse depth
            validity_map : torch.Tensor[float32]
                N x 1 x H x W tensor validity map
        Returns:
            torch.Tensor[float32] : N x 1 x H x W sparse depth
            torch.Tensor[float32] : N x 1 x H x W validity map
        '''

        if self.kernel_size > 1:
            # Replace all zeros with large values
            max_value = 10 * torch.max(sparse_depth)
            sparse_depth_max_filled = torch.where(
                validity_map <= 0,
                torch.full_like(sparse_depth, fill_value=max_value),
                sparse_depth)

            # For each neighborhood find the smallest value
            padding = self.kernel_size // 2
            sparse_depth_max_filled = torch.nn.functional.pad(
                input=sparse_depth_max_filled,
                pad=(padding, padding, padding, padding),
                mode='constant',
                value=max_value)

            min_values = -torch.nn.functional.max_pool2d(
                input=-sparse_depth_max_filled,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0)

            # If measurement differs a lot from minimum value then remove
            validity_map_clean = torch.where(
                min_values < sparse_depth - self.threshold,
                torch.zeros_like(validity_map),
                torch.ones_like(validity_map))

            # Update sparse depth and validity map
            validity_map = validity_map * validity_map_clean
            sparse_depth = sparse_depth * validity_map_clean

        return sparse_depth, validity_map
