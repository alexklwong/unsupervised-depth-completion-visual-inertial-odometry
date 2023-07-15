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
import tensorflow as tf
import tensorflow.contrib.slim as slim


'''
  Util for creating network layers and blocks
'''
def vgg2d(x, ksize, n_conv=2, stride=2, padding='SAME', act_fn=tf.nn.relu,
          reuse=tf.AUTO_REUSE, name=None):
  '''
  Creates a VGG block of n_conv layers

  Args:
    x : tensor
      input tensor
    ksize : list
      3 x 1 list [k, k, f] of kernel size k, number of filters f
    nconv : int
      number of total convolutions
    stride : int
      stride size for downsampling
    padding : str
      padding on edges in case size doesn't match
    act_fn : func
      activation function after convolution
    reuse : bool
      if set, reuse weights if have already been defined in same variable scope
    name : str
      name of node in computational graph

  Returns:
    tensor : layer after VGG convolutions
  '''
  name = name+'_vgg2d_' if name is not None else 'vgg2d_'
  layers = [x]

  for n in range(n_conv-1):
    layers.append(slim.conv2d(layers[-1], num_outputs=ksize[2], kernel_size=ksize[0:2], stride=1,
                              padding=padding, activation_fn=act_fn, reuse=reuse, scope=name+'conv'+str(n+1)))

  convn = slim.conv2d(layers[-1], num_outputs=ksize[2], kernel_size=ksize[0:2], stride=stride,
                      padding=padding, activation_fn=act_fn, reuse=reuse, scope=name+'conv'+str(n_conv))
  return convn

def upconv2d(x, shape, ksize, stride=2, padding='SAME', act_fn=tf.nn.relu,
             reuse=tf.AUTO_REUSE, name=None):
  '''
  Creates a 2D up-convolution layer upsample and convolution

  Args:
    x : tensor
      input tensor
    shape : list
      2 element list of tensor y-x shape
    ksize : list
      3 x 1 list [k, k, f] of kernel size k, number of filters f
    stride : int
      stride size of convolution
    padding : str
      padding on edges in case size doesn't match
    act_fn : func
      activation function after convolution
    reuse : bool
      if set, reuse weights if have already been defined in same variable scope
    name : str
      name of node in computational graph

  Returns:
    tensor : layer after performing up-convolution
  '''
  name = name if name is not None else ''
  x_up = tf.image.resize_nearest_neighbor(x, shape)
  conv = slim.conv2d(x_up, num_outputs=ksize[2], kernel_size=ksize[0:2], stride=stride,
                     padding=padding, activation_fn=act_fn, reuse=reuse, scope=name+'upconv')
  return conv


'''
  Network architectures
'''
def posenet(data, is_training):
  '''
  Creates a pose network that predicts 6 degrees of freedom pose

  Args:
    data : tensor
      input data N x H x W x D
    is_training : bool
      if set then network is training (matters on using batch norm, but is
      better to be explicit)

  Returns:
    tensor : 6 degrees of freedom pose
  '''
  batch_norm_params = { 'is_training': is_training }
  with tf.variable_scope('posenet'):
    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        activation_fn=tf.nn.relu):
      conv1  = slim.conv2d(data,  16,  7, 2)
      conv2  = slim.conv2d(conv1, 32,  5, 2)
      conv3  = slim.conv2d(conv2, 64,  3, 2)
      conv4  = slim.conv2d(conv3, 128, 3, 2)
      conv5  = slim.conv2d(conv4, 256, 3, 2)
      conv6  = slim.conv2d(conv5, 256, 3, 2)
      conv7  = slim.conv2d(conv6, 256, 3, 2)
      pose   = slim.conv2d(conv7, 6, 1, 1,
                           normalizer_fn=None, activation_fn=None)
      pose_mean = tf.reduce_mean(pose, [1, 2])
      return 0.01*tf.reshape(pose_mean, [-1, 6])

def vggnet08(im, sz, n_output=1, act_fn=tf.nn.relu, out_fn=tf.identity,
             im_filter_pct=0.75, sz_filter_pct=0.25):
  '''
  Creates a VGG08 late fusion network

  Args:
    im : tensor
      input image (N x H x W x D)
    sz : tensor
      input sparse depth with valid map (N x H x W x 2)
    act_fn : func
      activation function after convolution
    out_fn : func
      activation function to produce output predictions
    im_filter_pct : float
      percent of parameters to allocate to the image branch
    sz_filter_pct : float
      percent of parameters to allocate to the sparse depth branch
    reuse_vars : bool
      if set, reuse weights if have already been defined in same variable scope

  Returns:
    list : list containing prediction and upsampled prediction at original resolution
  '''
  with tf.variable_scope('vggnet08'):
    shape = im.get_shape().as_list()[1:3]
    layers, skips = vggnet_encoder(im, sz,
        n_mod1=1, n_mod2=1, n_mod3=1, n_mod4=1, n_mod5=1,
        act_fn=act_fn,
        im_filter_pct=im_filter_pct,
        sz_filter_pct=sz_filter_pct)
    layers, outputs = decoder(layers, skips, shape,
        n_output=n_output,
        act_fn=act_fn,
        out_fn=out_fn)
    return outputs

def vggnet11(im, sz, n_output=1, act_fn=tf.nn.relu, out_fn=tf.identity,
             im_filter_pct=0.75, sz_filter_pct=0.25):
  '''
  Creates a VGG11 late fusion network

  Args:
    im : tensor
      input image (N x H x W x D)
    sz : tensor
      input sparse depth with valid map (N x H x W x 2)
    act_fn : func
      activation function after convolution
    out_fn : func
      activation function to produce output predictions
    im_filter_pct : float
      percent of parameters to allocate to the image branch
    sz_filter_pct : float
      percent of parameters to allocate to the sparse depth branch
    reuse_vars : bool
      if set, reuse weights if have already been defined in same variable scope

  Returns:
    list : list containing prediction and upsampled prediction at original resolution
  '''
  with tf.variable_scope('vggnet11'):
    shape = im.get_shape().as_list()[1:3]
    layers, skips = vggnet_encoder(im, sz,
        n_mod1=1, n_mod2=1, n_mod3=2, n_mod4=2, n_mod5=2,
        act_fn=act_fn,
        im_filter_pct=im_filter_pct,
        sz_filter_pct=sz_filter_pct)
    layers, outputs = decoder(layers, skips, shape,
        n_output=n_output,
        act_fn=act_fn,
        out_fn=out_fn)
    return outputs


'''
  Network encoder and decoder structures
'''
def vggnet_encoder(im, sz, n_mod1=1, n_mod2=2, n_mod3=2, n_mod4=2, n_mod5=2,
                   im_filter_pct=0.75, sz_filter_pct=0.25,
                   act_fn=tf.nn.relu, reuse_vars=tf.AUTO_REUSE):
  '''
  Creates an early or late fusion (two branches, one for processing image and the other depth)
  VGGnet encoder with 5 VGG blocks each with resolution 1 -> 1/32

  Args:
    im : tensor
      input image (N x H x W x D)
    sz : tensor
      input sparse depth with valid map (N x H x W x 2)
    n_mod<n> : int
      number of convolutional layers to perform in nth VGG block
    im_filter_pct : float
      percent of parameters to allocate to the image branch
    sz_filter_pct : float
      percent of parameters to allocate to the sparse depth branch
    act_fn : func
      activation function after convolution
    reuse_vars : bool
      if set, reuse weights if have already been defined in same variable scope

  Returns:
    list : list containing all the layers last element is the
      latent representation (1/32 resolution)
    list : list containing all the skip connections
  '''
  layers = []
  skips  = []
  with tf.variable_scope('enc0', reuse=reuse_vars):
    ksize = [5, 5, 64]
    e0_convi = vgg2d(im, ksize=ksize[0:2]+[int(im_filter_pct*ksize[2])], n_conv=n_mod1,
                     padding='SAME', act_fn=act_fn, reuse=reuse_vars, name='im')
    e0_convz = vgg2d(sz, ksize=ksize[0:2]+[int(sz_filter_pct*ksize[2])], n_conv=n_mod1,
                     padding='SAME', act_fn=act_fn, reuse=reuse_vars, name='sz')
    layers.append(e0_convz)
    layers.append(e0_convi)
    skips.append(tf.concat([e0_convz, e0_convi], axis=-1))

  with tf.variable_scope('enc1', reuse=reuse_vars):
    ksize = [3, 3, 128]
    e1_convi = vgg2d(layers[-1], ksize=ksize[0:2]+[int(im_filter_pct*ksize[2])], n_conv=n_mod2,
                      padding='SAME', act_fn=act_fn, reuse=reuse_vars, name='im')
    e1_convz = vgg2d(layers[-2], ksize=ksize[0:2]+[int(sz_filter_pct*ksize[2])], n_conv=n_mod2,
                      padding='SAME', act_fn=act_fn, reuse=reuse_vars, name='sz')
    layers.append(e1_convz)
    layers.append(e1_convi)
    skips.append(tf.concat([e1_convz, e1_convi], axis=-1))

  with tf.variable_scope('enc2', reuse=reuse_vars):
    ksize = [3, 3, 256]
    e2_convi = vgg2d(layers[-1], ksize=ksize[0:2]+[int(im_filter_pct*ksize[2])], n_conv=n_mod3,
                      padding='SAME', act_fn=act_fn, reuse=reuse_vars, name='im')
    e2_convz = vgg2d(layers[-2], ksize=ksize[0:2]+[int(sz_filter_pct*ksize[2])], n_conv=n_mod3,
                      padding='SAME', act_fn=act_fn, reuse=reuse_vars, name='sz')
    layers.append(e2_convz)
    layers.append(e2_convi)
    skips.append(tf.concat([e2_convz, e2_convi], axis=-1))

  with tf.variable_scope('enc3', reuse=reuse_vars):
    ksize = [3, 3, 512]
    e3_convi = vgg2d(layers[-1], ksize=ksize[0:2]+[int(im_filter_pct*ksize[2])], n_conv=n_mod4,
                      padding='SAME', act_fn=act_fn, reuse=reuse_vars, name='im')
    e3_convz = vgg2d(layers[-2], ksize=ksize[0:2]+[int(sz_filter_pct*ksize[2])], n_conv=n_mod4,
                      padding='SAME', act_fn=act_fn, reuse=reuse_vars, name='sz')
    layers.append(e3_convz)
    layers.append(e3_convi)
    skips.append(tf.concat([e3_convz, e3_convi], axis=-1))

  with tf.variable_scope('enc4', reuse=reuse_vars):
    ksize = [3, 3, 512]
    e4_convi = vgg2d(layers[-1], ksize=ksize[0:2]+[int(im_filter_pct*ksize[2])], n_conv=n_mod5,
                      padding='SAME', act_fn=act_fn, reuse=reuse_vars, name='im')
    e4_convz = vgg2d(layers[-2], ksize=ksize[0:2]+[int(sz_filter_pct*ksize[2])], n_conv=n_mod5,
                      padding='SAME', act_fn=act_fn, reuse=reuse_vars, name='sz')
    layers.append(tf.concat([e4_convz, e4_convi], axis=-1))

  return layers, skips

def decoder(layer, skips, shape, n_output=1, act_fn=tf.nn.relu, out_fn=tf.identity,
            reuse_vars=tf.AUTO_REUSE):
  '''
  Creates a decoder with up-convolves the latent representation to 5 times the
  resolution from 1/32 -> 1

  Args:
    layer : tensor (or list)
      N x H x W x D latent representation (will also handle list of layers
      for backwards compatibility)
    skips : list
      list of skip connections
    shape : list
      [H W] list of dimensions for the final output
    n_output : int
      number of channels in output
    act_fn : func
      activation function after convolution
    out_fn : func
      activation function to produce output predictions
    reuse_vars : bool
      if set, reuse weights if have already been defined in same variable scope

  Returns:
    list : list containing all layers
    list : list containing prediction and upsampled prediction at original resolution
  '''
  layers = layer if isinstance(layer, list) else [layer]
  outputs = []
  with tf.variable_scope('dec4', reuse=reuse_vars):
    ksize = [3, 3, 256]
    # Perform up-convolution
    d4_upconv = upconv2d(layers[-1], shape=skips[3].get_shape().as_list()[1:3], ksize=ksize, stride=1,
                         act_fn=act_fn, reuse=tf.AUTO_REUSE)
    # Concatenate with skip connection
    d4_concat = tf.concat([d4_upconv, skips[3]], axis=-1)
    # Convolve again
    layers.append(slim.conv2d(d4_concat, num_outputs=ksize[2], kernel_size=ksize[0:2], stride=1,
                              activation_fn=act_fn, padding='SAME', reuse=reuse_vars,
                              scope='conv1'))

  with tf.variable_scope('dec3', reuse=reuse_vars):
    ksize = [3, 3, 128]
    # Perform up-convolution
    d3_upconv = upconv2d(layers[-1], shape=skips[2].get_shape().as_list()[1:3], ksize=ksize, stride=1,
                         act_fn=act_fn, reuse=tf.AUTO_REUSE, name='upconv')
    # Concatenate with skip connection
    d3_concat = tf.concat([d3_upconv, skips[2]], axis=-1)
    # Convolve again
    layers.append(slim.conv2d(d3_concat, num_outputs=ksize[2], kernel_size=ksize[0:2], stride=1,
                              activation_fn=act_fn, padding='SAME', reuse=reuse_vars,
                              scope='conv1'))

  with tf.variable_scope('dec2', reuse=reuse_vars):
    ksize = [3, 3, 128]
    # Perform up-convolution
    d2_upconv = upconv2d(layers[-1], shape=skips[1].get_shape().as_list()[1:3], ksize=ksize, stride=1,
                         act_fn=act_fn, reuse=tf.AUTO_REUSE, name='upconv')
    # Concatenate with skip connection
    d2_concat = tf.concat([d2_upconv, skips[1]], axis=-1)
    # Convolve again
    layers.append(slim.conv2d(d2_concat, num_outputs=ksize[2], kernel_size=ksize[0:2], stride=1,
                              activation_fn=act_fn, padding='SAME', reuse=reuse_vars,
                              scope='conv1'))

  with tf.variable_scope('dec1', reuse=reuse_vars):
    ksize = [3, 3, 64]
    # Perform up-convolution
    d1_upconv = upconv2d(layers[-1], shape=skips[0].get_shape().as_list()[1:3], ksize=ksize, stride=1,
                         act_fn=act_fn, reuse=tf.AUTO_REUSE, name='upconv')
    # Concatenate with skip connection
    d1_concat = tf.concat([d1_upconv, skips[0]], axis=-1)
    # Convolve again
    outputs.append(slim.conv2d(d1_concat, num_outputs=n_output, kernel_size=ksize[0:2], stride=1,
                               activation_fn=out_fn, padding='SAME', reuse=reuse_vars,
                               scope='output'))

  with tf.variable_scope('dec0', reuse=reuse_vars):
    # Perform up-sampling to original resolution
    d0_upout = tf.reshape(
        tf.image.resize_nearest_neighbor(outputs[-1], shape),
        [-1, shape[0], shape[1], 1])
    layers.append(d0_upout)
    outputs.append(d0_upout)

  return layers, outputs
