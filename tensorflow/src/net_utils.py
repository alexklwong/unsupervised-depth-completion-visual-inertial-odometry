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
import matplotlib.cm


def remove_occlusions(szvm, threshold=1.5, ksize=7):
  '''
  Removes erroneous measurements from occlusions

  Args:
    szvm : tensor
      N x H x W x 2 tensor containing sparse depth and validity map
    threshold : float
      threshold for difference in depth in a neighborhood to be considered occlusion
    ksize : float
      kernel size of neighbhorhood

  Returns:
    tensor : N x H x W x 2 tensor
  '''
  shape = szvm.get_shape().as_list()
  sz = tf.reshape(szvm[..., 0], [shape[0], shape[1], shape[2], 1])
  vm = tf.reshape(szvm[..., 1], [shape[0], shape[1], shape[2], 1])
  sz_spa = sz*vm
  max_val = tf.reduce_max(sz_spa)+100.0
  # We only care about min, so we remove all zeros by setting to max
  sz_mod = tf.where(sz_spa <= 0.0,
                    max_val*tf.ones([shape[0], shape[1], shape[2], 1]),
                    sz_spa)
  sz_mod = tf.pad(sz_mod,
                  paddings=[[0, 0], [int(ksize/2), int(ksize/2)], [int(ksize/2), int(ksize/2)], [0, 0]],
                  mode='CONSTANT',
                  constant_values=max_val)
  patches = tf.extract_image_patches(sz_mod,
      ksizes=[1, ksize, ksize, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
  sz_min = tf.reduce_min(patches, axis=-1, keepdims=True)
  # Find mark all possible occlusions as zeros
  vm_noc = tf.where(sz_min < sz_spa-threshold,
                    tf.zeros([shape[0], shape[1], shape[2], 1]),
                    tf.ones([shape[0], shape[1], shape[2], 1]))
  # Final set of valid depths are the AND relation between vm and vm no-occlusions
  vm = vm*vm_noc
  return tf.concat([sz, vm], axis=-1)

def colorize(value, colors, vmin=None, vmax=None):
  '''
  Maps a grayscale image to a matplotlib colormap

  Args:
    value : tensor
      N x H x W x 1 tensor
    vmin : float
      the minimum value of the range used for normalization
    vmax : float
      the maximum value of the range used for normalization

  Returns:
    tensor : N x H x W x 3 tensor
  '''
  # Normalize
  vmin = tf.reduce_min(value) if vmin is None else vmin
  vmax = tf.reduce_max(value) if vmax is None else vmax
  value = (value-vmin)/(vmax-vmin)

  # Squeeze last dim if it exists
  value = tf.squeeze(value)

  # Quantize
  indices = tf.to_int32(tf.round(value * 255))
  value = tf.gather(colors, indices)

  return value

def gray2color(gray, colormap):
  '''
  Converts grayscale tensor (image) to RGB using gist_stern colormap

  Args:
    gray : tensor
      N x H x W x 1 grayscale tensor
    colormap : str
      name of matplotlib color map (e.g. plasma, jet, gist_stern)

  Returns:
    tensor : N x H x W x 3 color tensor
  '''
  cm = tf.constant(matplotlib.cm.get_cmap(colormap).colors,
                       dtype=tf.float32)
  return colorize(gray, cm)
