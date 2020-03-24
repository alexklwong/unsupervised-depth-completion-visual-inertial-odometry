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
import so3


'''
  Loss functions for VOICED model
'''
def color_consistency_loss_func(src, tgt, v):
  '''
  Computes photometric error using L1 penalty

  Args:
    src : tensor
      source tensor that reconstructs tgt
    tgt : tensor
      target tensor
    v : tensor
      validity map for locations without sparse depth

  Returns:
    tensor : mean per pixel photometric difference
  '''
  loss = tf.reduce_sum(v*tf.abs(src-tgt), axis=[1, 2, 3], keepdims=True)
  return tf.reduce_mean(loss/tf.reduce_sum(v, axis=[1, 2, 3], keepdims=True))

def structural_loss_func(src, tgt, v):
  '''
  Computes photometric error using Structural Similarity Index (SSIM)

  Args:
    src : tensor
      source tensor that reconstructs tgt
    tgt : tensor
      target tensor
    v : tensor
      validity map for locations without sparse depth

  Returns:
    tensor : mean per pixel distance measured by SSIM
  '''
  shape = tf.shape(src)[1:3]
  dist = tf.image.resize_nearest_neighbor(ssim(src, tgt), shape)
  loss = tf.reduce_sum(v*dist, axis=[1, 2, 3], keepdims=True)
  return tf.reduce_mean(loss/tf.reduce_sum(v, axis=[1, 2, 3], keepdims=True))

def sparse_depth_loss_func(src, tgt, v):
  '''
  Computes sparse depth reconstruction loss with L1 penalty

  Args:
    src : tensor
      source depth tensor that reconstructs tgt
    tgt : tensor
      target depth tensor
    v : tensor
      validity map for locations with sparse depth

  Returns:
    tensor : mean per pixel depth difference in valid locations
  '''
  return tf.reduce_sum(tf.abs(v*src-v*tgt))/tf.reduce_sum(v)

def pose_consistency_loss_func(pose0, pose1):
  '''
  Compute L2 distance between the composition of pose0 and pose1 and
  the identity matrix (pose0 . pose1 should be the identity)

  Args:
    pose0 : tensor
      N x 4 x 4 tensor representing the pose of the camera
    pose1:
      N x 4 x 4 tensor representing the pose of the camera

  Returns:
    tensor : mean difference between the rotations and tranlations of the two poses
  '''
  shape = pose0.get_shape().as_list()
  eye = tf.eye(num_rows=shape[1], num_columns=shape[2], batch_shape=[shape[0]])
  return tf.reduce_mean(tf.square(tf.matmul(pose0, pose1)-eye))

def log_pose_consistency_loss_func(pose0, pose1):
  '''
  Compute l2 distance in tangent space (Lie algebra) of rotation groups

  Args:
    pose0 : tensor
      N x 4 x 4 tensor representing the pose of the camera
    pose1:
      N x 4 x 4 tensor representing the pose of the camera

  Returns:
    tensor : mean difference between the rotations and tranlations of the two poses
  '''
  g = tf.matmul(pose0, pose1)
  R = g[..., :3, :3]
  t = g[..., :3, 3]
  omega = so3.batch_log(R)
  return tf.reduce_mean(tf.square(omega)) + tf.reduce_mean(tf.square(t))


def smoothness_loss_func(predict, data):
  '''
  Computes local smoothness loss using L1 penalty
  Penalty is weighted by image gradients (higher gradients, lower weight)

  Args:
    predict : tensor
      model prediction
    data : tensor
      RGB image

  Returns:
    tensor : mean x-y prediction gradients weighted by data
  '''
  pred_gy, pred_gx = gradient_yx(predict)
  data_gy, data_gx = gradient_yx(data)
  w_x = tf.exp(-tf.reduce_mean(tf.abs(data_gx), 3, keepdims=True))
  w_y = tf.exp(-tf.reduce_mean(tf.abs(data_gy), 3, keepdims=True))

  smoothness_x = tf.reduce_mean(tf.abs(pred_gx)*w_x)
  smoothness_y = tf.reduce_mean(tf.abs(pred_gy)*w_y)
  return smoothness_x+smoothness_y


'''
  Util functions for constructing losses
'''
def ssim(A, B):
  '''
  Computes Structural Similarity Index (SSIM) which decomposes color
  into luminance, contrast and structure

  Args:
    src : tensor
      source tensor that reconstructs tgt
    tgt : tensor
      target tensor

  Returns:
    tensor : per pixel distance measured by SSIM
  '''
  C1 = 0.01**2
  C2 = 0.03**2
  mu_A = slim.avg_pool2d(A, 3, 1, 'VALID')
  mu_B = slim.avg_pool2d(B, 3, 1, 'VALID')
  sigma_A  = slim.avg_pool2d(A**2, 3, 1, 'VALID')-mu_A**2
  sigma_B  = slim.avg_pool2d(B**2, 3, 1, 'VALID')-mu_B**2
  sigma_AB = slim.avg_pool2d(A*B , 3, 1, 'VALID')-mu_A*mu_B
  numer = (2*mu_A*mu_B+C1)*(2*sigma_AB+C2)
  denom = (mu_A**2+mu_B**2+C1)*(sigma_A+sigma_B+C2)
  score = numer/denom
  return tf.clip_by_value((1-score)/2, 0, 1)

def gradient_yx(T):
  '''
  Computes x and y gradients of a tensor

  Args:
    T : tensor
      input tensor to compute gradients

  Returns:
    tensor : y gradients of T
    tensor : x gradients of T
  '''
  gx = T[:, :, :-1, :]-T[:, :, 1:, :]
  gy = T[:, :-1, :, :]-T[:, 1:, :, :]
  return gy, gx
