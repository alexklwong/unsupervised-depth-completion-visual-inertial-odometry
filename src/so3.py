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
import global_constants as settings


def tilde(v):
  '''
  Tilde (hat) operation

  Args:
    v : tensor
      3 element vector

  Returns:
    skew matrix
  '''
  with tf.name_scope('tilde'):
    v1, v2, v3 = v[0], v[1], v[2]
    r1 = tf.stack([0, -v3, v2], axis=0)
    r2 = tf.stack([v3, 0, -v1], axis=0)
    r3 = tf.stack([-v2, v1, 0], axis=0)
    return tf.stack([r1, r2, r3], axis=0)

def tilde_inv(R):
  '''
  Inverse of the tilde operation

  R : tensor
    3 x 3 inverse skew matrix

  Returns:
    3-dim vector
  '''
  return 0.5 * tf.stack([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]])

def log(R):
  '''
  Logarithm map of rotation matrix element

  Args:
    R : 3x3 rotation matrix

  Returns:
    3-dim rotation vector
  '''
  with tf.name_scope('invrodrigues'):
    trR = 0.5 * (tf.trace(R) - 1)

    def true_fn():
      return tilde_inv(R)

    def false_fn():
      th = tf.acos(trR)
      v = tilde_inv(R) * (th / tf.sin(th))
      return v

    return tf.cond(trR >= 1.0, true_fn, false_fn)

def exp(v):
  '''
  Exponential map of rotation vector.

  Args:
   v : tensor
    3-dim rotation vector

  Returns:
    3 x 3 rotation matrix
  '''
  with tf.name_scope('rodrigues'):
    th = tf.norm(v)

    def true_fn():
      return tilde(v)

    def false_fn():
        sin_th = tf.sin(th)
        cos_th = tf.cos(th)
        W = tilde(v / th)
        WW = tf.matmul(W, W)
        R = sin_th * W + (1-cos_th) * WW
        return R

    R = tf.cond(th < settings.EPSILON, true_fn, false_fn) + tf.diag([1., 1., 1.])
    return R

def batch_log(R):
    return tf.map_fn(log, R)

def batch_exp(v):
    return tf.map_fn(exp, v)
