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
import numpy as np


def root_mean_sq_err(z, gt):
  '''
  Root mean squared error

  Args:
    z : numpy
      depth map
    gt : numpy
      ground truth depth

  Returns:
    float : root mean squared error
  '''
  return np.sqrt(np.mean((gt-z) ** 2))

def mean_abs_err(z, gt):
  '''
  Mean absolute error

  Args:
    z : numpy
      depth map
    gt : numpy
      ground truth  depth

  Returns:
    float : mean absolute error
  '''
  return np.mean(np.abs(gt-z))

def inv_root_mean_sq_err(z, gt):
  '''
  Inverse root mean squared error

  Args:
    z : numpy
      depth map
    gt : numpy
      ground truth  depth

  Returns:
    float : Inverse root mean squared error
  '''
  return np.sqrt(np.mean((1.0/gt-1.0/z)**2))

def inv_mean_abs_err(z, gt):
  '''
  Inverse mean absolute error

  Args:
    z : numpy
      depth map
    gt : numpy
      ground truth  depth

  Returns:
    float : inverse mean absolute error
  '''
  return np.mean(np.abs(1.0/gt-1.0/z))
