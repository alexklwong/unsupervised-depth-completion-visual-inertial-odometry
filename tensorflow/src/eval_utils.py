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
import cv2
import numpy as np
import data_utils


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

def rigid_transform_3d(A, B):
    '''
    Aligns two point clouds by computing the relative transformation from A to B

    Args:
      A : numpy
        N x 3 matrix of 3D points
      B : numpy
        N x 3 matrix of 3D points

    Returns:
      numpy : 3 x 3 rotation matrix R
      numpy : 3 x 1 translation matrix t
    '''

    assert len(A) == len(B)

    N = A.shape[0]
    A_centroid = np.mean(A, axis=0)
    B_centroid = np.mean(B, axis=0)

    # Center the point clouds
    A_centered = A - np.tile(A_centroid, (N, 1))
    B_centered = B - np.tile(B_centroid, (N, 1))

    # Compute covariance and solve
    H = np.matmul(np.transpose(A_centered), B_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # Special reflection case
    if np.linalg.det(R) < 0:
      Vt[2, :] *= -1
      R = np.matmul(Vt.T, U.T)

    t = np.matmul(-R, A_centroid.T) + B_centroid.T

    return R, t

def rel_rotation_err(pose_src, pose_tgt):
  '''
  Computes relative rotation error (RRE)

  Args:
    pose_src : numpy
      N x 3 x 4 source pose matrix relative to 0-th frame
    pose_tgt : numpy
      N x 3 x 4 target (ground truth) pose matrix relative to 0-th frame

  Returns:
    float : relative rotation error
  '''

  assert pose_src.shape == pose_tgt.shape

  rre = []

  for t in range(0, pose_src.shape[0]-1):

    # Get relative pose from t to t+1
    pose01_tgt = data_utils.compose_pose(
      data_utils.inverse_pose(pose_tgt[t]),
      pose_tgt[t+1])

    pose01_src = data_utils.compose_pose(
      data_utils.inverse_pose(pose_src[t]),
      pose_src[t+1])

    # Compute change in pose
    delta_pose = data_utils.compose_pose(
        data_utils.inverse_pose(pose01_tgt),
        pose01_src)

  # Compute RRE (Relative Rotation Error)
  omega, _ = cv2.Rodrigues(delta_pose[:3, :3])
  rre.append((180.0 / np.pi * np.linalg.norm(omega)) ** 2)

  return np.sqrt(np.mean(rre))

def rel_pose_err(pose_src, pose_tgt):
  '''
  Computes relative pose error (RPE)

  Args:
    pose_src : numpy
      N x 3 x 4 source pose matrix relative to 0-th frame
    pose_tgt : numpy
      N x 3 x 4 target (ground truth) pose matrix relative to 0-th frame

  Returns:
    float : relative pose error
  '''

  assert pose_src.shape == pose_tgt.shape

  rpe = []

  for t in range(0, pose_src.shape[0]-1):

    # Get relative pose from t to t+1
    pose01_tgt = data_utils.compose_pose(
      data_utils.inverse_pose(pose_tgt[t]),
      pose_tgt[t+1])

    pose01_src = data_utils.compose_pose(
      data_utils.inverse_pose(pose_src[t]),
      pose_src[t+1])

    # Compute change in pose
    delta_pose = data_utils.compose_pose(
        data_utils.inverse_pose(pose01_tgt),
        pose01_src)

    # Compute RPE (Relative Pose Error)
    rpe.append(np.linalg.norm(delta_pose[:3, 3]) ** 2)

  return np.sqrt(np.mean(rpe))

def abs_trajectory_err(pose_src, pose_tgt):
  '''
  Computes absolute trajectory error (ATE)

  Args:
    pose_src : numpy
      N x 3 x 4 source pose matrix relative to 0-th frame
    pose_tgt : numpy
      N x 3 x 4 target (ground truth) pose matrix relative to 0-th frame

  Returns:
    float : absolute trajectory error
  '''

  assert pose_src.shape == pose_tgt.shape

  ate = []

  for t in range(pose_src.shape[0]):

    # Compute ATE (Absolute Trajectory Error)
    delta_pose = data_utils.compose_pose(
      data_utils.inverse_pose(pose_tgt[t]),
      pose_src[t])

    ate.append(np.linalg.norm(delta_pose[:3, 3]) ** 2)

  return np.sqrt(np.mean(ate))
