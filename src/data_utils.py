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
import os
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator


def log(s, filepath=None, to_console=True):
  '''
  Pads the filepaths based on the batch size (n_batch)
  e.g. if n_batch is 8 and number of filepaths is 14, then we pad with 2
  Args:
    filepaths : list
      list of filepaths to be read
    n_batch : int
      number of examples in a batch

  Returns:
    list : list of paths with padding
  '''
  if to_console:
    print(s)
  if filepath is not None:
    if not os.path.isdir(os.path.dirname(filepath)):
      os.makedirs(os.path.dirname(filepath))
      with open(filepath, "w+") as o:
        o.write(s+'\n')
    else:
      with open(filepath, "a+") as o:
        o.write(s+'\n')

def read_paths(filepath):
  '''
  Stores a depth map into an image (16 bit PNG)

  Args:
    path : str
      path to file where data will be stored
  '''
  path_list = []
  with open(filepath) as f:
     while True:
      path = f.readline().rstrip('\n')
      # If there was nothing to read
      if path == '':
        break
      path_list.append(path)

  return path_list

def load_depth_with_validity_map(path):
  '''
  Loads a depth map and validity map from a 16-bit PNG file

  Args:
    path : str
      path to 16-bit PNG file

  Returns:
    numpy : depth map
    numpy : binary validity map for available depth measurement locations
  '''
  # Loads depth map from 16-bit PNG file
  z = np.array(Image.open(path), dtype=np.float32)
  # Assert 16-bit (not 8-bit) depth map
  z = z/256.0
  z[z <= 0] = 0.0
  v = z.astype(np.float32)
  v[z > 0]  = 1.0
  return z, v

def load_depth(path):
  '''
  Loads a depth map from a 16-bit PNG file

  Args:
    path : str
      path to 16-bit PNG file

  Returns:
    numpy : depth map
  '''
  # Loads depth map from 16-bit PNG file
  z = np.array(Image.open(path), dtype=np.float32)
  # Assert 16-bit (not 8-bit) depth map
  z = z/256.0
  z[z <= 0] = 0.0
  return z

def save_depth(z, path):
  '''
  Saves a depth map to a 16-bit PNG file

  Args:
    z : numpy
      depth map
    path : str
      path to store depth map
  '''
  z = np.uint32(z*256.0)
  z = Image.fromarray(z, mode='I')
  z.save(path)

def load_validity_map(path):
  '''
  Loads a validity map from a 16-bit PNG file

  Args:
    path : str
      path to 16-bit PNG file

  Returns:
    numpy : binary validity map for available depth measurement locations
  '''
  # Loads depth map from 16-bit PNG file
  v = np.array(Image.open(path), dtype=np.float32)
  assert(np.all(np.unique(v) == [0, 256]))
  v[v > 0] = 1
  return v


def save_validity_map(v, path):
  '''
  Saves a validity map to a 16-bit PNG file

  Args:
    v : numpy
      validity map
    path : str
      path to store validity map
  '''
  v[v <= 0] = 0.0
  v[v > 0] = 1.0
  v = np.uint32(v*256.0)
  v = Image.fromarray(v, mode='I')
  v.save(path)

def load_calibration(path):
  '''
  Loads the calibration matrices for each camera (KITTI) and stores it as map

  Args:
    path : str
      path to file to be read

  Returns:
    dict : map containing camera intrinsics keyed by camera id
  '''
  float_chars = set("0123456789.e+- ")
  data = {}
  with open(path, 'r') as f:
    for line in f.readlines():
      key, value = line.split(':', 1)
      value = value.strip()
      data[key] = value
      if float_chars.issuperset(value):
        try:
          data[key] = np.asarray([float(x) for x in value.split(' ')])
        except ValueError:
          pass
  return data

def interpolate_depth(depth_map, validity_map, log_space=False):
  '''
  Interpolate sparse depth with barycentric coordinates

  Args:
    depth_map : np.float32
      H x W depth map
    validity_map : np.float32
      H x W depth map
    log_space : bool
      if set then produce in log space

  Returns:
    np.float32 : H x W interpolated depth map
  '''
  assert depth_map.ndim == 2 and validity_map.ndim == 2
  rows, cols = depth_map.shape
  data_row_idx, data_col_idx = np.where(validity_map)
  depth_values = depth_map[data_row_idx, data_col_idx]
  # Perform linear interpolation in log space
  if log_space:
    depth_values = np.log(depth_values)
  interpolator = LinearNDInterpolator(
      # points=Delaunay(np.stack([data_row_idx, data_col_idx], axis=1).astype(np.float32)),
      points=np.stack([data_row_idx, data_col_idx], axis=1),
      values=depth_values,
      fill_value=0 if not log_space else np.log(1e-3))
  query_row_idx, query_col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
  query_coord = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
  Z = interpolator(query_coord).reshape([rows, cols])
  if log_space:
    Z = np.exp(Z)
    Z[Z < 1e-1] = 0.0
  return Z

def compose_pose(g1, g2):
  '''
  Compose two 3 x 4 or 4 x 4 pose matrices
  Args:
    g1 : 3 x 4, or 4 x 4 numpy array
    g2 : 3 x 4, or 4 x 4 numpy array

  Returns:
    numpy : 3 x 4, or 4 x 4 pose depending on input
  '''
  g = np.concatenate(
      (g1[:3, :3].dot(g2[:3, :3]), g1[:3, :3].dot(g2[:3, [3]]) + g1[:3, [3]]),
      axis=1)

  if g1.shape[0] == 4:
    g = np.concatenate((g, [[0, 0, 0, 1]]), axis=0)

  return g

def inverse_pose(g1):
  '''
  Compute the inverse pose
    g1 = [R|t]
    then g^{-1} = [R.T| -R.T * t]
  Args:
    g1 : 3 x 4, or 4 x 4 numpy array

  Returns:
    numpy : 3 x 4, or 4 x 4 pose depending on input
  '''
  Rt = g1[:3, :3].T
  g = np.concatenate((Rt, -Rt.dot(g1[:3, [3]])), axis=1)

  if g1.shape[0] == 4:
    g = np.concatenate((g, [[0, 0, 0, 1]]), axis=0)

  return g

def pad_batch(filepaths, n_batch):
  '''
  Pads the filepaths based on the batch size (n_batch)
  e.g. if n_batch is 8 and number of filepaths is 14, then we pad with 2
  Args:
    filepaths : list
      list of filepaths to be read
    n_batch : int
      number of examples in a batch

  Returns:
    list : list of paths with padding
  '''
  n_samples = len(filepaths)
  if n_samples % n_batch > 0:
    n_pad = n_batch-(n_samples % n_batch)
    filepaths.extend([filepaths[-1]]*n_pad)
  return filepaths

def make_epoch(input_arr, n_batch):
  '''
  Generates a random order and shuffles each list in the input_arr according
  to the order
  Args:
    input_arr : list of lists
      list of lists of inputs
    n_batch : int
      number of examples in a batch

  Returns:
    list : list of lists of shuffled inputs
  '''
  assert len(input_arr)
  n = 0
  for inp in input_arr:
    if inp is not None:
      n = len(inp)
      break
  # at least one of the input arrays is not None
  assert n > 0
  idx = np.arange(n)
  n = (n // n_batch) * n_batch
  np.random.shuffle(idx)
  output_arr = []
  for inp in input_arr:
    output_arr.append(None if inp is None else [inp[i] for i in idx[:n]])
  return output_arr
