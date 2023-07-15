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
import tensorflow as tf
import data_utils
import global_constants as settings


class DataLoader(object):

  def __init__(self, shape,
               name=None,
               is_training=True,
               normalize=True,
               n_thread=settings.N_THREAD,
               prefetch_size=settings.N_THREAD):
    self.n_batch = shape[0]
    self.n_height = shape[1]
    self.n_width = shape[2]
    self.n_channel = shape[3]
    self.is_training = is_training
    self.normalize = normalize
    self.n_thread = n_thread
    self.prefetch_size = prefetch_size

    self.scope_name = name if name is not None else 'dataloader'
    with tf.variable_scope(self.scope_name):
      # Set up placeholders for entry
      self.image_placeholder = tf.placeholder(tf.string, shape=[None])
      self.interp_depth_placeholder = tf.placeholder(tf.string, shape=[None])
      self.validity_map_placeholder = tf.placeholder(tf.string, shape=[None])
      # If we are training we also need intrinsics
      if is_training:
        self.intrinsics_placeholder = tf.placeholder(tf.string, shape=[None])
        self.crop_placeholder = tf.placeholder(tf.bool, shape=())

      if is_training:
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.image_placeholder, self.interp_depth_placeholder,
              self.validity_map_placeholder, self.intrinsics_placeholder))
      else:
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.image_placeholder, self.interp_depth_placeholder,
              self.validity_map_placeholder))

      if is_training:
        self.dataset = self.dataset \
            .map(self._load_func, num_parallel_calls=self.n_thread) \
            .map(self._crop_func, num_parallel_calls=self.n_thread) \
            .batch(self.n_batch) \
            .prefetch(buffer_size=self.prefetch_size)
      else:
        self.dataset = self.dataset \
            .map(self._load_func, num_parallel_calls=self.n_thread) \
            .batch(self.n_batch) \
            .prefetch(buffer_size=self.prefetch_size)

      self.iterator = self.dataset.make_initializable_iterator()
      self.next_element = self.iterator.get_next()

      # Image at time 0
      self.next_element[0].set_shape(
          [self.n_batch, self.n_height, self.n_width, self.n_channel])
      # Image at time 1
      self.next_element[1].set_shape(
          [self.n_batch, self.n_height, self.n_width, self.n_channel])
      # Image at time 2
      self.next_element[2].set_shape(
          [self.n_batch, self.n_height, self.n_width, self.n_channel])
      # Depth input at time 0
      self.next_element[3].set_shape(
          [self.n_batch, self.n_height, self.n_width, 2])
      if is_training:
        # Camera intrinsics 3x3 matrix
        self.next_element[4].set_shape([self.n_batch, 3, 3])

  def _load_func(self, image_path, interp_depth_path, validity_map_path, intrinsics_path=None):
    with tf.variable_scope('load_func'):
      im0, im1, im2 = self._load_image_composite_func(image_path)
      iz0 = self._load_depth_func(interp_depth_path)
      vm0 = self._load_validity_map_func(validity_map_path)
      sz0 = tf.concat([
          tf.expand_dims(iz0, axis=-1),
          tf.expand_dims(vm0, axis=-1)], axis=-1)

      if self.is_training:
        # Load camera intrinsics
        kin = self._load_kin_func(intrinsics_path)
        return im0, im1, im2, sz0, kin
      else:
        return im0, im1, im2, sz0

  def _crop_func(self, *args):

    def crop_func(in0, in1, in2, in3, k):
      # Bottom center crop to specified height and width instead of resize
      shape = tf.shape(in0)
      o_height = shape[0]-self.n_height
      o_width = tf.to_int32(tf.to_float(shape[1]-self.n_width)/tf.to_float(2.0))
      in0 = in0[o_height:shape[0], o_width:self.n_width+o_width, :]
      in1 = in1[o_height:shape[0], o_width:self.n_width+o_width, :]
      in2 = in2[o_height:shape[0], o_width:self.n_width+o_width, :]
      in3 = in3[o_height:shape[0], o_width:self.n_width+o_width, :]
      # Adjust camera intrinsics after crop
      k_adj = tf.to_float([[0, 0, -o_width ],
                           [0, 0, -o_height],
                           [0, 0, 0        ]])
      k = k+k_adj
      return in0, in1, in2, in3, k

    with tf.variable_scope('crop_func'):
      im0, im1, im2, sz0, kin = args

      im0, im1, im2, sz0, kin = tf.cond(self.crop_placeholder,
          lambda: crop_func(im0, im1, im2, sz0, kin),
          lambda: (im0, im1, im2, sz0, kin))

      return im0, im1, im2, sz0, kin

  def _load_image_composite_func(self, path):
    with tf.variable_scope('load_image_composite_func'):
      imc = tf.to_float(tf.image.decode_png(tf.read_file(path)))
      im1, im0, im2 = tf.split(imc, num_or_size_splits=3, axis=1)
      if self.normalize:
        im1 = im1/255.0
        im0 = im0/255.0
        im2 = im2/255.0

      return tf.squeeze(im0), tf.squeeze(im1), tf.squeeze(im2)

  def _load_depth_func(self, path):
    with tf.variable_scope('load_depth_func'):
      z = tf.py_func(data_utils.load_depth, [path], [tf.float32])
      return tf.squeeze(z)

  def _load_validity_map_func(self, path):
    with tf.variable_scope('load_validity_map_func'):
      v = tf.py_func(data_utils.load_validity_map, [path], [tf.float32])
      return tf.squeeze(v)

  def _load_kin_func(self, path):
    with tf.variable_scope('load_kin_func'):
      return tf.reshape(self._load_npy_func(path), [3, 3])

  def _load_npy_func(self, path):
    with tf.variable_scope('load_npy_func'):
      return tf.to_float(tf.py_func(
          lambda path: np.load(path.decode()).astype(np.float32), [path], [tf.float32]))

  def initialize(self, session,
                 image_paths=None,
                 interp_depth_paths=None,
                 validity_map_paths=None,
                 intrinsics_paths=None,
                 do_crop=False):

    assert session is not None

    if self.is_training:
      feed_dict = {
        self.image_placeholder        : image_paths,         # Images at time 0, 1 and 2
        self.interp_depth_placeholder : interp_depth_paths,  # Sparse depth at time 0
        self.validity_map_placeholder : validity_map_paths,  # Validity map at time 0
        self.intrinsics_placeholder   : intrinsics_paths,    # Camera intrinsics
        self.crop_placeholder         : do_crop              # Perform center crop on images and sparse depth
      }
    else:
      feed_dict = {
        self.image_placeholder        : image_paths,         # Images at time 0, 1 and 2
        self.interp_depth_placeholder : interp_depth_paths,  # Sparse depth at time 0
        self.validity_map_placeholder : validity_map_paths   # Validity map at time 0
      }

    session.run(self.iterator.initializer, feed_dict)
