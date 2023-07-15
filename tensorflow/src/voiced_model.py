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
import networks, loss_utils, losses, net_utils
import global_constants as settings


class VOICEDModel(object):
  def __init__(self, im0, im1, im2, sz0, kin,
               is_training=True,
               # Occlusion removal parameters
               occ_threshold=settings.OCC_THRESHOLD,
               occ_ksize=settings.OCC_KSIZE,
               # Network settings
               net_type=settings.NET_TYPE,
               im_filter_pct=settings.IM_FILTER_PCT,
               sz_filter_pct=settings.SZ_FILTER_PCT,
               # Range for depth predictions
               min_predict_z=settings.MIN_Z,
               max_predict_z=settings.MAX_Z,
               # Rotation parameterization
               rot_param=settings.ROT_PARAM,
               pose_norm=settings.POSE_NORM,
               # Loss function settings
               w_ph=settings.W_PH,
               w_co=settings.W_CO,
               w_st=settings.W_ST,
               w_sm=settings.W_SM,
               w_sz=settings.W_SZ,
               w_pc=settings.W_PC):
    # Remove occlusions from sparse depth or interpolated depth
    sz0 = net_utils.remove_occlusions(sz0, threshold=occ_threshold, ksize=occ_ksize)
    # Initialize variables
    self.im0 = im0
    self.im1 = im1
    self.im2 = im2
    self.iz0 = tf.expand_dims(sz0[..., 0], axis=-1)
    self.vm0 = tf.expand_dims(sz0[..., 1], axis=-1)
    self.sz0 = self.iz0*self.vm0
    self.kin = kin
    self.rot_param = rot_param
    self.pose_norm = pose_norm
    self.w_ph = w_ph
    self.w_co = w_co
    self.w_st = w_st
    self.w_sm = w_sm
    self.w_sz = w_sz
    self.w_pc = w_pc
    self.im_shape = self.im0.get_shape().as_list()

    '''
    Build depth prediction network
    '''
    # Select the network type
    if net_type == 'vggnet08':
      self.inv_z = networks.vggnet08(im0, sz0,
          n_output=1,
          act_fn=tf.nn.leaky_relu,
          out_fn=tf.nn.sigmoid,
          im_filter_pct=im_filter_pct,
          sz_filter_pct=sz_filter_pct)[-1]
    elif net_type == 'vggnet11':
      self.inv_z = networks.vggnet11(im0, sz0,
          n_output=1,
          act_fn=tf.nn.leaky_relu,
          out_fn=tf.nn.sigmoid,
          im_filter_pct=im_filter_pct,
          sz_filter_pct=sz_filter_pct)[-1]
    else:
      raise ValueError('Supported architectures: vggnet08, vggnet11')

    # For sigmoid (inverse depth) set the min and max depth
    self.inv_z0 = self.inv_z[0:self.im_shape[0], ...]
    self.z0 = min_predict_z/(self.inv_z0+min_predict_z/max_predict_z)

    if is_training:
      '''
      Build pose regression network
      '''
      self.pose = networks.posenet(
          tf.concat([
            tf.concat([im0, im1], axis=-1),
            tf.concat([im0, im2], axis=-1),
            tf.concat([im1, im0], axis=-1),
            tf.concat([im2, im0], axis=-1)], axis=0),
          is_training=is_training)

      if rot_param == 'euler':
        # Euler parametrization for rotation
        self.pose01, self.pose02, self.pose10, self.pose20 = [
            loss_utils.pose_vec2mat(v) for v in tf.split(self.pose, 4, axis=0)]
      elif rot_param == 'exponential':
        # Exponential parametrization for rotation
        self.pose01, self.pose02, self.pose10, self.pose20 = [
            loss_utils.pose_expm(v) for v in tf.split(self.pose, 4, axis=0)]
      # Build loss function
      self.loss = self.build_loss()

    # Prediction
    self.predict = self.z0

  def build_loss(self):
    # Select norm for pose consistency loss
    if self.pose_norm == 'frobenius':
      pose_consistency_loss_func = losses.pose_consistency_loss_func
    elif self.pose_norm == 'geodesic':
      pose_consistency_loss_func = losses.log_pose_consistency_loss_func
    else:
      raise ValueError('Supported distance to measure pose difference: frobenius, geodesic')

    '''
    Temporal (video) rigid warping
    '''
    # Compute flow from im0 to im1
    flow01 = loss_utils.compute_rigid_flow(tf.squeeze(self.z0, axis=3),
                                           pose=self.pose01,
                                           intrinsics=self.kin)
    # Compute flow from im0 to im2
    flow02 = loss_utils.compute_rigid_flow(tf.squeeze(self.z0, axis=3),
                                           pose=self.pose02,
                                           intrinsics=self.kin)
    # Reconstruct im0 using im1 with rigid flow
    im01w = tf.reshape(loss_utils.flow_warp(self.im1, flow01), self.im_shape)
    # Reconstruct im0 using im1 with rigid flow
    im02w = tf.reshape(loss_utils.flow_warp(self.im2, flow02), self.im_shape)

    '''
    Construct loss function
    '''
    vm_co = 1.0-self.vm0
    # Construct color consistency reconstruction loss
    loss_co01w = losses.color_consistency_loss_func(self.im0, im01w, vm_co)
    loss_co02w = losses.color_consistency_loss_func(self.im0, im02w, vm_co)
    loss_co = loss_co01w+loss_co02w
    # Construct structural reconstruction loss
    loss_st01w = losses.structural_loss_func(self.im0, im01w, vm_co)
    loss_st02w = losses.structural_loss_func(self.im0, im02w, vm_co)
    loss_st = loss_st01w+loss_st02w
    # Construct photometric reconstruction loss
    loss_ph = self.w_co*loss_co+self.w_st*loss_st
    # Construct sparse depth loss
    loss_sz = losses.sparse_depth_loss_func(self.z0, self.sz0, self.vm0)
    # Construct smoothness loss
    loss_sm = losses.smoothness_loss_func(self.z0, self.im0)
    # Compute pose consistency loss
    loss_pc01 = pose_consistency_loss_func(self.pose01, self.pose10)
    loss_pc02 = pose_consistency_loss_func(self.pose02, self.pose20)
    loss_pc = loss_pc01+loss_pc02

    # Construct total loss
    loss = self.w_ph*loss_ph+self.w_sm*loss_sm+self.w_sz*loss_sz+self.w_pc+loss_pc

    '''
    Construct summary for tensorboard
    '''
    for var_name in ['loss_ph', 'loss_sm', 'loss_sz', 'loss_pc']:
        tf.summary.scalar('{}'.format(var_name), eval(var_name))
    tf.summary.scalar('total_loss', loss)
    # Create a histogram of depth values
    tf.summary.histogram('z0', self.z0)
    # Visualize reconstruction
    tf.summary.image('im0_recon',
        tf.concat([self.im0, im01w, im02w], axis=1),
        max_outputs=3)
    # Visualize depth maps
    tf.summary.image('im0_z0_iz0_sz0',
        tf.concat([
          self.im0,
          net_utils.gray2color(self.z0, 'viridis'),
          net_utils.gray2color(self.iz0, 'viridis'),
          net_utils.gray2color(self.sz0, 'viridis')],
          axis=1),
        max_outputs=3)

    # Return total loss
    return loss
