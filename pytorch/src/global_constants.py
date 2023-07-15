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

# Dataloader settings
N_BATCH                             = 8
N_HEIGHT                            = 320
N_WIDTH                             = 768
N_CHANNEL                           = 3
AUGMENTATION_RANDOM_CROP_TYPE       = ['horizontal', 'anchor', 'bottom']

# Network settings
ENCODER_TYPE_AVAILABLE              = ['vggnet08', 'vggnet11']
DECONV_TYPE_AVAILABLE               = ['transpose', 'up']
WEIGHT_INITIALIZER_AVAILABLE        = ['kaiming_normal',
                                       'kaiming_uniform',
                                       'xavier_normal',
                                       'xavier_uniform']
ACTIVATION_FUNC_AVAILABLE           = ['relu', 'leaky_relu', 'elu']

ENCODER_TYPE                        = ['vggnet08']
N_FILTERS_ENCODER_IMAGE             = [48, 96, 192, 384, 384]
N_FILTERS_ENCODER_DEPTH             = [16, 32, 64, 128, 128]
DECODER_TYPE                        = ['multi-scale']
N_FILTERS_DECODER                   = [256, 128, 128, 64, 32]
DECONV_TYPE                         = 'up'
WEIGHT_INITIALIZER                  = 'kaiming_uniform'
ACTIVATION_FUNC                     = 'leaky_relu'

# Training settings
N_EPOCH                             = 60
LEARNING_RATES                      = [1.00e-4, 0.50e-4, 0.25e-4]
LEARNING_SCHEDULE                   = [20, 40]
W_COLOR                             = 0.20
W_STRUCTURE                         = 0.80
W_SPARSE_DEPTH                      = 0.20
W_SMOOTHNESS                        = 0.01
W_POSE                              = 0.00
W_WEIGHT_DECAY_DEPTH                = 0.00
W_WEIGHT_DECAY_POSE                 = 1e-4

# Depth range settings
EPSILON                             = 1e-10
OUTLIER_REMOVAL_METHOD_AVAILABLE    = ['remove', 'set_to_min']
OUTLIER_REMOVAL_METHOD              = 'remove'
OUTLIER_REMOVAL_KERNEL_SIZE         = 7
OUTLIER_REMOVAL_THRESHOLD           = 1.5
MIN_PREDICT_DEPTH                   = 1e-3
MAX_PREDICT_DEPTH                   = 655.0
MIN_EVALUATE_DEPTH                  = 1e-3
MAX_EVALUATE_DEPTH                  = 655.0

# Rotation parameterization
ROTATION_PARAMETERIZATION           = 'exponential'

# Checkpoint settings
N_DISPLAY                           = 4
N_SUMMARY                           = 1000
N_CHECKPOINT                        = 5000
CHECKPOINT_PATH                     = ''
RESTORE_PATH                        = ''

# Hardware settings
DEVICE                              = 'cuda'
CUDA                                = 'cuda'
CPU                                 = 'cpu'
GPU                                 = 'gpu'
N_THREAD                            = 8
