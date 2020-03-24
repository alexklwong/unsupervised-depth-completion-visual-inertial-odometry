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
import argparse
import global_constants as settings
from voiced_main import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_image_path',
    type=str, required=True, help='Path to list of training image paths')
parser.add_argument('--train_interp_depth_path',
    type=str, required=True, help='Path to list of training interpolated depth paths')
parser.add_argument('--train_validity_map_path',
    type=str, required=True, help='Path to list of training validity map paths')
parser.add_argument('--train_intrinsics_path',
    type=str, required=True, help='Path to list of training intrinsics paths')
# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=settings.N_BATCH, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=settings.N_HEIGHT, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=settings.N_WIDTH, help='Width of each sample')
parser.add_argument('--n_channel',
    type=int, default=settings.N_CHANNEL, help='Number of channels for each image')
# Training settings
parser.add_argument('--n_epoch',
    type=int, default=settings.N_EPOCH, help='Number of epochs for training')
parser.add_argument('--learning_rates',
    type=str, default=settings.LEARNING_RATES_TXT, help='Comma delimited list of learning rates')
parser.add_argument('--learning_bounds',
    type=str, default=settings.LEARNING_BOUNDS_TXT, help='Comma delimited list to change learning rate')
# Weights on loss function
parser.add_argument('--w_ph',
    type=float, default=settings.W_PH, help='Weight for photometric loss term')
parser.add_argument('--w_co',
    type=float, default=settings.W_CO, help='Weight of color consistency loss term')
parser.add_argument('--w_st',
    type=float, default=settings.W_ST, help='Weight of structural (SSIM) loss term')
parser.add_argument('--w_sm',
    type=float, default=settings.W_SM, help='Weight of local smoothness loss term')
parser.add_argument('--w_sz',
    type=float, default=settings.W_SZ, help='Weight of sparse depth consistency loss term')
parser.add_argument('--w_pc',
    type=float, default=settings.W_PC, help='Weight of pose consistency loss term')
# Network settings
parser.add_argument('--occ_threshold',
    type=float, default=settings.OCC_THRESHOLD, help='Threshold for max change in sparse depth')
parser.add_argument('--occ_ksize',
    type=int, default=settings.OCC_KSIZE, help='Kernel size for checking for possible occlusion')
parser.add_argument('--net_type',
    type=str, default=settings.NET_TYPE, help='Network architecture types: vggnet08, vggnet11')
parser.add_argument('--im_filter_pct',
    type=float, default=settings.IM_FILTER_PCT, help='Percentage filters for the image branch')
parser.add_argument('--sz_filter_pct',
    type=float, default=settings.SZ_FILTER_PCT, help='Percentage filter for the sparse depth branch')
parser.add_argument('--min_predict_z',
    type=float, default=settings.MIN_Z, help='Minimum depth prediction')
parser.add_argument('--max_predict_z',
    type=float, default=settings.MAX_Z, help='Maximum depth prediction')
# Pose parameterization
parser.add_argument('--pose_norm',
    type=str, default=settings.POSE_NORM, help='Norm for pose consistency: frobenius, geodesic')
parser.add_argument('--rot_param',
    type=str, default=settings.ROT_PARAM, help='Rotation parameterization: euler, exponential')
# Checkpoints and restore paths
parser.add_argument('--n_checkpoint',
    type=int, default=settings.N_CHECKPOINT, help='Number of iterations for each checkpoint')
parser.add_argument('--n_summary',
    type=int, default=settings.N_SUMMARY, help='Number of iterations before logging summary')
parser.add_argument('--checkpoint_path',
    type=str, default=settings.CHECKPOINT_PATH, help='Path to save checkpoints')
parser.add_argument('--restore_path',
    type=str, default=settings.RESTORE_PATH, help='Path to restore from checkpoint')
# Hardware settings
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

  args.learning_rates = args.learning_rates.split(',')
  args.learning_rates = [float(r) for r in args.learning_rates]
  args.learning_bounds = args.learning_bounds.split(',')
  args.learning_bounds = [int(b) for b in args.learning_bounds]
  assert(len(args.learning_rates) == len(args.learning_bounds)+1)

  train(train_image_path=args.train_image_path,
        train_interp_depth_path=args.train_interp_depth_path,
        train_validity_map_path=args.train_validity_map_path,
        train_intrinsics_path=args.train_intrinsics_path,
        n_batch=args.n_batch,
        n_height=args.n_height,
        n_width=args.n_width,
        n_channel=args.n_channel,
        n_epoch=args.n_epoch,
        learning_rates=args.learning_rates,
        learning_bounds=args.learning_bounds,
        w_ph=args.w_ph,
        w_co=args.w_co,
        w_st=args.w_st,
        w_sm=args.w_sm,
        w_sz=args.w_sz,
        w_pc=args.w_pc,
        occ_threshold=args.occ_threshold,
        occ_ksize=args.occ_ksize,
        net_type=args.net_type,
        im_filter_pct=args.im_filter_pct,
        sz_filter_pct=args.sz_filter_pct,
        min_predict_z=args.min_predict_z,
        max_predict_z=args.max_predict_z,
        pose_norm=args.pose_norm,
        rot_param=args.rot_param,
        n_checkpoint=args.n_checkpoint,
        n_summary=args.n_summary,
        checkpoint_path=args.checkpoint_path,
        restore_path=args.restore_path,
        n_thread=args.n_thread)
