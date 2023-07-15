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
import os, sys, argparse
import numpy as np
import tensorflow as tf
import global_constants as settings
import data_utils, eval_utils
from dataloader import DataLoader
from voiced_model import VOICEDModel
from data_utils import log


parser = argparse.ArgumentParser()

N_HEIGHT = 352
N_WIDTH = 1216

# Model path
parser.add_argument('--restore_path',
    type=str, required=True, help='Path to restore model')
# Input paths
parser.add_argument('--image_path',
    type=str, required=True, help='Path to list of image paths')
parser.add_argument('--interp_depth_path',
    type=str, required=True, help='Path to list of interpolated depth paths')
parser.add_argument('--validity_map_path',
    type=str, required=True, help='Path to list of validity map paths')
parser.add_argument('--ground_truth_path',
    type=str, default='', help='Path to list of ground truth paths')
parser.add_argument('--start_idx',
    type=int, default=0, help='Start index of the list of paths to evaluate')
parser.add_argument('--end_idx',
    type=int, default=1000, help='Last index of the list of paths to evaluate')
# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=settings.N_BATCH, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=settings.N_HEIGHT, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=settings.N_WIDTH, help='Width of each sample')
parser.add_argument('--n_channel',
    type=int, default=settings.N_CHANNEL, help='Number of channels for each image')
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
parser.add_argument('--min_evaluate_z',
    type=float, default=settings.MIN_Z, help='Minimum depth to evaluate')
parser.add_argument('--max_evaluate_z',
    type=float, default=settings.MAX_Z, help='Maximum depth to evaluate')
# Output options
parser.add_argument('--save_depth',
    action='store_true', help='If set, saves depth maps into output_path')
parser.add_argument('--output_path',
    type=str, default='output', help='Directory to store output')
# Hardware settings
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads for fetching')


args = parser.parse_args()

log_path = os.path.join(args.output_path, 'results.txt')
if not os.path.exists(args.output_path):
  os.makedirs(args.output_path)

# Load image paths from file for evaluation
im_paths = sorted(data_utils.read_paths(args.image_path))[args.start_idx:args.end_idx]
iz_paths = sorted(data_utils.read_paths(args.interp_depth_path))[args.start_idx:args.end_idx]
vm_paths = sorted(data_utils.read_paths(args.validity_map_path))[args.start_idx:args.end_idx]
assert(len(im_paths) == len(iz_paths))
assert(len(im_paths) == len(vm_paths))
n_sample = len(im_paths)

if args.ground_truth_path != '':
  gt_paths = sorted(data_utils.read_paths(args.ground_truth_path))[args.start_idx:args.end_idx]
  assert(len(im_paths) == len(gt_paths))

# Pad all paths based on batch
im_paths = data_utils.pad_batch(im_paths, args.n_batch)
iz_paths = data_utils.pad_batch(iz_paths, args.n_batch)
vm_paths = data_utils.pad_batch(vm_paths, args.n_batch)
n_step = len(im_paths)//args.n_batch

gt_arr = []
if args.ground_truth_path != '':
  # Load ground truth
  for idx in range(n_sample):
    sys.stdout.write(
        'Loading {}/{} groundtruth depth maps \r'.format(idx+1, n_sample))
    sys.stdout.flush()

    gt, vm = data_utils.load_depth_with_validity_map(gt_paths[idx])
    gt = np.concatenate([np.expand_dims(gt, axis=-1), np.expand_dims(vm, axis=-1)], axis=-1)
    gt_arr.append(gt)

  print('Completed loading {} groundtruth depth maps'.format(n_sample))

with tf.Graph().as_default():
  # Initialize dataloader
  dataloader = DataLoader(shape=[args.n_batch, args.n_height, args.n_width, 3],
                          name='dataloader',
                          is_training=False,
                          n_thread=args.n_thread,
                          prefetch_size=2*args.n_thread)
  # Fetch the input from dataloader
  im0 = dataloader.next_element[0]
  sz0 = dataloader.next_element[3]

  # Build computation graph
  model = VOICEDModel(im0, im0, im0, sz0, None,
                      is_training=False,
                      occ_threshold=args.occ_threshold,
                      occ_ksize=args.occ_ksize,
                      net_type=args.net_type,
                      im_filter_pct=args.im_filter_pct,
                      sz_filter_pct=args.sz_filter_pct,
                      min_predict_z=args.min_predict_z,
                      max_predict_z=args.max_predict_z)

  # Initialize Tensorflow session
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  # Load from checkpoint
  train_saver = tf.train.Saver()
  session.run(tf.global_variables_initializer())
  session.run(tf.local_variables_initializer())
  train_saver.restore(session, args.restore_path)

  log('Evaluating {}'.format(args.restore_path), log_path)
  # Load image, dense depth, sparse depth, intrinsics, and ground-truth
  dataloader.initialize(session,
                        image_paths=im_paths,
                        interp_depth_paths=iz_paths,
                        validity_map_paths=vm_paths)

  z_arr = np.zeros([n_step*args.n_batch, args.n_height, args.n_width, 1])
  step = 0
  while True:
    try:
      sys.stdout.write(
          'Processed {}/{} examples \r'.format(step*args.n_batch, n_sample))
      sys.stdout.flush()

      batch_start = step*args.n_batch
      batch_end = step*args.n_batch+args.n_batch
      step += 1
      z_arr[batch_start:batch_end, ...] = session.run(model.predict)
    except tf.errors.OutOfRangeError:
      break
  # Remove the padded examples
  z_arr = z_arr[0:n_sample, ...]

  # Run evaluation
  if len(gt_arr) > 0:
    mae   = np.zeros(n_sample, np.float32)
    rmse  = np.zeros(n_sample, np.float32)
    imae  = np.zeros(n_sample, np.float32)
    irmse = np.zeros(n_sample, np.float32)

    for idx in range(n_sample):
      z = np.squeeze(z_arr[idx, ...])
      gt = np.squeeze(gt_arr[idx][..., 0])
      vm = np.squeeze(gt_arr[idx][..., 1])

      # Create mask for evaluation
      valid_mask = np.where(vm > 0, 1, 0)
      min_max_mask = np.logical_and(gt > args.min_evaluate_z, gt < args.max_evaluate_z)
      mask = np.where(np.logical_and(valid_mask, min_max_mask) > 0)
      z = z[mask]
      gt = gt[mask]

      # Run evaluations: MAE, RMSE in meters, iMAE, iRMSE in 1/kilometers
      mae[idx] = eval_utils.mean_abs_err(1000.0*z, 1000.0*gt)
      rmse[idx] = eval_utils.root_mean_sq_err(1000.0*z, 1000.0*gt)
      imae[idx] = eval_utils.inv_mean_abs_err(0.001*z, 0.001*gt)
      irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001*z, 0.001*gt)

    # Compute mean error
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)
    log('{:>10} {:>10} {:>10} {:>10}'.format('MAE', 'RMSE', 'iMAE', 'iRMSE'), log_path)
    log('{:10.4f} {:10.4f} {:10.4f} {:10.4f}'.format(mae, rmse, imae, irmse), log_path)

  # Store output depth as images
  if args.save_depth:
    output_dirpath = os.path.join(args.output_path, 'saved')
    print('Storing output depth as PNG into {}'.format(output_dirpath))

    if not os.path.exists(output_dirpath):
      os.makedirs(output_dirpath)

    for idx in range(n_sample):
      z = np.squeeze(z_arr[idx, ...])
      _, filename = os.path.split(iz_paths[idx])
      output_path = os.path.join(output_dirpath, filename)
      data_utils.save_depth(z, output_path)
