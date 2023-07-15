import argparse
import torch
from voiced import train


parser = argparse.ArgumentParser()


# Training and validation input filepaths
parser.add_argument('--train_images_path',
    type=str, required=True, help='Path to list of training image paths')
parser.add_argument('--train_sparse_depth_path',
    type=str, required=True, help='Path to list of training sparse depth paths')
parser.add_argument('--train_intrinsics_path',
    type=str, required=True, help='Path to list of training camera intrinsics paths')
parser.add_argument('--val_image_path',
    type=str, default=None, help='Path to list of validation image paths')
parser.add_argument('--val_sparse_depth_path',
    type=str, default=None, help='Path to list of validation sparse depth paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default=None, help='Path to list of validation ground truth depth paths')

# Dataloader settings
parser.add_argument('--n_batch',
    type=int, default=8, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=320, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=768, help='Width of each sample')

# Input settings
parser.add_argument('--input_channels_image',
    type=int, default=3, help='Number of input image channels')
parser.add_argument('--input_channels_depth',
    type=int, default=2, help='Number of input depth channels')
parser.add_argument('--outlier_removal_kernel_size',
    type=int, default=7, help='Kernel size to filter outlier sparse depth')
parser.add_argument('--outlier_removal_threshold',
    type=float, default=1.5, help='Difference threshold to consider a point an outlier')

# Network settings
parser.add_argument('--encoder_type',
    nargs='+', type=str, default=['vggnet08'], help='Encoder type to use')
parser.add_argument('--n_filters_encoder_image',
    nargs='+', type=int, default=[48, 96, 192, 384, 384], help='Space delimited list of filters to use in each block of image encoder')
parser.add_argument('--n_filters_encoder_depth',
    nargs='+', type=int, default=[16, 32, 64, 128, 128], help='Space delimited list of filters to use in each block of depth encoder')
parser.add_argument('--n_filters_decoder',
    nargs='+', type=int, default=[256, 128, 128, 64, 32], help='Space delimited list of filters to use in each block of decoder')
parser.add_argument('--min_predict_depth',
    type=float, default=1.5, help='Minimum value of depth prediction')
parser.add_argument('--max_predict_depth',
    type=float, default=100.0, help='Maximum value of depth prediction')

# Weight settings
parser.add_argument('--weight_initializer',
    type=str, default='xavier_normal', help='Weight initialization type: kaiming_uniform, kaiming_normal, xavier_uniform, xavier_normal')
parser.add_argument('--activation_func',
    type=str, default='leaky_relu', help='Activation function after each layer: relu, leaky_relu, elu, sigmoid')

# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[1.00e-4, 0.50e-4, 0.25e-4], help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[20, 40, 60], help='Space delimited list to change learning rate')
parser.add_argument('--rotation_parameterization',
    type=str, default='euler', help='Rotation parameterization: euler, exponential')
parser.add_argument('--augmentation_random_crop_type',
    nargs='+', type=str, default=['horizontal', 'vertical', 'anchored', 'bottom'], help='Random crop type for data augmentation: horizontal, vertical, anchored, bottom')

# Loss function settings
parser.add_argument('--w_color',
    type=float, default=0.20, help='Weight of color consistency loss')
parser.add_argument('--w_structure',
    type=float, default=0.80, help='Weight of structural consistency loss')
parser.add_argument('--w_sparse_depth',
    type=float, default=1.00, help='Weight of sparse depth consistency loss')
parser.add_argument('--w_smoothness',
    type=float, default=0.15, help='Weight of local smoothness loss')
parser.add_argument('--w_pose',
    type=float, default=0.10, help='Weight of pose consistency loss')
parser.add_argument('--w_weight_decay_depth',
    type=float, default=0.0, help='Weight of weight decay regularization for depth')
parser.add_argument('--w_weight_decay_pose',
    type=float, default=1e-4, help='Weight of weight decay regularization for pose')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.0, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.0, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=5000, help='Number of iterations for each checkpoint')
parser.add_argument('--n_step_per_summary',
    type=int, default=5000, help='Number of iterations before logging summary')
parser.add_argument('--n_image_per_summary',
    type=int, default=4, help='Number of samples to include in visual display summary')
parser.add_argument('--start_step_validation',
    type=int, default=200000, help='Number of steps before starting validation')
parser.add_argument('--depth_model_restore_path',
    type=str, default=None, help='Path to restore depth model from checkpoint')
parser.add_argument('--pose_model_restore_path',
    type=str, default=None, help='Path to restore pose model from checkpoint')

# Hardware settings
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: cuda, gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=8, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    # Weight settings
    args.weight_initializer = args.weight_initializer.lower()

    args.activation_func = args.activation_func.lower()

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)

    args.augmentation_random_crop_type = [
        crop_type.lower() for crop_type in args.augmentation_random_crop_type
    ]

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in ['cuda', 'gpu', 'cpu']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    train(train_images_path=args.train_images_path,
          train_sparse_depth_path=args.train_sparse_depth_path,
          train_intrinsics_path=args.train_intrinsics_path,
          val_image_path=args.val_image_path,
          val_sparse_depth_path=args.val_sparse_depth_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Dataloader settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          # Input settings
          input_channels_image=args.input_channels_image,
          input_channels_depth=args.input_channels_depth,
          outlier_removal_kernel_size=args.outlier_removal_kernel_size,
          outlier_removal_threshold=args.outlier_removal_threshold,
          # Network settings
          encoder_type=args.encoder_type,
          n_filters_encoder_image=args.n_filters_encoder_image,
          n_filters_encoder_depth=args.n_filters_encoder_depth,
          n_filters_decoder=args.n_filters_decoder,
          min_predict_depth=args.min_predict_depth,
          max_predict_depth=args.max_predict_depth,
          # Weight settings
          weight_initializer=args.weight_initializer,
          activation_func=args.activation_func,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          rotation_parameterization=args.rotation_parameterization,
          augmentation_random_crop_type=args.augmentation_random_crop_type,
          # Loss function settings
          w_color=args.w_color,
          w_structure=args.w_structure,
          w_sparse_depth=args.w_sparse_depth,
          w_smoothness=args.w_smoothness,
          w_pose=args.w_pose,
          w_weight_decay_depth=args.w_weight_decay_depth,
          w_weight_decay_pose=args.w_weight_decay_pose,
          # Depth evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          # Checkpoint settings
          checkpoint_path=args.checkpoint_path,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          n_step_per_summary=args.n_step_per_summary,
          n_image_per_summary=args.n_image_per_summary,
          start_step_validation=args.start_step_validation,
          depth_model_restore_path=args.depth_model_restore_path,
          pose_model_restore_path=args.pose_model_restore_path,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)
