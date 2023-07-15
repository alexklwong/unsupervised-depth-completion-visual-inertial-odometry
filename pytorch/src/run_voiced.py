import argparse
import torch
from voiced import run


parser = argparse.ArgumentParser()


# Checkpoint settings
parser.add_argument('--depth_model_restore_path',
    type=str, required=True, help='Path to restore depth model from checkpoint')

# Input filepaths
parser.add_argument('--image_path',
    type=str, required=True, help='Path to list of image paths')
parser.add_argument('--sparse_depth_path',
    type=str, required=True, help='Path to list of sparse depth paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground truth depth paths')

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

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.0, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.0, help='Maximum value of depth to evaluate')

# Output settings
parser.add_argument('--output_path',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--save_outputs',
    action='store_true', help='If set then store inputs and outputs into output path')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set then keep original input filenames')

# Hardware settings
parser.add_argument('--device',
    type=str, default='cuda', help='Device to use: cuda, gpu, cpu')


args = parser.parse_args()

if __name__ == '__main__':

    '''
    Assert inputs
    '''
    # Weight settings
    args.weight_initializer = args.weight_initializer.lower()

    args.activation_func = args.activation_func.lower()

  # Hardware settings
    args.device = args.device.lower()
    if args.device not in ['cuda', 'gpu', 'cpu']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    run(args.image_path,
        args.sparse_depth_path,
        ground_truth_path=args.ground_truth_path,
        # Checkpoint settings
        depth_model_restore_path=args.depth_model_restore_path,
        # Input settings
        input_channels_image=args.input_channels_image,
        input_channels_depth=args.input_channels_depth,
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
        # Depth evaluation settings
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        # Output settings
        output_path=args.output_path,
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        # Hardware settings
        device=args.device)
