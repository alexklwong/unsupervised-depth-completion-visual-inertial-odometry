#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_voiced.py \
--train_images_path training/kitti/unsupervised/kitti_train_nonstatic_images.txt \
--train_sparse_depth_path training/kitti/unsupervised/kitti_train_nonstatic_sparse_depth.txt \
--train_intrinsics_path training/kitti/unsupervised/kitti_train_nonstatic_intrinsics.txt \
--val_image_path validation/kitti/kitti_val_image.txt \
--val_sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--val_ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--n_batch 8 \
--n_height 320 \
--n_width 768 \
--input_channels_image 3 \
--input_channels_depth 2 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--encoder_type vggnet11 \
--n_filters_encoder_image 48 96 192 384 384 \
--n_filters_encoder_depth 16 32 64 128 128 \
--n_filters_decoder 256 128 128 64 0 \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--learning_rates 1.2e-4 0.6e-4 0.3e-4 \
--learning_schedule 18 24 30 \
--rotation_parameterization exponential \
--augmentation_random_crop_type horizontal bottom \
--w_color 0.20 \
--w_structure 0.80 \
--w_sparse_depth 0.20 \
--w_smoothness 0.01 \
--w_pose 0.10 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 1e-3 \
--max_evaluate_depth 100.0 \
--checkpoint_path trained_voiced/kitti/voiced_vgg11 \
--n_step_per_checkpoint 5000 \
--n_step_per_summary 5000 \
--n_image_per_summary 4 \
--start_step_validation 10000 \
--device gpu \
--n_thread 8
