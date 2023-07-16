#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_voiced.py \
--train_images_path training/void/unsupervised/void_train_image_1500.txt \
--train_sparse_depth_path training/void/unsupervised/void_train_sparse_depth_1500.txt \
--train_intrinsics_path training/void/unsupervised/void_train_intrinsics_1500.txt \
--val_image_path testing/void/void_test_image_1500.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--val_ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--n_batch 12 \
--n_height 480 \
--n_width 640 \
--input_channels_image 3 \
--input_channels_depth 2 \
--outlier_removal_kernel_size 7 \
--outlier_removal_threshold 1.5 \
--encoder_type vggnet11 \
--n_filters_encoder_image 48 96 192 384 384 \
--n_filters_encoder_depth 16 32 64 128 128 \
--n_filters_decoder 256 128 128 64 0 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--learning_rates 1e-4 5e-5 \
--learning_schedule 10 20 \
--rotation_parameterization exponential \
--augmentation_random_crop_type none \
--w_color 0.20 \
--w_structure 0.80 \
--w_sparse_depth 0.50 \
--w_smoothness 2.00 \
--w_pose 0.10 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--checkpoint_path trained_voiced/void1500/voiced_vgg11 \
--n_step_per_checkpoint 1000 \
--n_step_per_summary 1000 \
--n_image_per_summary 4 \
--start_step_validation 1000 \
--device gpu \
--n_thread 8