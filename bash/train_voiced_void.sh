#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train_voiced.py \
--train_image_path training/void_train_image_1500.txt \
--train_interp_depth_path training/void_train_interp_depth_1500.txt \
--train_validity_map_path training/void_train_validity_map_1500.txt \
--train_intrinsics_path training/void_train_intrinsics_1500.txt \
--n_batch 8 \
--n_height 480 \
--n_width 640 \
--n_channel 3 \
--n_epoch 20 \
--learning_rates 1.00e-4,0.50e-4,0.25e-4 \
--learning_bounds 12,16 \
--occ_threshold 1.5 \
--occ_ksize 7 \
--net_type vggnet11 \
--im_filter_pct 0.75 \
--sz_filter_pct 0.25 \
--min_predict_z 0.1 \
--max_predict_z 8.0 \
--w_ph 1.00 \
--w_co 0.20 \
--w_st 0.80 \
--w_sm 0.10 \
--w_sz 1.00 \
--w_pc 0.10 \
--pose_norm frobenius \
--rot_param exponential \
--n_summary 1000 \
--n_checkpoint 5000 \
--checkpoint_path trained_models/vggnet11_void_model
