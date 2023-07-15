#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/evaluate_model.py \
--image_path testing/void_test_image_1500.txt \
--interp_depth_path testing/void_test_interp_depth_1500.txt \
--validity_map_path testing/void_test_validity_map_1500.txt \
--ground_truth_path testing/void_test_ground_truth_1500.txt \
--start_idx 0 \
--end_idx 800 \
--n_batch 8 \
--n_height 480 \
--n_width 640 \
--occ_threshold 1.5 \
--occ_ksize 7 \
--net_type vggnet11 \
--im_filter_pct 0.75 \
--sz_filter_pct 0.25 \
--min_predict_z 0.1 \
--max_predict_z 8.0 \
--min_evaluate_z 0.2 \
--max_evaluate_z 5.0 \
--save_depth \
--output_path pretrained_models/voiced_vggnet11_void/output \
--restore_path pretrained_models/voiced_vggnet11_void/model.ckpt-void
