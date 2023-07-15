#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/evaluate_model.py \
--image_path validation/kitti_val_image.txt \
--interp_depth_path validation/kitti_val_interp_depth.txt \
--validity_map_path validation/kitti_val_validity_map.txt \
--ground_truth_path validation/kitti_val_semi_dense_depth.txt \
--start_idx 0 \
--end_idx 1000 \
--n_batch 8 \
--n_height 352 \
--n_width 1216 \
--occ_threshold 1.5 \
--occ_ksize 7 \
--net_type vggnet11 \
--im_filter_pct 0.75 \
--sz_filter_pct 0.25 \
--min_predict_z 1.5 \
--max_predict_z 100.0 \
--min_evaluate_z 0.0 \
--max_evaluate_z 100.0 \
--save_depth \
--output_path pretrained_models/voiced_vggnet11_kitti/output \
--restore_path pretrained_models/voiced_vggnet11_kitti/model.ckpt-kitti
