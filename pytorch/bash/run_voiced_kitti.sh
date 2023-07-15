export CUDA_VISIBLE_DEVICES=0

python src/run_voiced.py \
--image_path validation/kitti/kitti_val_image.txt \
--sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--depth_model_restore_path \
    trained_voiced/kitti/vgg11_exp_12x320x768_lr0-12e5_18-6e5_24-3e5_30_co020_st080_sz020_sm001_po010/voiced-23000.pth \
--input_channels_image 3 \
--input_channels_depth 2 \
--encoder_type vggnet11 \
--n_filters_encoder_image 48 96 192 384 384 \
--n_filters_encoder_depth 16 32 64 128 128 \
--n_filters_decoder 256 128 128 64 0 \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--save_outputs \
--keep_input_filenames \
--output_path \
    trained_voiced/kitti/vgg11_exp_12x320x768_lr0-12e5_18-6e5_24-3e5_30_co020_st080_sz020_sm001_po010/evaluation_results/kitti_val \
--device gpu
