export CUDA_VISIBLE_DEVICES=0

python src/run_voiced.py \
--image_path testing/void/void_test_image_1500.txt \
--sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--depth_model_restore_path \
    trained_voiced/void1500/vgg11_exp_12x480x640_lr0-1e4_10-5e5_20_co020_st080_sz050_sm200_po010_wd000/voiced-23000.pth \
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
    trained_voiced/void1500/vgg11_exp_12x480x640_lr0-1e4_10-5e5_20_co020_st080_sz050_sm200_po010_wd000/evaluation_results/void1500 \
--device gpu
