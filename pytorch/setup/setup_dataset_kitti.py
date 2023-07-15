'''
Authors:
Tian Yu Liu <tianyu@cs.ucla.edu>
Parth Agrawal <parthagrawal24@ucla.edu>
Allison Chen <allisonchen2@ucla.edu>
Alex Wong <alex.wong@yale.edu>

If you use this code, please cite the following paper:
T.Y. Liu, P. Agrawal, A. Chen, B.W. Hong, and A. Wong. Monitored Distillation for Positive Congruent Depth Completion.
https://arxiv.org/abs/2203.16034

@inproceedings{liu2022monitored,
  title={Monitored distillation for positive congruent depth completion},
  author={Liu, Tian Yu and Agrawal, Parth and Chen, Allison and Hong, Byung-Woo and Wong, Alex},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
'''

import os, sys, glob, argparse, cv2
import numpy as np
import multiprocessing as mp
sys.path.insert(0, 'src')
import data_utils


'''
Paths for KITTI dataset
'''
KITTI_RAW_DATA_DIRPATH = os.path.join('data', 'kitti_raw_data')
KITTI_DEPTH_COMPLETION_DIRPATH = os.path.join('data', 'kitti_depth_completion')

KITTI_TRAINVAL_SPARSE_DEPTH_DIRPATH = os.path.join(
    KITTI_DEPTH_COMPLETION_DIRPATH, 'train_val_split', 'sparse_depth')
KITTI_TRAINVAL_GROUND_TRUTH_DIRPATH = os.path.join(
    KITTI_DEPTH_COMPLETION_DIRPATH, 'train_val_split', 'ground_truth')
KITTI_VALIDATION_DIRPATH = os.path.join(
    KITTI_DEPTH_COMPLETION_DIRPATH, 'validation')
KITTI_TESTING_DIRPATH = os.path.join(
    KITTI_DEPTH_COMPLETION_DIRPATH, 'testing')
KITTI_CALIBRATION_FILENAME = 'calib_cam_to_cam.txt'

KITTI_STATIC_FRAMES_FILEPATH = os.path.join('data_split', 'kitti', 'kitti_static_frames.txt')
KITTI_STATIC_FRAMES_PATHS = data_utils.read_paths(KITTI_STATIC_FRAMES_FILEPATH)

# To be concatenated to sequence path
KITTI_TRAINVAL_IMAGE_REFPATH = os.path.join('proj_depth', 'velodyne_raw')
KITTI_TRAINVAL_SPARSE_DEPTH_REFPATH = os.path.join('proj_depth', 'velodyne_raw')
KITTI_TRAINVAL_GROUND_TRUTH_REFPATH = os.path.join('proj_depth', 'groundtruth')

'''
Output paths
'''
KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH = os.path.join(
    'data', 'kitti_depth_completion_derived')

TRAIN_SUPERVISED_REF_DIRPATH = os.path.join('training', 'kitti', 'supervised')
TRAIN_UNSUPERVISED_REF_DIRPATH = os.path.join('training', 'kitti', 'unsupervised')
TRAIN_UNUSED_REF_DIRPATH = os.path.join('training', 'kitti', 'unused')
VAL_REF_DIRPATH = os.path.join('validation', 'kitti')
TEST_REF_DIRPATH = os.path.join('testing', 'kitti')

# Paths to files for supervised training
TRAIN_SUPERVISED_IMAGE_FILEPATH = os.path.join(
    TRAIN_SUPERVISED_REF_DIRPATH, 'kitti_train_image.txt')
TRAIN_SUPERVISED_SPARSE_DEPTH_FILEPATH = os.path.join(
    TRAIN_SUPERVISED_REF_DIRPATH, 'kitti_train_sparse_depth.txt')
TRAIN_SUPERVISED_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_SUPERVISED_REF_DIRPATH, 'kitti_train_ground_truth.txt')
TRAIN_SUPERVISED_FOCAL_LENGTH_BASELINE_FILEPATH = os.path.join(
    TRAIN_SUPERVISED_REF_DIRPATH, 'kitti_train_focal_length_baseline.txt')
TRAIN_SUPERVISED_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_SUPERVISED_REF_DIRPATH, 'kitti_train_intrinsics.txt')

# Paths to files for unsupervised training
TRAIN_UNSUPERVISED_IMAGES_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_images.txt')
TRAIN_UNSUPERVISED_SPARSE_DEPTH_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_sparse_depth.txt')
TRAIN_UNSUPERVISED_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_ground_truth.txt')
TRAIN_UNSUPERVISED_FOCAL_LENGTH_BASELINE_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_focal_length_baseline.txt')
TRAIN_UNSUPERVISED_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_intrinsics.txt')

TRAIN_UNSUPERVISED_IMAGES_LEFT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_images_left.txt')
TRAIN_UNSUPERVISED_IMAGES_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_images_right.txt')
TRAIN_UNSUPERVISED_SPARSE_DEPTH_LEFT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_sparse_depth_left.txt')
TRAIN_UNSUPERVISED_SPARSE_DEPTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_sparse_depth_right.txt')
TRAIN_UNSUPERVISED_GROUND_TRUTH_LEFT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_ground_truth_left.txt')
TRAIN_UNSUPERVISED_GROUND_TRUTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_ground_truth_right.txt')
TRAIN_UNSUPERVISED_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_focal_length_baseline_left.txt')
TRAIN_UNSUPERVISED_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_focal_length_baseline_right.txt')
TRAIN_UNSUPERVISED_INTRINSICS_LEFT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_intrinsics_left.txt')
TRAIN_UNSUPERVISED_INTRINSICS_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_intrinsics_right.txt')

# Paths to files containing nonstatic data for unsupervised training
TRAIN_UNSUPERVISED_NONSTATIC_IMAGES_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_images.txt')
TRAIN_UNSUPERVISED_NONSTATIC_SPARSE_DEPTH_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_sparse_depth.txt')
TRAIN_UNSUPERVISED_NONSTATIC_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_ground_truth.txt')
TRAIN_UNSUPERVISED_NONSTATIC_FOCAL_LENGTH_BASELINE_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_focal_length_baseline.txt')
TRAIN_UNSUPERVISED_NONSTATIC_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_intrinsics.txt')

TRAIN_UNSUPERVISED_NONSTATIC_IMAGES_LEFT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_images_left.txt')
TRAIN_UNSUPERVISED_NONSTATIC_IMAGES_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_images_right.txt')
TRAIN_UNSUPERVISED_NONSTATIC_SPARSE_DEPTH_LEFT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_sparse_depth_left.txt')
TRAIN_UNSUPERVISED_NONSTATIC_SPARSE_DEPTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_sparse_depth_right.txt')
TRAIN_UNSUPERVISED_NONSTATIC_GROUND_TRUTH_LEFT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_ground_truth_left.txt')
TRAIN_UNSUPERVISED_NONSTATIC_GROUND_TRUTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_ground_truth_right.txt')
TRAIN_UNSUPERVISED_NONSTATIC_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_focal_length_baseline_left.txt')
TRAIN_UNSUPERVISED_NONSTATIC_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_focal_length_baseline_right.txt')
TRAIN_UNSUPERVISED_NONSTATIC_INTRINSICS_LEFT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_intrinsics_left.txt')
TRAIN_UNSUPERVISED_NONSTATIC_INTRINSICS_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNSUPERVISED_REF_DIRPATH, 'kitti_train_nonstatic_intrinsics_right.txt')

# Paths to unused data
UNUSED_IMAGE_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_image.txt')
UNUSED_SPARSE_DEPTH_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_sparse_depth.txt')
UNUSED_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_ground_truth_left.txt')
UNUSED_FOCAL_LENGTH_BASELINE_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_focal_length_baseline.txt')
UNUSED_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_intrinsics.txt')

UNUSED_IMAGE_LEFT_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_image_left.txt')
UNUSED_IMAGE_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_image_right.txt')
UNUSED_SPARSE_DEPTH_LEFT_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_sparse_depth_left.txt')
UNUSED_SPARSE_DEPTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_sparse_depth_right.txt')
UNUSED_GROUND_TRUTH_LEFT_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_ground_truth_left.txt')
UNUSED_GROUND_TRUTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_ground_truth_right.txt')
UNUSED_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_focal_length_baseline_left.txt')
UNUSED_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_focal_length_baseline_right.txt')
UNUSED_INTRINSICS_LEFT_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_intrinsics_left.txt')
UNUSED_INTRINSICS_RIGHT_FILEPATH = os.path.join(
    TRAIN_UNUSED_REF_DIRPATH, 'kitti_unused_intrinsics_right.txt')

# Paths to files for validation
VAL_IMAGE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'kitti_val_image.txt')
VAL_SPARSE_DEPTH_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'kitti_val_sparse_depth.txt')
VAL_GROUND_TRUTH_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'kitti_val_ground_truth.txt')
VAL_INTRINSICS_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'kitti_val_intrinsics.txt')

# Paths to files for testing
TEST_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_image.txt')
TEST_SPARSE_DEPTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_sparse_depth.txt')
TEST_GROUND_TRUTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_ground_truth.txt')
TEST_INTRINSICS_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_intrinsics.txt')


def map_intrinsics_and_focal_length_baseline(paths_only=False):
    '''
    Map camera intrinsics and focal length + baseline to directories

    Arg(s):
        paths_only : bool
            boolean flag if set then create paths only
    Returns:
        dict[str, str] : sequence dates and camera id to focal length and baseline paths
        dict[str, str] : sequence dates and camera id to camera intrinsics paths
    '''

    # Build a mapping between the camera intrinsics to the directories
    intrinsics_files = sorted(glob.glob(
        os.path.join(KITTI_RAW_DATA_DIRPATH, '*', KITTI_CALIBRATION_FILENAME)))

    intrinsics_dkeys = {}
    focal_length_baseline_dkeys = {}
    for intrinsics_file in intrinsics_files:
        # Example: data/kitti_depth_completion_mondi/data/2011_09_26/intrinsics_left.npy
        intrinsics_left_path = intrinsics_file \
            .replace(KITTI_RAW_DATA_DIRPATH, os.path.join(KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH, 'data')) \
            .replace(KITTI_CALIBRATION_FILENAME, 'intrinsics_left.npy')

        intrinsics_right_path = intrinsics_file \
            .replace(KITTI_RAW_DATA_DIRPATH, os.path.join(KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH, 'data')) \
            .replace(KITTI_CALIBRATION_FILENAME, 'intrinsics_right.npy')

        # Example: data/kitti_depth_completion_mondi/data/2011_09_26/focal_length_baseline_left.npy
        focal_length_baseline_left_path = intrinsics_file \
            .replace(KITTI_RAW_DATA_DIRPATH, os.path.join(KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH, 'data')) \
            .replace(KITTI_CALIBRATION_FILENAME, 'focal_length_baseline_left.npy')

        focal_length_baseline_right_path = intrinsics_file \
            .replace(KITTI_RAW_DATA_DIRPATH, os.path.join(KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH, 'data')) \
            .replace(KITTI_CALIBRATION_FILENAME, 'focal_length_baseline_right.npy')

        sequence_dirpath = os.path.split(intrinsics_left_path)[0]

        if not os.path.exists(sequence_dirpath):
            os.makedirs(sequence_dirpath)

        calibration = data_utils.load_calibration(intrinsics_file)

        # Obtain calibration for camera 0 and camera 1
        camera_left = np.reshape(np.asarray(calibration['P_rect_02'], np.float32), [3, 4])
        camera_right = np.reshape(np.asarray(calibration['P_rect_03'], np.float32), [3, 4])

        # Extract camera parameters
        intrinsics_left = camera_left[:3, :3]
        intrinsics_right = camera_right[:3, :3]

        # Focal length of the cameras
        focal_length_left = intrinsics_left[0, 0]
        focal_length_right = intrinsics_right[0, 0]

        # camera2 is left of camera0 (-6cm) camera3 is right of camera2 (+53.27cm)
        translation_left = camera_left[0, 3] / focal_length_left
        translation_right = camera_right[0, 3] / focal_length_right
        baseline = translation_left - translation_right

        position_left = camera_left[0:3, 3] / focal_length_left
        position_right = camera_right[0:3, 3] / focal_length_right

        # Baseline should be just translation along x
        assert np.abs(baseline - np.linalg.norm(position_left - position_right)) < 0.01, \
            'baseline={}'.format(baseline)

        # Concatenate together as fB
        focal_length_baseline_left = np.concatenate([
            np.expand_dims(focal_length_left, axis=-1),
            np.expand_dims(baseline, axis=-1)],
            axis=-1)

        focal_length_baseline_right = np.concatenate([
            np.expand_dims(focal_length_right, axis=-1),
            np.expand_dims(baseline, axis=-1)],
            axis=-1)

        # Store as numpy
        if not paths_only:
            np.save(focal_length_baseline_left_path, focal_length_baseline_left)
            np.save(focal_length_baseline_right_path, focal_length_baseline_right)

            np.save(intrinsics_left_path, intrinsics_left)
            np.save(intrinsics_right_path, intrinsics_right)

        # Add as keys to instrinsics and focal length baseline dictionaries
        sequence_date = intrinsics_file.split(os.sep)[2]

        focal_length_baseline_dkeys[(sequence_date, 'image_02')] = focal_length_baseline_left_path
        focal_length_baseline_dkeys[(sequence_date, 'image_03')] = focal_length_baseline_right_path

        intrinsics_dkeys[(sequence_date, 'image_02')] = intrinsics_left_path
        intrinsics_dkeys[(sequence_date, 'image_03')] = intrinsics_right_path

    return focal_length_baseline_dkeys, intrinsics_dkeys

def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple[str]
            image path at time t=0,
            image path at time t=-1,
            image path at time t=1,
            sparse depth path at time t,
            ground truth path at time t,
            boolean flag if set then create paths only
    Returns:
        str : output concatenated image path at time t
        str : output sparse depth path at time t
        str : output validity map path at time t
        str : output ground truth path at time t
    '''

    image_curr_path, \
        image_prev_path, \
        image_next_path, \
        sparse_depth_path, \
        ground_truth_path, \
        focal_length_baseline_path, \
        intrinsics_path, \
        paths_only = inputs

    if not paths_only:
        # Read images and concatenate together
        image_curr = cv2.imread(image_curr_path)
        image_prev = cv2.imread(image_prev_path)
        image_next = cv2.imread(image_next_path)
        image = np.concatenate([image_prev, image_curr, image_next], axis=1)

    image_output_path = sparse_depth_path \
        .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH) \
        .replace(os.path.join(os.sep + 'proj_depth', 'velodyne_raw'), '') \
        .replace('sparse_depth', 'image_triplet')

    # Create output directories
    output_dirpath = os.path.dirname(image_output_path)
    if not os.path.exists(output_dirpath):
        try:
            os.makedirs(output_dirpath)
        except FileExistsError:
            pass

    if not paths_only:
        # Write to disk
        cv2.imwrite(image_output_path, image)

    return (image_output_path,
            sparse_depth_path,
            ground_truth_path,
            focal_length_baseline_path,
            intrinsics_path)

def filter_static_frames(inputs):
    '''
    Return tuple of list of paths containing the nonstatic scenes

    Arg(s):
        inputs : tuple(list[str])
            image paths
            sparse depth paths
            ground truth paths
            focal length and baseline paths
            intrinsics paths
    Returns:
        list[str] : nonstatic image paths
        list[str] : nonstatic sparse depth paths
        list[str] : nonstatic ground truth paths
        list[str] : nonstatic focal length and baseline paths
        list[str] : nonstatic intrinsics paths
    '''

    # Process static frames file
    kitti_static_frames_parts = []
    for path in KITTI_STATIC_FRAMES_PATHS:
        parts = path.split(' ')
        kitti_static_frames_parts.append((parts[1], parts[2]))

    image_paths, \
        sparse_depth_paths, \
        ground_truth_paths, \
        focal_length_baseline_paths, \
        intrinsics_paths = inputs

    nonstatic_image_paths = []
    nonstatic_sparse_depth_paths = []
    nonstatic_ground_truth_paths = []
    nonstatic_focal_length_baseline_paths = []
    nonstatic_intrinsics_paths = []

    n_removed = 0
    n_sample = len(image_paths)

    for idx in range(n_sample):
        image_path = image_paths[idx]
        sparse_depth_path = sparse_depth_paths[idx]
        ground_truth_path = ground_truth_paths[idx]
        focal_length_baseline_path = focal_length_baseline_paths[idx]
        intrinsics_path = intrinsics_paths[idx]

        # Sanity checks
        filename = os.path.basename(image_path)
        assert filename == os.path.basename(sparse_depth_path)
        assert filename == os.path.basename(ground_truth_path)

        is_static = False

        # Flag if file is in static frames file
        for parts in kitti_static_frames_parts:
            if parts[0] in image_path and parts[1] in image_path:
                is_static = True
                break

        if is_static:
            n_removed = n_removed + 1
        else:
            nonstatic_image_paths.append(image_path)
            nonstatic_sparse_depth_paths.append(sparse_depth_path)
            nonstatic_ground_truth_paths.append(ground_truth_path)
            nonstatic_focal_length_baseline_paths.append(focal_length_baseline_path)
            nonstatic_intrinsics_paths.append(intrinsics_path)

        sys.stdout.write(
            'Processed {}/{} examples \r'.format(idx + 1, n_sample))
        sys.stdout.flush()

    print('Removed {} static frames from {} examples'.format(n_removed, n_sample))

    return (nonstatic_image_paths,
            nonstatic_sparse_depth_paths,
            nonstatic_ground_truth_paths,
            nonstatic_focal_length_baseline_paths,
            nonstatic_intrinsics_paths)

def setup_dataset_kitti_training(paths_only=False, n_thread=8):
    '''
    Fetch image, sparse depth, and ground truth paths for training

    Arg(s):
        paths_only : bool
            if set, then only produces paths
    '''

    focal_length_baseline_dkeys, \
        intrinsics_dkeys = map_intrinsics_and_focal_length_baseline(paths_only=paths_only)

    '''
    Fetch paths for training
    '''
    # Paths for supervised training
    train_supervised_image_paths = []
    train_supervised_sparse_depth_paths = []
    train_supervised_ground_truth_paths = []
    train_supervised_focal_length_baseline_paths = []
    train_supervised_intrinsics_paths = []

    # Paths for unsupervised training
    train_unsupervised_images_left_paths = []
    train_unsupervised_images_right_paths = []
    train_unsupervised_sparse_depth_left_paths = []
    train_unsupervised_sparse_depth_right_paths = []
    train_unsupervised_ground_truth_left_paths = []
    train_unsupervised_ground_truth_right_paths = []
    train_unsupervised_focal_length_baseline_left_paths = []
    train_unsupervised_focal_length_baseline_right_paths = []
    train_unsupervised_intrinsics_left_paths = []
    train_unsupervised_intrinsics_right_paths = []

    train_unsupervised_nonstatic_images_left_paths = []
    train_unsupervised_nonstatic_images_right_paths = []
    train_unsupervised_nonstatic_sparse_depth_left_paths = []
    train_unsupervised_nonstatic_sparse_depth_right_paths = []
    train_unsupervised_nonstatic_ground_truth_left_paths = []
    train_unsupervised_nonstatic_ground_truth_right_paths = []
    train_unsupervised_nonstatic_focal_length_baseline_left_paths = []
    train_unsupervised_nonstatic_focal_length_baseline_right_paths = []
    train_unsupervised_nonstatic_intrinsics_left_paths = []
    train_unsupervised_nonstatic_intrinsics_right_paths = []

    unused_image_left_paths = []
    unused_image_right_paths = []
    unused_sparse_depth_left_paths = []
    unused_sparse_depth_right_paths = []
    unused_ground_truth_left_paths = []
    unused_ground_truth_right_paths = []
    unused_focal_length_baseline_left_paths = []
    unused_focal_length_baseline_right_paths = []
    unused_intrinsics_left_paths = []
    unused_intrinsics_right_paths = []

    # Iterate through train and val directories
    for refdir in ['train', 'val']:

        sparse_depth_sequence_dirpath = sorted(glob.glob(
            os.path.join(KITTI_TRAINVAL_SPARSE_DEPTH_DIRPATH, refdir, '*/')))

        # Iterate through sequences
        for sequence_dirpath in sparse_depth_sequence_dirpath:

            # Fetch sparse depth paths
            sparse_depth_left_paths = sorted(glob.glob(
                os.path.join(sequence_dirpath, KITTI_TRAINVAL_SPARSE_DEPTH_REFPATH, 'image_02', '*.png')))

            sparse_depth_right_paths = sorted(glob.glob(
                os.path.join(sequence_dirpath, KITTI_TRAINVAL_SPARSE_DEPTH_REFPATH, 'image_03', '*.png')))

            # Fetch ground_truth paths
            ground_truth_sequence_dirpath = sequence_dirpath.replace(
                'sparse_depth', 'ground_truth')

            ground_truth_left_paths = sorted(glob.glob(
                os.path.join(ground_truth_sequence_dirpath, KITTI_TRAINVAL_GROUND_TRUTH_REFPATH, 'image_02', '*.png')))

            ground_truth_right_paths = sorted(glob.glob(
                os.path.join(ground_truth_sequence_dirpath, KITTI_TRAINVAL_GROUND_TRUTH_REFPATH, 'image_03', '*.png')))

            # Fetch image paths
            sequence = sparse_depth_left_paths[0].split(os.sep)[5]
            sequence_date = sequence[0:10]

            image_left_all_paths = sorted(glob.glob(
                os.path.join(KITTI_RAW_DATA_DIRPATH, sequence_date, sequence, 'image_02', 'data', '*.png')))

            image_right_all_paths = sorted(glob.glob(
                os.path.join(KITTI_RAW_DATA_DIRPATH, sequence_date, sequence, 'image_03', 'data', '*.png')))

            # Get image paths that correspond to sparse depth paths
            keep_indices = []
            for idx in range(len(image_left_all_paths)):

                filename_left = image_left_all_paths[idx].split(os.sep)[-1]
                filename_right = image_right_all_paths[idx].split(os.sep)[-1]

                assert filename_left == filename_right

                for sparse_depth_left_path in sparse_depth_left_paths:
                    if filename_left in sparse_depth_left_path:
                        keep_indices.append(idx)
                        break

            # Sparse depth and ground truth are subset of images
            image_left_paths = [
                image_left_all_paths[idx] for idx in keep_indices
            ]
            image_left_paths = sorted(image_left_paths)

            image_right_paths = [
                image_right_all_paths[idx] for idx in keep_indices
            ]
            image_right_paths = sorted(image_right_paths)

            n_sample = len(sparse_depth_left_paths)

            # Check that data streams are aligned
            assert n_sample == len(sparse_depth_left_paths)
            assert n_sample == len(ground_truth_left_paths)
            assert n_sample == len(ground_truth_right_paths)
            assert n_sample == len(image_left_paths)
            assert n_sample == len(image_right_paths)

            data = zip(
                sparse_depth_left_paths,
                sparse_depth_right_paths,
                ground_truth_left_paths,
                ground_truth_right_paths,
                image_left_paths,
                image_right_paths)

            for datum in data:
                image_left_path, image_right_path = datum[-2:]

                filename_left = image_left_path.split(os.sep)[-1]
                filename_right = image_right_path.split(os.sep)[-1]

                assert filename_left == filename_right

                for path in datum:
                    assert filename_left in path

            # Same intrinsics, focal length and baseline for sequence
            intrinsics_left_paths = \
                [intrinsics_dkeys[(sequence_date, 'image_02')]] * n_sample
            intrinsics_right_paths = \
                [intrinsics_dkeys[(sequence_date, 'image_03')]] * n_sample

            focal_length_baseline_left_paths = \
                [focal_length_baseline_dkeys[(sequence_date, 'image_02')]] * n_sample
            focal_length_baseline_right_paths = \
                [focal_length_baseline_dkeys[(sequence_date, 'image_03')]] * n_sample

            # Add paths to running lists
            if refdir == 'train':
                data_left_paths = [
                    image_left_paths,
                    sparse_depth_left_paths,
                    ground_truth_left_paths,
                    focal_length_baseline_left_paths,
                    intrinsics_left_paths
                ]

                data_right_paths = [
                    image_right_paths,
                    sparse_depth_right_paths,
                    ground_truth_right_paths,
                    focal_length_baseline_right_paths,
                    intrinsics_right_paths
                ]

                # Save triplets of images as frames
                for camera_idx, data_paths in zip([0, 1], [data_left_paths, data_right_paths]):
                    image_paths, \
                        sparse_depth_paths, \
                        ground_truth_paths, \
                        focal_length_baseline_paths, \
                        intrinsics_paths = data_paths

                    # Save paths for supervised training
                    train_supervised_image_paths += image_paths
                    train_supervised_sparse_depth_paths += sparse_depth_left_paths
                    train_supervised_ground_truth_paths += ground_truth_paths
                    train_supervised_focal_length_baseline_paths += focal_length_baseline_paths
                    train_supervised_intrinsics_paths += intrinsics_paths

                    pool_inputs = []
                    for idx in range(len(sparse_depth_paths)):

                        # Get current, previous, and next frame
                        image_curr_path = image_paths[idx]

                        if camera_idx == 0:
                            image_idx = image_left_all_paths.index(image_curr_path)
                            image_prev_path = image_left_all_paths[image_idx-1]
                            image_next_path = image_left_all_paths[image_idx+1]
                        elif camera_idx == 1:
                            image_idx = image_right_all_paths.index(image_curr_path)
                            image_prev_path = image_right_all_paths[image_idx-1]
                            image_next_path = image_right_all_paths[image_idx+1]

                        sparse_depth_path = sparse_depth_paths[idx]
                        ground_truth_path = ground_truth_paths[idx]
                        focal_length_baseline_path = focal_length_baseline_paths[idx]
                        intrinsics_path = intrinsics_paths[idx]

                        pool_inputs.append((
                            image_curr_path,
                            image_prev_path,
                            image_next_path,
                            sparse_depth_path,
                            ground_truth_path,
                            focal_length_baseline_path,
                            intrinsics_path,
                            paths_only))

                    # Save paths for unsupervised training
                    with mp.Pool(n_thread) as pool:
                        pool_results = pool.map(process_frame, pool_inputs)

                        for result in pool_results:
                            images_path, \
                                sparse_depth_path, \
                                ground_truth_path, \
                                focal_length_baseline_path, \
                                intrinsics_path = result

                            # Save paths for left images
                            if camera_idx == 0:
                                train_unsupervised_images_left_paths.append(images_path)
                                train_unsupervised_sparse_depth_left_paths.append(sparse_depth_path)
                                train_unsupervised_ground_truth_left_paths.append(ground_truth_path)
                                train_unsupervised_intrinsics_left_paths.append(intrinsics_path)
                                train_unsupervised_focal_length_baseline_left_paths.append(focal_length_baseline_path)
                            # Save paths for right images
                            else:
                                train_unsupervised_images_right_paths.append(images_path)
                                train_unsupervised_sparse_depth_right_paths.append(sparse_depth_path)
                                train_unsupervised_ground_truth_right_paths.append(ground_truth_path)
                                train_unsupervised_focal_length_baseline_right_paths.append(focal_length_baseline_path)
                                train_unsupervised_intrinsics_right_paths.append(intrinsics_path)

            elif refdir == 'val':
                unused_image_left_paths += image_left_paths
                unused_image_right_paths += image_right_paths
                unused_sparse_depth_left_paths += sparse_depth_left_paths
                unused_sparse_depth_right_paths += sparse_depth_right_paths
                unused_ground_truth_left_paths += ground_truth_left_paths
                unused_ground_truth_right_paths += ground_truth_right_paths
                unused_focal_length_baseline_left_paths += focal_length_baseline_left_paths
                unused_focal_length_baseline_right_paths += focal_length_baseline_right_paths
                unused_intrinsics_left_paths += intrinsics_left_paths
                unused_intrinsics_right_paths += intrinsics_right_paths

            print('Processed {} samples using KITTI sequence={}'.format(
                n_sample, sequence_dirpath.split(os.sep)[-2]))

    '''
    Sort train paths alphabetically (multiprocessing might have messed it up)
    '''
    train_unsupervised_images_left_paths = sorted(train_unsupervised_images_left_paths)
    train_unsupervised_images_right_paths = sorted(train_unsupervised_images_right_paths)
    train_unsupervised_sparse_depth_left_paths = sorted(train_unsupervised_sparse_depth_left_paths)
    train_unsupervised_sparse_depth_right_paths = sorted(train_unsupervised_sparse_depth_right_paths)
    train_unsupervised_ground_truth_left_paths = sorted(train_unsupervised_ground_truth_left_paths)
    train_unsupervised_ground_truth_right_paths = sorted(train_unsupervised_ground_truth_right_paths)
    train_unsupervised_focal_length_baseline_left_paths = sorted(train_unsupervised_focal_length_baseline_left_paths)
    train_unsupervised_focal_length_baseline_right_paths = sorted(train_unsupervised_focal_length_baseline_right_paths)
    train_unsupervised_intrinsics_left_paths = sorted(train_unsupervised_intrinsics_left_paths)
    train_unsupervised_intrinsics_right_paths = sorted(train_unsupervised_intrinsics_right_paths)

    train_unsupervised_images_paths = \
         train_unsupervised_images_left_paths + train_unsupervised_images_right_paths
    train_unsupervised_sparse_depth_paths = \
        train_unsupervised_sparse_depth_left_paths + train_unsupervised_sparse_depth_right_paths
    train_unsupervised_ground_truth_paths = \
        train_unsupervised_ground_truth_left_paths + train_unsupervised_ground_truth_right_paths
    train_unsupervised_focal_length_baseline_paths = \
        train_unsupervised_focal_length_baseline_left_paths + train_unsupervised_focal_length_baseline_right_paths
    train_unsupervised_intrinsics_paths = \
        train_unsupervised_intrinsics_left_paths + train_unsupervised_intrinsics_right_paths

    '''
    Accumulate unused paths
    '''
    unused_image_paths = sorted(unused_image_left_paths + unused_image_right_paths)
    unused_sparse_depth_paths = sorted(unused_sparse_depth_left_paths + unused_sparse_depth_right_paths)
    unused_ground_truth_paths = sorted(unused_ground_truth_left_paths + unused_ground_truth_right_paths)
    unused_focal_length_baseline_paths = sorted(unused_focal_length_baseline_left_paths + unused_focal_length_baseline_right_paths)
    unused_intrinsics_paths = sorted(unused_intrinsics_left_paths + unused_intrinsics_right_paths)

    '''
    Save separate clean data (static scenes only)
    '''
    # Filter clean data from left images
    train_data_left_paths = [
        train_unsupervised_images_left_paths,
        train_unsupervised_sparse_depth_left_paths,
        train_unsupervised_ground_truth_left_paths,
        train_unsupervised_focal_length_baseline_left_paths,
        train_unsupervised_intrinsics_left_paths
    ]

    train_unsupervised_nonstatic_images_left_paths, \
        train_unsupervised_nonstatic_sparse_depth_left_paths, \
        train_unsupervised_nonstatic_ground_truth_left_paths, \
        train_unsupervised_nonstatic_focal_length_baseline_left_paths, \
        train_unsupervised_nonstatic_intrinsics_left_paths = filter_static_frames(train_data_left_paths)

    # Filter clean data from right images
    train_data_right_paths = [
        train_unsupervised_images_right_paths,
        train_unsupervised_sparse_depth_right_paths,
        train_unsupervised_ground_truth_right_paths,
        train_unsupervised_focal_length_baseline_right_paths,
        train_unsupervised_intrinsics_right_paths
    ]

    train_unsupervised_nonstatic_images_right_paths, \
        train_unsupervised_nonstatic_sparse_depth_right_paths, \
        train_unsupervised_nonstatic_ground_truth_right_paths, \
        train_unsupervised_nonstatic_focal_length_baseline_right_paths, \
        train_unsupervised_nonstatic_intrinsics_right_paths = filter_static_frames(train_data_right_paths)

    # Sanity checks
    n_nonstatic_paths = len(train_unsupervised_nonstatic_images_left_paths) + \
        len(train_unsupervised_nonstatic_images_right_paths)

    assert n_nonstatic_paths == len(train_unsupervised_nonstatic_sparse_depth_left_paths) + \
        len(train_unsupervised_nonstatic_sparse_depth_right_paths)
    assert n_nonstatic_paths == len(train_unsupervised_nonstatic_ground_truth_left_paths) + \
        len(train_unsupervised_nonstatic_ground_truth_right_paths)
    assert n_nonstatic_paths == len(train_unsupervised_nonstatic_focal_length_baseline_left_paths) + \
        len(train_unsupervised_nonstatic_focal_length_baseline_right_paths)
    assert n_nonstatic_paths == len(train_unsupervised_nonstatic_intrinsics_left_paths) + \
        len(train_unsupervised_nonstatic_intrinsics_right_paths)

    train_unsupervised_nonstatic_images_paths = \
        train_unsupervised_nonstatic_images_left_paths + train_unsupervised_nonstatic_images_right_paths
    train_unsupervised_nonstatic_sparse_depth_paths = \
        train_unsupervised_nonstatic_sparse_depth_left_paths + train_unsupervised_nonstatic_sparse_depth_right_paths
    train_unsupervised_nonstatic_ground_truth_paths = \
        train_unsupervised_nonstatic_ground_truth_left_paths + train_unsupervised_nonstatic_ground_truth_right_paths
    train_unsupervised_nonstatic_focal_length_baseline_paths = \
        train_unsupervised_nonstatic_focal_length_baseline_left_paths + train_unsupervised_nonstatic_focal_length_baseline_right_paths
    train_unsupervised_nonstatic_intrinsics_paths = \
        train_unsupervised_nonstatic_intrinsics_left_paths + train_unsupervised_nonstatic_intrinsics_right_paths

    '''
    Write supervised training paths to file
    '''
    print('Storing {} supervised training image file paths into: {}'.format(
        len(train_supervised_image_paths), TRAIN_SUPERVISED_IMAGE_FILEPATH))
    data_utils.write_paths(TRAIN_SUPERVISED_IMAGE_FILEPATH, train_supervised_image_paths)

    print('Storing {} supervised training sparse depth file paths into: {}'.format(
        len(train_supervised_sparse_depth_paths), TRAIN_SUPERVISED_SPARSE_DEPTH_FILEPATH))
    data_utils.write_paths(TRAIN_SUPERVISED_SPARSE_DEPTH_FILEPATH, train_supervised_sparse_depth_paths)

    print('Storing {} supervised training ground truth file paths into: {}'.format(
        len(train_supervised_ground_truth_paths), TRAIN_SUPERVISED_GROUND_TRUTH_FILEPATH))
    data_utils.write_paths(TRAIN_SUPERVISED_GROUND_TRUTH_FILEPATH, train_supervised_ground_truth_paths)

    print('Storing {} supervised training focal length and baseline file paths into: {}'.format(
        len(train_supervised_focal_length_baseline_paths), TRAIN_SUPERVISED_FOCAL_LENGTH_BASELINE_FILEPATH))
    data_utils.write_paths(TRAIN_SUPERVISED_FOCAL_LENGTH_BASELINE_FILEPATH, train_supervised_focal_length_baseline_paths)

    print('Storing {} supervised training intrinsics file paths into: {}'.format(
        len(train_supervised_intrinsics_paths), TRAIN_SUPERVISED_INTRINSICS_FILEPATH))
    data_utils.write_paths(TRAIN_SUPERVISED_INTRINSICS_FILEPATH, train_supervised_intrinsics_paths)

    '''
    Write unsupervised training paths to file
    '''
    # All unsupervised paths
    print('Storing {} unsupervised training images file paths into: {}'.format(
        len(train_unsupervised_images_paths), TRAIN_UNSUPERVISED_IMAGES_FILEPATH))
    data_utils.write_paths(TRAIN_UNSUPERVISED_IMAGES_FILEPATH, train_unsupervised_images_paths)

    print('Storing {} unsupervised training sparse depth file paths into: {}'.format(
        len(train_unsupervised_sparse_depth_paths), TRAIN_UNSUPERVISED_SPARSE_DEPTH_FILEPATH))
    data_utils.write_paths(TRAIN_UNSUPERVISED_SPARSE_DEPTH_FILEPATH, train_unsupervised_sparse_depth_paths)

    print('Storing {} unsupervised training ground truth file paths into: {}'.format(
        len(train_unsupervised_ground_truth_paths), TRAIN_UNSUPERVISED_GROUND_TRUTH_FILEPATH))
    data_utils.write_paths(TRAIN_UNSUPERVISED_GROUND_TRUTH_FILEPATH, train_unsupervised_ground_truth_paths)

    print('Storing {} unsupervised training focal length and baseline file paths into: {}'.format(
        len(train_unsupervised_focal_length_baseline_paths), TRAIN_UNSUPERVISED_FOCAL_LENGTH_BASELINE_FILEPATH))
    data_utils.write_paths(TRAIN_UNSUPERVISED_FOCAL_LENGTH_BASELINE_FILEPATH, train_unsupervised_focal_length_baseline_paths)

    print('Storing {} unsupervised training intrinsics file paths into: {}'.format(
        len(train_unsupervised_intrinsics_paths), TRAIN_UNSUPERVISED_INTRINSICS_FILEPATH))
    data_utils.write_paths(TRAIN_UNSUPERVISED_INTRINSICS_FILEPATH, train_unsupervised_intrinsics_paths)

    # All nonstatic unsupervised paths
    print('Storing {} nonstatic unsupervised training images file paths into: {}'.format(
        len(train_unsupervised_nonstatic_images_paths), TRAIN_UNSUPERVISED_NONSTATIC_IMAGES_FILEPATH))
    data_utils.write_paths(TRAIN_UNSUPERVISED_NONSTATIC_IMAGES_FILEPATH, train_unsupervised_nonstatic_images_paths)

    print('Storing {} nonstatic unsupervised training sparse depth file paths into: {}'.format(
        len(train_unsupervised_nonstatic_sparse_depth_paths), TRAIN_UNSUPERVISED_NONSTATIC_SPARSE_DEPTH_FILEPATH))
    data_utils.write_paths(TRAIN_UNSUPERVISED_NONSTATIC_SPARSE_DEPTH_FILEPATH, train_unsupervised_nonstatic_sparse_depth_paths)

    print('Storing {} nonstatic unsupervised training ground truth file paths into: {}'.format(
        len(train_unsupervised_nonstatic_ground_truth_paths), TRAIN_UNSUPERVISED_NONSTATIC_GROUND_TRUTH_FILEPATH))
    data_utils.write_paths(TRAIN_UNSUPERVISED_NONSTATIC_GROUND_TRUTH_FILEPATH, train_unsupervised_nonstatic_ground_truth_paths)

    print('Storing {} nonstatic unsupervised training focal length and baseline file paths into: {}'.format(
        len(train_unsupervised_nonstatic_focal_length_baseline_paths), TRAIN_UNSUPERVISED_NONSTATIC_FOCAL_LENGTH_BASELINE_FILEPATH))
    data_utils.write_paths(TRAIN_UNSUPERVISED_NONSTATIC_FOCAL_LENGTH_BASELINE_FILEPATH, train_unsupervised_nonstatic_focal_length_baseline_paths)

    print('Storing {} nonstatic unsupervised training intrinsics file paths into: {}'.format(
        len(train_unsupervised_nonstatic_intrinsics_paths), TRAIN_UNSUPERVISED_NONSTATIC_INTRINSICS_FILEPATH))
    data_utils.write_paths(TRAIN_UNSUPERVISED_NONSTATIC_INTRINSICS_FILEPATH, train_unsupervised_nonstatic_intrinsics_paths)

    # All unused paths
    print('Storing {} unused images file paths into: {}'.format(
        len(unused_image_paths), UNUSED_IMAGE_FILEPATH))
    data_utils.write_paths(UNUSED_IMAGE_FILEPATH, unused_image_paths)

    print('Storing {} unused sparse depth file paths into: {}'.format(
        len(unused_sparse_depth_paths), UNUSED_SPARSE_DEPTH_FILEPATH))
    data_utils.write_paths(UNUSED_SPARSE_DEPTH_FILEPATH, unused_sparse_depth_paths)

    print('Storing {} unused ground truth file paths into: {}'.format(
        len(unused_ground_truth_paths), UNUSED_GROUND_TRUTH_FILEPATH))
    data_utils.write_paths(UNUSED_GROUND_TRUTH_FILEPATH, unused_ground_truth_paths)

    print('Storing {} unused focal length and baseline file paths into: {}'.format(
        len(unused_focal_length_baseline_paths), UNUSED_FOCAL_LENGTH_BASELINE_FILEPATH))
    data_utils.write_paths(UNUSED_FOCAL_LENGTH_BASELINE_FILEPATH, unused_focal_length_baseline_paths)

    print('Storing {} unused intrinsics file paths into: {}'.format(
        len(unused_intrinsics_paths), UNUSED_INTRINSICS_FILEPATH))
    data_utils.write_paths(UNUSED_INTRINSICS_FILEPATH, unused_intrinsics_paths)

    # Separate left and right camera paths
    train_filepaths = [
        (train_unsupervised_images_left_paths, TRAIN_UNSUPERVISED_IMAGES_LEFT_FILEPATH),
        (train_unsupervised_images_right_paths, TRAIN_UNSUPERVISED_IMAGES_RIGHT_FILEPATH),
        (train_unsupervised_sparse_depth_left_paths, TRAIN_UNSUPERVISED_SPARSE_DEPTH_LEFT_FILEPATH),
        (train_unsupervised_sparse_depth_right_paths, TRAIN_UNSUPERVISED_SPARSE_DEPTH_RIGHT_FILEPATH),
        (train_unsupervised_ground_truth_left_paths, TRAIN_UNSUPERVISED_GROUND_TRUTH_LEFT_FILEPATH),
        (train_unsupervised_ground_truth_right_paths, TRAIN_UNSUPERVISED_GROUND_TRUTH_RIGHT_FILEPATH),
        (train_unsupervised_focal_length_baseline_left_paths, TRAIN_UNSUPERVISED_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH),
        (train_unsupervised_focal_length_baseline_right_paths, TRAIN_UNSUPERVISED_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH),
        (train_unsupervised_intrinsics_left_paths, TRAIN_UNSUPERVISED_INTRINSICS_LEFT_FILEPATH),
        (train_unsupervised_intrinsics_right_paths, TRAIN_UNSUPERVISED_INTRINSICS_RIGHT_FILEPATH)
    ]

    train_nonstatic_filepaths = [
        (train_unsupervised_nonstatic_images_left_paths, TRAIN_UNSUPERVISED_NONSTATIC_IMAGES_LEFT_FILEPATH),
        (train_unsupervised_nonstatic_images_right_paths, TRAIN_UNSUPERVISED_NONSTATIC_IMAGES_RIGHT_FILEPATH),
        (train_unsupervised_nonstatic_sparse_depth_left_paths, TRAIN_UNSUPERVISED_NONSTATIC_SPARSE_DEPTH_LEFT_FILEPATH),
        (train_unsupervised_nonstatic_sparse_depth_right_paths, TRAIN_UNSUPERVISED_NONSTATIC_SPARSE_DEPTH_RIGHT_FILEPATH),
        (train_unsupervised_nonstatic_ground_truth_left_paths, TRAIN_UNSUPERVISED_NONSTATIC_GROUND_TRUTH_LEFT_FILEPATH),
        (train_unsupervised_nonstatic_ground_truth_right_paths, TRAIN_UNSUPERVISED_NONSTATIC_GROUND_TRUTH_RIGHT_FILEPATH),
        (train_unsupervised_nonstatic_focal_length_baseline_left_paths, TRAIN_UNSUPERVISED_NONSTATIC_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH),
        (train_unsupervised_nonstatic_focal_length_baseline_right_paths, TRAIN_UNSUPERVISED_NONSTATIC_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH),
        (train_unsupervised_nonstatic_intrinsics_left_paths, TRAIN_UNSUPERVISED_NONSTATIC_INTRINSICS_LEFT_FILEPATH),
        (train_unsupervised_nonstatic_intrinsics_right_paths, TRAIN_UNSUPERVISED_NONSTATIC_INTRINSICS_RIGHT_FILEPATH)
    ]

    unused_filepaths = [
        (unused_image_left_paths, UNUSED_IMAGE_LEFT_FILEPATH),
        (unused_image_right_paths, UNUSED_IMAGE_RIGHT_FILEPATH),
        (unused_sparse_depth_left_paths, UNUSED_SPARSE_DEPTH_LEFT_FILEPATH),
        (unused_sparse_depth_right_paths, UNUSED_SPARSE_DEPTH_RIGHT_FILEPATH),
        (unused_ground_truth_left_paths, UNUSED_GROUND_TRUTH_LEFT_FILEPATH),
        (unused_ground_truth_right_paths, UNUSED_GROUND_TRUTH_RIGHT_FILEPATH),
        (unused_focal_length_baseline_left_paths, UNUSED_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH),
        (unused_focal_length_baseline_right_paths, UNUSED_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH),
        (unused_intrinsics_left_paths, UNUSED_INTRINSICS_LEFT_FILEPATH),
        (unused_intrinsics_right_paths, UNUSED_INTRINSICS_RIGHT_FILEPATH)
    ]

    tags = ['training', 'nonstatic', 'unused']
    data_filepaths = [train_filepaths, train_nonstatic_filepaths, unused_filepaths]

    for tag, filepaths in zip(tags, data_filepaths):

        (image_left_paths, image_left_filepath), \
            (image_right_paths, image_right_filepath), \
            (sparse_depth_left_paths, sparse_depth_left_filepath), \
            (sparse_depth_right_paths, sparse_depth_right_filepath), \
            (ground_truth_left_paths, ground_truth_left_filepath), \
            (ground_truth_right_paths, ground_truth_right_filepath), \
            (focal_length_baseline_left_paths, focal_length_baseline_left_filepath), \
            (focal_length_baseline_right_paths, focal_length_baseline_right_filepath), \
            (intrinsics_left_paths, intrinsics_left_filepath), \
            (intrinsics_right_paths, intrinsics_right_filepath) = filepaths

        print('Storing {} left {} stereo image file paths into: {}'.format(
            len(image_left_paths), tag, image_left_filepath))
        data_utils.write_paths(image_left_filepath, image_left_paths)

        print('Storing {} right {} stereo image file paths into: {}'.format(
            len(image_right_paths), tag, image_right_filepath))
        data_utils.write_paths(image_right_filepath, image_right_paths)

        print('Storing {} left {} sparse depth file paths into: {}'.format(
            len(sparse_depth_left_paths), tag, sparse_depth_left_filepath))
        data_utils.write_paths(sparse_depth_left_filepath, sparse_depth_left_paths)

        print('Storing {} right {} sparse depth file paths into: {}'.format(
            len(sparse_depth_right_paths), tag, sparse_depth_right_filepath))
        data_utils.write_paths(sparse_depth_right_filepath, sparse_depth_right_paths)

        print('Storing {} left {} ground truth file paths into: {}'.format(
            len(ground_truth_left_paths), tag, ground_truth_left_filepath))
        data_utils.write_paths(ground_truth_left_filepath, ground_truth_left_paths)

        print('Storing {} right {} ground truth file paths into: {}'.format(
            len(ground_truth_right_paths), tag, ground_truth_right_filepath))
        data_utils.write_paths(ground_truth_right_filepath, ground_truth_right_paths)

        print('Storing {} left {} focal length baseline file paths into: {}'.format(
            len(focal_length_baseline_left_paths), tag, focal_length_baseline_left_filepath))
        data_utils.write_paths(focal_length_baseline_left_filepath, focal_length_baseline_left_paths)

        print('Storing {} right {} focal length baseline file paths into: {}'.format(
            len(focal_length_baseline_right_paths), tag, focal_length_baseline_right_filepath))
        data_utils.write_paths(focal_length_baseline_right_filepath, focal_length_baseline_right_paths)

        print('Storing {} left {} intrinsics file paths into: {}'.format(
            len(intrinsics_left_paths), tag, intrinsics_left_filepath))
        data_utils.write_paths(intrinsics_left_filepath, intrinsics_left_paths)

        print('Storing {} right {} intrinsics file paths into: {}'.format(
            len(intrinsics_right_paths), tag, intrinsics_right_filepath))
        data_utils.write_paths(intrinsics_right_filepath, intrinsics_right_paths)


def setup_dataset_kitti_validation_testing(paths_only=False):
    '''
    Fetch image, sparse depth, and ground truth paths for validation and testing

    Arg(s):
        paths_only : bool
            if set, then only produces paths
    '''

    val_image_paths = []
    val_sparse_depth_paths = []
    val_ground_truth_paths = []
    val_intrinsics_paths = []

    test_image_paths = []
    test_sparse_depth_paths = []
    test_ground_truth_paths = []
    test_intrinsics_paths = []

    modes = [
        [
            'validation',
            KITTI_VALIDATION_DIRPATH,
            (val_image_paths, VAL_IMAGE_FILEPATH),
            (val_sparse_depth_paths, VAL_SPARSE_DEPTH_FILEPATH),
            (val_ground_truth_paths, VAL_GROUND_TRUTH_FILEPATH),
            (val_intrinsics_paths, VAL_INTRINSICS_FILEPATH)
        ],
        [
            'testing',
            KITTI_TESTING_DIRPATH,
            (test_image_paths, TEST_IMAGE_FILEPATH),
            (test_sparse_depth_paths, TEST_SPARSE_DEPTH_FILEPATH),
            (test_ground_truth_paths, TEST_GROUND_TRUTH_FILEPATH),
            (test_intrinsics_paths, TEST_INTRINSICS_FILEPATH)
        ]
    ]

    for mode in modes:
        tag, \
            kitti_dirpath, \
            (image_paths, image_filepath), \
            (sparse_depth_paths, sparse_depth_filepath), \
            (ground_truth_paths, ground_truth_filepath), \
            (intrinsics_paths, intrinsics_filepath) = mode

        # Iterate through image, intrinsics, sparse depth and ground-truth directories
        for refdir in ['image', 'intrinsics', 'sparse_depth', 'ground_truth']:

            ext = '*.txt' if refdir == 'intrinsics' else '*.png'

            filepaths = sorted(glob.glob(
                os.path.join(kitti_dirpath, refdir, ext)))

            # Iterate filepaths
            for idx in range(len(filepaths)):
                path = filepaths[idx]

                if refdir == 'image':
                    image_paths.append(path)

                elif refdir == 'intrinsics':
                    intrinsics = np.reshape(np.loadtxt(path), (3, 3))

                    intrinsics_path = path \
                        .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_DERIVED_DIRPATH) \
                        .replace('.txt', '.npy')

                    intrinsics_paths.append(intrinsics_path)

                    if not os.path.exists(os.path.dirname(intrinsics_path)):
                        os.makedirs(os.path.dirname(intrinsics_path))

                    if not paths_only:
                        np.save(intrinsics_path, intrinsics)

                elif refdir == 'sparse_depth':
                    sparse_depth_paths.append(path)

                elif refdir == 'ground_truth':
                    ground_truth_paths.append(path)

                print('Processed {}/{} {} {} samples \r'.format(
                    idx + 1, len(filepaths), tag, refdir),
                    end='\r')

            print('Completed generating {} {} {} samples'.format(
                len(filepaths), tag, refdir))

        print('Storing {} {} image file paths into: {}'.format(
            len(image_paths), tag, image_filepath))
        data_utils.write_paths(image_filepath, image_paths)

        print('Storing {} {} sparse depth file paths into: {}'.format(
            len(sparse_depth_paths), tag, sparse_depth_filepath))
        data_utils.write_paths(sparse_depth_filepath, sparse_depth_paths)

        print('Storing {} {} ground truth file paths into: {}'.format(
            len(ground_truth_paths), tag, ground_truth_filepath))
        data_utils.write_paths(ground_truth_filepath, ground_truth_paths)

        print('Storing {} {} intrinsics file paths into: {}'.format(
            len(intrinsics_paths), tag, intrinsics_filepath))
        data_utils.write_paths(intrinsics_filepath, intrinsics_paths)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--paths_only', action='store_true', help='If set, then generate paths only')
    parser.add_argument('--n_thread',  type=int, default=8)

    args = parser.parse_args()

    dirpaths = [
        TRAIN_SUPERVISED_REF_DIRPATH,
        TRAIN_UNSUPERVISED_REF_DIRPATH,
        TRAIN_UNUSED_REF_DIRPATH,
        VAL_REF_DIRPATH,
        TEST_REF_DIRPATH
    ]

    # Create directories for output files
    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    # Set up dataset
    setup_dataset_kitti_training(args.paths_only, args.n_thread)

    setup_dataset_kitti_validation_testing(args.paths_only)
