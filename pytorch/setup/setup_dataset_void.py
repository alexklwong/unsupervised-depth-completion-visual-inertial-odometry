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

import os, sys, glob, argparse
import multiprocessing as mp
import numpy as np
import cv2
sys.path.insert(0, 'src')
import data_utils


VOID_ROOT_DIRPATH = os.path.join('data', 'void_release')
VOID_DATA_150_DIRPATH  = os.path.join(VOID_ROOT_DIRPATH, 'void_150')
VOID_DATA_500_DIRPATH = os.path.join(VOID_ROOT_DIRPATH, 'void_500')
VOID_DATA_1500_DIRPATH = os.path.join(VOID_ROOT_DIRPATH, 'void_1500')

VOID_OUTPUT_DIRPATH = os.path.join('data', 'void_derived')

VOID_TRAIN_IMAGE_FILENAME = 'train_image.txt'
VOID_TRAIN_SPARSE_DEPTH_FILENAME = 'train_sparse_depth.txt'
VOID_TRAIN_VALIDITY_MAP_FILENAME = 'train_validity_map.txt'
VOID_TRAIN_GROUND_TRUTH_FILENAME = 'train_ground_truth.txt'
VOID_TRAIN_INTRINSICS_FILENAME = 'train_intrinsics.txt'
VOID_TEST_IMAGE_FILENAME = 'test_image.txt'
VOID_TEST_SPARSE_DEPTH_FILENAME = 'test_sparse_depth.txt'
VOID_TEST_VALIDITY_MAP_FILENAME = 'test_validity_map.txt'
VOID_TEST_GROUND_TRUTH_FILENAME = 'test_ground_truth.txt'
VOID_TEST_INTRINSICS_FILENAME = 'test_intrinsics.txt'

TRAIN_REFS_DIRPATH = os.path.join('training', 'void')
TEST_REFS_DIRPATH = os.path.join('testing', 'void')

TRAIN_SUPERVISED_REFS_DIRPATH = os.path.join(TRAIN_REFS_DIRPATH, 'supervised')
TRAIN_UNSUPERVISED_REFS_DIRPATH = os.path.join(TRAIN_REFS_DIRPATH, 'unsupervised')

'''
Paths to files for supervised training
'''
# VOID training set 150 density
VOID_TRAIN_SUPERVISED_IMAGE_150_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_image_150.txt')
VOID_TRAIN_SUPERVISED_SPARSE_DEPTH_150_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_sparse_depth_150.txt')
VOID_TRAIN_SUPERVISED_VALIDITY_MAP_150_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_validity_map_150.txt')
VOID_TRAIN_SUPERVISED_GROUND_TRUTH_150_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_ground_truth_150.txt')
VOID_TRAIN_SUPERVISED_INTRINSICS_150_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_intrinsics_150.txt')
# VOID training set 500 density
VOID_TRAIN_SUPERVISED_IMAGE_500_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_image_500.txt')
VOID_TRAIN_SUPERVISED_SPARSE_DEPTH_500_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_sparse_depth_500.txt')
VOID_TRAIN_SUPERVISED_VALIDITY_MAP_500_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_validity_map_500.txt')
VOID_TRAIN_SUPERVISED_GROUND_TRUTH_500_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_ground_truth_500.txt')
VOID_TRAIN_SUPERVISED_INTRINSICS_500_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_intrinsics_500.txt')
# VOID training set 1500 density
VOID_TRAIN_SUPERVISED_IMAGE_1500_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_image_1500.txt')
VOID_TRAIN_SUPERVISED_SPARSE_DEPTH_1500_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_sparse_depth_1500.txt')
VOID_TRAIN_SUPERVISED_VALIDITY_MAP_1500_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_validity_map_1500.txt')
VOID_TRAIN_SUPERVISED_GROUND_TRUTH_1500_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_ground_truth_1500.txt')
VOID_TRAIN_SUPERVISED_INTRINSICS_1500_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REFS_DIRPATH, 'void_train_intrinsics_1500.txt')

'''
Paths to files for unsupervised training
'''
# VOID training set 150 density
VOID_TRAIN_UNSUPERVISED_IMAGE_150_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_image_150.txt')
VOID_TRAIN_UNSUPERVISED_SPARSE_DEPTH_150_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_sparse_depth_150.txt')
VOID_TRAIN_UNSUPERVISED_VALIDITY_MAP_150_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_validity_map_150.txt')
VOID_TRAIN_UNSUPERVISED_GROUND_TRUTH_150_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_ground_truth_150.txt')
VOID_TRAIN_UNSUPERVISED_INTRINSICS_150_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_intrinsics_150.txt')
# VOID training set 500 density
VOID_TRAIN_UNSUPERVISED_IMAGE_500_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_image_500.txt')
VOID_TRAIN_UNSUPERVISED_SPARSE_DEPTH_500_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_sparse_depth_500.txt')
VOID_TRAIN_UNSUPERVISED_VALIDITY_MAP_500_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_validity_map_500.txt')
VOID_TRAIN_UNSUPERVISED_GROUND_TRUTH_500_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_ground_truth_500.txt')
VOID_TRAIN_UNSUPERVISED_INTRINSICS_500_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_intrinsics_500.txt')
# VOID training set 1500 density
VOID_TRAIN_UNSUPERVISED_IMAGE_1500_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_image_1500.txt')
VOID_TRAIN_UNSUPERVISED_SPARSE_DEPTH_1500_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_sparse_depth_1500.txt')
VOID_TRAIN_UNSUPERVISED_VALIDITY_MAP_1500_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_validity_map_1500.txt')
VOID_TRAIN_UNSUPERVISED_GROUND_TRUTH_1500_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_ground_truth_1500.txt')
VOID_TRAIN_UNSUPERVISED_INTRINSICS_1500_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REFS_DIRPATH, 'void_train_intrinsics_1500.txt')

'''
Paths to files for testing
'''
# VOID testing set 150 density
VOID_TEST_IMAGE_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_image_150.txt')
VOID_TEST_SPARSE_DEPTH_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_sparse_depth_150.txt')
VOID_TEST_VALIDITY_MAP_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_validity_map_150.txt')
VOID_TEST_GROUND_TRUTH_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_ground_truth_150.txt')
VOID_TEST_INTRINSICS_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_intrinsics_150.txt')
# VOID testing set 500 density
VOID_TEST_IMAGE_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_image_500.txt')
VOID_TEST_SPARSE_DEPTH_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_sparse_depth_500.txt')
VOID_TEST_VALIDITY_MAP_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_validity_map_500.txt')
VOID_TEST_GROUND_TRUTH_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_ground_truth_500.txt')
VOID_TEST_INTRINSICS_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_intrinsics_500.txt')
# VOID testing set 1500 density
VOID_TEST_IMAGE_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_image_1500.txt')
VOID_TEST_SPARSE_DEPTH_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_sparse_depth_1500.txt')
VOID_TEST_VALIDITY_MAP_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_validity_map_1500.txt')
VOID_TEST_GROUND_TRUTH_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_ground_truth_1500.txt')
VOID_TEST_INTRINSICS_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_test_intrinsics_1500.txt')
# VOID unused testing set 150 density
VOID_UNUSED_IMAGE_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_image_150.txt')
VOID_UNUSED_SPARSE_DEPTH_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_sparse_depth_150.txt')
VOID_UNUSED_VALIDITY_MAP_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_validity_map_150.txt')
VOID_UNUSED_GROUND_TRUTH_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_ground_truth_150.txt')
VOID_UNUSED_INTRINSICS_150_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_intrinsics_150.txt')
# VOID unused testing set 500 density
VOID_UNUSED_IMAGE_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_image_500.txt')
VOID_UNUSED_SPARSE_DEPTH_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_sparse_depth_500.txt')
VOID_UNUSED_VALIDITY_MAP_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_validity_map_500.txt')
VOID_UNUSED_GROUND_TRUTH_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_ground_truth_500.txt')
VOID_UNUSED_INTRINSICS_500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_intrinsics_500.txt')
# VOID unused testing set 1500 density
VOID_UNUSED_IMAGE_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_image_1500.txt')
VOID_UNUSED_SPARSE_DEPTH_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_sparse_depth_1500.txt')
VOID_UNUSED_VALIDITY_MAP_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_validity_map_1500.txt')
VOID_UNUSED_GROUND_TRUTH_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_ground_truth_1500.txt')
VOID_UNUSED_INTRINSICS_1500_FILEPATH = \
    os.path.join(TEST_REFS_DIRPATH, 'void_unused_intrinsics_1500.txt')


def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple
            image path at time t=0,
            image path at time t=1,
            image path at time t=-1,
            sparse depth path at time t=0,
            validity map path at time t=0,
            ground truth path at time t=0,
            boolean flag if set then create image triplets (for training)
            boolean flag if set then create paths only
    Returns:
        str : image reference directory path
        str : output concatenated image path at time t=0
        str : output sparse depth path at time t=0
        str : output validity map path at time t=0
        str : output ground truth path at time t=0
    '''

    image_path1, \
        image_path0, \
        image_path2, \
        sparse_depth_path, \
        validity_map_path, \
        ground_truth_path, \
        create_triplet, \
        paths_only = inputs

    if not paths_only:
        # Create image composite of triplets
        if create_triplet:
            image1 = cv2.imread(image_path1)
            image0 = cv2.imread(image_path0)
            image2 = cv2.imread(image_path2)
            imagec = np.concatenate([image1, image0, image2], axis=1)
        else:
            imagec = cv2.imread(image_path0)

    image_refpath = os.path.join(*image_path0.split(os.sep)[2:])

    # Set output paths
    image_outpath = os.path.join(VOID_OUTPUT_DIRPATH, image_refpath)
    sparse_depth_outpath = sparse_depth_path
    validity_map_outpath = validity_map_path
    ground_truth_outpath = ground_truth_path

    # Verify that all filenames match
    _, image_filename = os.path.split(image_outpath)
    sparse_depth_filename = os.path.basename(sparse_depth_outpath)
    validity_map_filename = os.path.basename(validity_map_outpath)
    ground_truth_filename = os.path.basename(ground_truth_outpath)

    assert image_filename == sparse_depth_filename
    assert image_filename == validity_map_filename
    assert image_filename == ground_truth_filename

    if not paths_only:
        cv2.imwrite(image_outpath, imagec)

    return (image_refpath,
            image_outpath,
            sparse_depth_outpath,
            validity_map_outpath,
            ground_truth_outpath)


parser = argparse.ArgumentParser()

parser.add_argument('--paths_only', action='store_true')

args = parser.parse_args()


data_dirpaths = [
    VOID_DATA_150_DIRPATH,
    VOID_DATA_500_DIRPATH,
    VOID_DATA_1500_DIRPATH
]

train_supervised_output_filepaths = [
    [
        VOID_TRAIN_SUPERVISED_IMAGE_150_FILEPATH,
        VOID_TRAIN_SUPERVISED_SPARSE_DEPTH_150_FILEPATH,
        VOID_TRAIN_SUPERVISED_VALIDITY_MAP_150_FILEPATH,
        VOID_TRAIN_SUPERVISED_GROUND_TRUTH_150_FILEPATH,
        VOID_TRAIN_SUPERVISED_INTRINSICS_150_FILEPATH
    ],
    [
        VOID_TRAIN_SUPERVISED_IMAGE_500_FILEPATH,
        VOID_TRAIN_SUPERVISED_SPARSE_DEPTH_500_FILEPATH,
        VOID_TRAIN_SUPERVISED_VALIDITY_MAP_500_FILEPATH,
        VOID_TRAIN_SUPERVISED_GROUND_TRUTH_500_FILEPATH,
        VOID_TRAIN_SUPERVISED_INTRINSICS_500_FILEPATH
    ],
    [
        VOID_TRAIN_SUPERVISED_IMAGE_1500_FILEPATH,
        VOID_TRAIN_SUPERVISED_SPARSE_DEPTH_1500_FILEPATH,
        VOID_TRAIN_SUPERVISED_VALIDITY_MAP_1500_FILEPATH,
        VOID_TRAIN_SUPERVISED_GROUND_TRUTH_1500_FILEPATH,
        VOID_TRAIN_SUPERVISED_INTRINSICS_1500_FILEPATH
    ]
]
train_unsupervised_output_filepaths = [
    [
        VOID_TRAIN_UNSUPERVISED_IMAGE_150_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_SPARSE_DEPTH_150_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_VALIDITY_MAP_150_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_GROUND_TRUTH_150_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_INTRINSICS_150_FILEPATH
    ],
    [
        VOID_TRAIN_UNSUPERVISED_IMAGE_500_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_SPARSE_DEPTH_500_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_VALIDITY_MAP_500_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_GROUND_TRUTH_500_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_INTRINSICS_500_FILEPATH
    ],
    [
        VOID_TRAIN_UNSUPERVISED_IMAGE_1500_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_SPARSE_DEPTH_1500_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_VALIDITY_MAP_1500_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_GROUND_TRUTH_1500_FILEPATH,
        VOID_TRAIN_UNSUPERVISED_INTRINSICS_1500_FILEPATH
    ]
]
test_output_filepaths = [
    [
        VOID_TEST_IMAGE_150_FILEPATH,
        VOID_TEST_SPARSE_DEPTH_150_FILEPATH,
        VOID_TEST_VALIDITY_MAP_150_FILEPATH,
        VOID_TEST_GROUND_TRUTH_150_FILEPATH,
        VOID_TEST_INTRINSICS_150_FILEPATH
    ],
    [
        VOID_TEST_IMAGE_500_FILEPATH,
        VOID_TEST_SPARSE_DEPTH_500_FILEPATH,
        VOID_TEST_VALIDITY_MAP_500_FILEPATH,
        VOID_TEST_GROUND_TRUTH_500_FILEPATH,
        VOID_TEST_INTRINSICS_500_FILEPATH
    ],
    [
        VOID_TEST_IMAGE_1500_FILEPATH,
        VOID_TEST_SPARSE_DEPTH_1500_FILEPATH,
        VOID_TEST_VALIDITY_MAP_1500_FILEPATH,
        VOID_TEST_GROUND_TRUTH_1500_FILEPATH,
        VOID_TEST_INTRINSICS_1500_FILEPATH
    ]
]
unused_output_filepaths = [
    [
        VOID_UNUSED_IMAGE_150_FILEPATH,
        VOID_UNUSED_SPARSE_DEPTH_150_FILEPATH,
        VOID_UNUSED_VALIDITY_MAP_150_FILEPATH,
        VOID_UNUSED_GROUND_TRUTH_150_FILEPATH,
        VOID_UNUSED_INTRINSICS_150_FILEPATH
    ],
    [
        VOID_UNUSED_IMAGE_500_FILEPATH,
        VOID_UNUSED_SPARSE_DEPTH_500_FILEPATH,
        VOID_UNUSED_VALIDITY_MAP_500_FILEPATH,
        VOID_UNUSED_GROUND_TRUTH_500_FILEPATH,
        VOID_UNUSED_INTRINSICS_500_FILEPATH
    ],
    [
        VOID_UNUSED_IMAGE_1500_FILEPATH,
        VOID_UNUSED_SPARSE_DEPTH_1500_FILEPATH,
        VOID_UNUSED_VALIDITY_MAP_1500_FILEPATH,
        VOID_UNUSED_GROUND_TRUTH_1500_FILEPATH,
        VOID_UNUSED_INTRINSICS_1500_FILEPATH
    ]
]

for dirpath in [TRAIN_SUPERVISED_REFS_DIRPATH, TRAIN_UNSUPERVISED_REFS_DIRPATH, TEST_REFS_DIRPATH]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

data_filepaths = zip(
    data_dirpaths,
    train_supervised_output_filepaths,
    train_unsupervised_output_filepaths,
    test_output_filepaths,
    unused_output_filepaths)

for data_dirpath, train_supervised_filepaths, train_unsupervised_filepaths, test_filepaths, unused_filepaths in data_filepaths:
    # Training set
    train_image_filepath = os.path.join(data_dirpath, VOID_TRAIN_IMAGE_FILENAME)
    train_sparse_depth_filepath = os.path.join(data_dirpath, VOID_TRAIN_SPARSE_DEPTH_FILENAME)
    train_validity_map_filepath = os.path.join(data_dirpath, VOID_TRAIN_VALIDITY_MAP_FILENAME)
    train_ground_truth_filepath = os.path.join(data_dirpath, VOID_TRAIN_GROUND_TRUTH_FILENAME)
    train_intrinsics_filepath = os.path.join(data_dirpath, VOID_TRAIN_INTRINSICS_FILENAME)

    # Read training paths
    train_image_paths = data_utils.read_paths(train_image_filepath)
    train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_filepath)
    train_validity_map_paths = data_utils.read_paths(train_validity_map_filepath)
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_filepath)
    train_intrinsics_paths = data_utils.read_paths(train_intrinsics_filepath)

    assert len(train_image_paths) == len(train_sparse_depth_paths)
    assert len(train_image_paths) == len(train_validity_map_paths)
    assert len(train_image_paths) == len(train_ground_truth_paths)
    assert len(train_image_paths) == len(train_intrinsics_paths)

    # Testing set
    test_image_filepath = os.path.join(data_dirpath, VOID_TEST_IMAGE_FILENAME)
    test_sparse_depth_filepath = os.path.join(data_dirpath, VOID_TEST_SPARSE_DEPTH_FILENAME)
    test_validity_map_filepath = os.path.join(data_dirpath, VOID_TEST_VALIDITY_MAP_FILENAME)
    test_ground_truth_filepath = os.path.join(data_dirpath, VOID_TEST_GROUND_TRUTH_FILENAME)
    test_intrinsics_filepath = os.path.join(data_dirpath, VOID_TEST_INTRINSICS_FILENAME)

    # Read testing paths
    test_image_paths = data_utils.read_paths(test_image_filepath)
    test_sparse_depth_paths = data_utils.read_paths(test_sparse_depth_filepath)
    test_validity_map_paths = data_utils.read_paths(test_validity_map_filepath)
    test_ground_truth_paths = data_utils.read_paths(test_ground_truth_filepath)
    test_intrinsics_paths = data_utils.read_paths(test_intrinsics_filepath)

    assert len(test_image_paths) == len(test_sparse_depth_paths)
    assert len(test_image_paths) == len(test_validity_map_paths)
    assert len(test_image_paths) == len(test_ground_truth_paths)
    assert len(test_image_paths) == len(test_intrinsics_paths)

    # Get test set directories
    test_seq_dirpaths = set(
        [test_image_paths[idx].split(os.sep)[-3] for idx in range(len(test_image_paths))])

    # Initialize placeholders for supervised training output paths
    train_supervised_image_outpaths = []
    train_supervised_sparse_depth_outpaths = []
    train_supervised_validity_map_outpaths = []
    train_supervised_ground_truth_outpaths = []
    train_supervised_intrinsics_outpaths = []

    # Initialize placeholders for unsupervised training output paths
    train_unsupervised_image_outpaths = []
    train_unsupervised_sparse_depth_outpaths = []
    train_unsupervised_validity_map_outpaths = []
    train_unsupervised_ground_truth_outpaths = []
    train_unsupervised_intrinsics_outpaths = []

    # Initialize placeholders for testing output paths
    test_image_outpaths = []
    test_sparse_depth_outpaths = []
    test_validity_map_outpaths = []
    test_ground_truth_outpaths = []
    test_intrinsics_outpaths = []

    # Initialize placeholders for unused testing output paths
    unused_image_outpaths = []
    unused_sparse_depth_outpaths = []
    unused_validity_map_outpaths = []
    unused_ground_truth_outpaths = []
    unused_intrinsics_outpaths = []

    # For each dataset density, grab the sequences
    seq_dirpaths = sorted(glob.glob(os.path.join(data_dirpath, 'data', '*')))
    n_sample = 0

    for seq_dirpath in seq_dirpaths:
        # For each sequence, grab the images, sparse depths and valid maps
        image_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'image', '*.png')))
        sparse_depth_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'sparse_depth', '*.png')))
        validity_map_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'validity_map', '*.png')))
        ground_truth_paths = \
            sorted(glob.glob(os.path.join(seq_dirpath, 'ground_truth', '*.png')))
        intrinsics_path = os.path.join(seq_dirpath, 'K.txt')

        assert len(image_paths) == len(sparse_depth_paths)
        assert len(image_paths) == len(validity_map_paths)

        # Load intrinsics
        kin = np.loadtxt(intrinsics_path)

        intrinsics_refpath = \
            os.path.join(*intrinsics_path.split(os.sep)[2:])
        intrinsics_outpath = \
            os.path.join(VOID_OUTPUT_DIRPATH, intrinsics_refpath[:-3]+'npy')
        image_out_dirpath = \
            os.path.join(os.path.dirname(intrinsics_outpath), 'image')

        if not os.path.exists(image_out_dirpath):
            os.makedirs(image_out_dirpath)

        # Save intrinsics
        if not args.paths_only:
            np.save(intrinsics_outpath, kin)

        # Collect all paths for supervised training
        train_supervised_image_outpaths += image_paths
        train_supervised_sparse_depth_outpaths += sparse_depth_paths
        train_supervised_validity_map_outpaths += validity_map_paths
        train_supervised_ground_truth_outpaths += ground_truth_paths
        train_supervised_intrinsics_outpaths += [intrinsics_outpath] * len(image_paths)

        if seq_dirpath.split(os.sep)[-1] in test_seq_dirpaths:
            start_idx = 0
            offset_idx = 0
            create_triplet = False
        else:
            # Skip first stationary 30 frames (1 second) and skip every 10
            start_idx = 30
            offset_idx = 10
            create_triplet = True

        # Process paths for unsupervised training
        pool_input = []
        for idx in range(start_idx, len(image_paths)-offset_idx-start_idx):
            pool_input.append((
                image_paths[idx-offset_idx],
                image_paths[idx],
                image_paths[idx+offset_idx],
                sparse_depth_paths[idx],
                validity_map_paths[idx],
                ground_truth_paths[idx],
                create_triplet,
                args.paths_only))

        with mp.Pool() as pool:
            pool_results = pool.map(process_frame, pool_input)

            for result in pool_results:
                image_refpath, \
                    image_outpath, \
                    sparse_depth_outpath, \
                    validity_map_outpath, \
                    ground_truth_outpath = result

                # Split into training, testing and unused testing sets
                if image_refpath in train_image_paths:
                    train_unsupervised_image_outpaths.append(image_outpath)
                    train_unsupervised_sparse_depth_outpaths.append(sparse_depth_outpath)
                    train_unsupervised_validity_map_outpaths.append(validity_map_outpath)
                    train_unsupervised_ground_truth_outpaths.append(ground_truth_outpath)
                    train_unsupervised_intrinsics_outpaths.append(intrinsics_outpath)
                elif image_refpath in test_image_paths:
                    test_image_outpaths.append(image_outpath)
                    test_sparse_depth_outpaths.append(sparse_depth_outpath)
                    test_validity_map_outpaths.append(validity_map_outpath)
                    test_ground_truth_outpaths.append(ground_truth_outpath)
                    test_intrinsics_outpaths.append(intrinsics_outpath)
                else:
                    unused_image_outpaths.append(image_outpath)
                    unused_sparse_depth_outpaths.append(sparse_depth_outpath)
                    unused_validity_map_outpaths.append(validity_map_outpath)
                    unused_ground_truth_outpaths.append(ground_truth_outpath)
                    unused_intrinsics_outpaths.append(intrinsics_outpath)

        n_sample = n_sample + len(pool_input)

        print('Completed processing {} examples for sequence={}'.format(
            len(pool_input), seq_dirpath))

    print('Completed processing {} examples for density={}'.format(n_sample, data_dirpath))

    '''
    Write supervised training paths to file
    '''
    void_train_supervised_image_filepath, \
        void_train_supervised_sparse_depth_filepath, \
        void_train_supervised_validity_map_filepath, \
        void_train_supervised_ground_truth_filepath, \
        void_train_supervised_intrinsics_filepath = train_supervised_filepaths

    print('Storing {} supervised training image file paths into: {}'.format(
        len(train_supervised_image_outpaths), void_train_supervised_image_filepath))
    data_utils.write_paths(
        void_train_supervised_image_filepath, train_supervised_image_outpaths)

    print('Storing {} supervised training sparse depth file paths into: {}'.format(
        len(train_supervised_sparse_depth_outpaths), void_train_supervised_sparse_depth_filepath))
    data_utils.write_paths(
        void_train_supervised_sparse_depth_filepath, train_supervised_sparse_depth_outpaths)

    print('Storing {} supervised training validity map file paths into: {}'.format(
        len(train_supervised_validity_map_outpaths), void_train_supervised_validity_map_filepath))
    data_utils.write_paths(
        void_train_supervised_validity_map_filepath, train_supervised_validity_map_outpaths)

    print('Storing {} supervised training groundtruth depth file paths into: {}'.format(
        len(train_supervised_ground_truth_outpaths), void_train_supervised_ground_truth_filepath))
    data_utils.write_paths(
        void_train_supervised_ground_truth_filepath, train_supervised_ground_truth_outpaths)

    print('Storing {} supervised training camera intrinsics file paths into: {}'.format(
        len(train_supervised_intrinsics_outpaths), void_train_supervised_intrinsics_filepath))
    data_utils.write_paths(
        void_train_supervised_intrinsics_filepath, train_supervised_intrinsics_outpaths)

    '''
    Write unsupervised training paths to file
    '''
    void_train_unsupervised_image_filepath, \
        void_train_unsupervised_sparse_depth_filepath, \
        void_train_unsupervised_validity_map_filepath, \
        void_train_unsupervised_ground_truth_filepath, \
        void_train_unsupervised_intrinsics_filepath = train_unsupervised_filepaths

    print('Storing {} unsupervised training image file paths into: {}'.format(
        len(train_unsupervised_image_outpaths), void_train_unsupervised_image_filepath))
    data_utils.write_paths(
        void_train_unsupervised_image_filepath, train_unsupervised_image_outpaths)

    print('Storing {} unsupervised training sparse depth file paths into: {}'.format(
        len(train_unsupervised_sparse_depth_outpaths), void_train_unsupervised_sparse_depth_filepath))
    data_utils.write_paths(
        void_train_unsupervised_sparse_depth_filepath, train_unsupervised_sparse_depth_outpaths)

    print('Storing {} unsupervised training validity map file paths into: {}'.format(
        len(train_unsupervised_validity_map_outpaths), void_train_unsupervised_validity_map_filepath))
    data_utils.write_paths(
        void_train_unsupervised_validity_map_filepath, train_unsupervised_validity_map_outpaths)

    print('Storing {} unsupervised training groundtruth depth file paths into: {}'.format(
        len(train_unsupervised_ground_truth_outpaths), void_train_unsupervised_ground_truth_filepath))
    data_utils.write_paths(
        void_train_unsupervised_ground_truth_filepath, train_unsupervised_ground_truth_outpaths)

    print('Storing {} unsupervised_training camera intrinsics file paths into: {}'.format(
        len(train_unsupervised_intrinsics_outpaths), void_train_unsupervised_intrinsics_filepath))
    data_utils.write_paths(
        void_train_unsupervised_intrinsics_filepath, train_unsupervised_intrinsics_outpaths)

    '''
    Write testing paths to file
    '''
    void_test_image_filepath, \
        void_test_sparse_depth_filepath, \
        void_test_validity_map_filepath, \
        void_test_ground_truth_filepath, \
        void_test_intrinsics_filepath = test_filepaths

    print('Storing {} testing image file paths into: {}'.format(
        len(test_image_outpaths), void_test_image_filepath))
    data_utils.write_paths(
        void_test_image_filepath, test_image_outpaths)

    print('Storing {} testing sparse depth file paths into: {}'.format(
        len(test_sparse_depth_outpaths), void_test_sparse_depth_filepath))
    data_utils.write_paths(
        void_test_sparse_depth_filepath, test_sparse_depth_outpaths)

    print('Storing {} testing validity map file paths into: {}'.format(
        len(test_validity_map_outpaths), void_test_validity_map_filepath))
    data_utils.write_paths(
        void_test_validity_map_filepath, test_validity_map_outpaths)

    print('Storing {} testing groundtruth depth file paths into: {}'.format(
        len(test_ground_truth_outpaths), void_test_ground_truth_filepath))
    data_utils.write_paths(
        void_test_ground_truth_filepath, test_ground_truth_outpaths)

    print('Storing {} testing camera intrinsics file paths into: {}'.format(
        len(test_intrinsics_outpaths), void_test_intrinsics_filepath))
    data_utils.write_paths(
        void_test_intrinsics_filepath, test_intrinsics_outpaths)

    '''
    Write unused paths to file
    '''
    void_unused_image_filepath, \
        void_unused_sparse_depth_filepath, \
        void_unused_validity_map_filepath, \
        void_unused_ground_truth_filepath, \
        void_unused_intrinsics_filepath = unused_filepaths

    print('Storing {} unused testing image file paths into: {}'.format(
        len(unused_image_outpaths), void_unused_image_filepath))
    data_utils.write_paths(
        void_unused_image_filepath, unused_image_outpaths)

    print('Storing {} unused testing sparse depth file paths into: {}'.format(
        len(unused_sparse_depth_outpaths), void_unused_sparse_depth_filepath))
    data_utils.write_paths(
        void_unused_sparse_depth_filepath, unused_sparse_depth_outpaths)

    print('Storing {} unused testing validity map file paths into: {}'.format(
        len(unused_validity_map_outpaths), void_unused_validity_map_filepath))
    data_utils.write_paths(
        void_unused_validity_map_filepath, unused_validity_map_outpaths)

    print('Storing {} unused testing groundtruth depth file paths into: {}'.format(
        len(unused_ground_truth_outpaths), void_unused_ground_truth_filepath))
    data_utils.write_paths(
        void_unused_ground_truth_filepath, unused_ground_truth_outpaths)

    print('Storing {} unused testing camera intrinsics file paths into: {}'.format(
        len(unused_intrinsics_outpaths), void_unused_intrinsics_filepath))
    data_utils.write_paths(
        void_unused_intrinsics_filepath, unused_intrinsics_outpaths)
