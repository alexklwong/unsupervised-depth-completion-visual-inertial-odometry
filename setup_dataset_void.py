'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Xiaohan Fei <feixh@cs.ucla.edu>
If you use this code, please cite the following paper:
A. Wong, X. Fei, S. Tsuei, and S. Soatto. Unsupervised Depth Completion from Visual Inertial Odometry.
https://arxiv.org/pdf/1905.08616.pdf
@article{wong2020unsupervised,
  title={Unsupervised Depth Completion From Visual Inertial Odometry},
  author={Wong, Alex and Fei, Xiaohan and Tsuei, Stephanie and Soatto, Stefano},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={1899--1906},
  year={2020},
  publisher={IEEE}
}
'''
import os, sys, glob, argparse
import multiprocessing as mp
import numpy as np
import cv2
sys.path.insert(0, 'src')
import data_utils


VOID_ROOT_DIRPATH       = os.path.join('data', 'void_release')
VOID_DATA_150_DIRPATH   = os.path.join(VOID_ROOT_DIRPATH, 'void_150')
VOID_DATA_500_DIRPATH   = os.path.join(VOID_ROOT_DIRPATH, 'void_500')
VOID_DATA_1500_DIRPATH  = os.path.join(VOID_ROOT_DIRPATH, 'void_1500')

VOID_OUT_DIRPATH        = os.path.join('data', 'void_voiced')

VOID_TRAIN_IMAGE_FILENAME         = 'train_image.txt'
VOID_TRAIN_SPARSE_DEPTH_FILENAME  = 'train_sparse_depth.txt'
VOID_TRAIN_VALIDITY_MAP_FILENAME  = 'train_validity_map.txt'
VOID_TRAIN_GROUND_TRUTH_FILENAME  = 'train_ground_truth.txt'
VOID_TRAIN_INTRINSICS_FILENAME    = 'train_intrinsics.txt'
VOID_TEST_IMAGE_FILENAME          = 'test_image.txt'
VOID_TEST_SPARSE_DEPTH_FILENAME   = 'test_sparse_depth.txt'
VOID_TEST_VALIDITY_MAP_FILENAME   = 'test_validity_map.txt'
VOID_TEST_GROUND_TRUTH_FILENAME   = 'test_ground_truth.txt'
VOID_TEST_INTRINSICS_FILENAME     = 'test_intrinsics.txt'

TRAIN_REFS_DIRPATH      = 'training'
TEST_REFS_DIRPATH       = 'testing'

# VOID training set 150 density
VOID_TRAIN_IMAGE_150_FILEPATH           = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_image_150.txt')
VOID_TRAIN_SPARSE_DEPTH_150_FILEPATH    = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_sparse_depth_150.txt')
VOID_TRAIN_INTERP_DEPTH_150_FILEPATH    = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_interp_depth_150.txt')
VOID_TRAIN_VALIDITY_MAP_150_FILEPATH    = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_validity_map_150.txt')
VOID_TRAIN_GROUND_TRUTH_150_FILEPATH    = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_ground_truth_150.txt')
VOID_TRAIN_INTRINSICS_150_FILEPATH      = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_intrinsics_150.txt')
# VOID training set 500 density
VOID_TRAIN_IMAGE_500_FILEPATH           = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_image_500.txt')
VOID_TRAIN_SPARSE_DEPTH_500_FILEPATH    = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_sparse_depth_500.txt')
VOID_TRAIN_INTERP_DEPTH_500_FILEPATH    = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_interp_depth_500.txt')
VOID_TRAIN_VALIDITY_MAP_500_FILEPATH    = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_validity_map_500.txt')
VOID_TRAIN_GROUND_TRUTH_500_FILEPATH    = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_ground_truth_500.txt')
VOID_TRAIN_INTRINSICS_500_FILEPATH      = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_intrinsics_500.txt')
# VOID training set 1500 density
VOID_TRAIN_IMAGE_1500_FILEPATH          = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_image_1500.txt')
VOID_TRAIN_SPARSE_DEPTH_1500_FILEPATH   = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_sparse_depth_1500.txt')
VOID_TRAIN_INTERP_DEPTH_1500_FILEPATH   = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_interp_depth_1500.txt')
VOID_TRAIN_VALIDITY_MAP_1500_FILEPATH   = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_validity_map_1500.txt')
VOID_TRAIN_GROUND_TRUTH_1500_FILEPATH   = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_ground_truth_1500.txt')
VOID_TRAIN_INTRINSICS_1500_FILEPATH     = os.path.join(TRAIN_REFS_DIRPATH, 'void_train_intrinsics_1500.txt')
# VOID testing set 150 density
VOID_TEST_IMAGE_150_FILEPATH            = os.path.join(TEST_REFS_DIRPATH, 'void_test_image_150.txt')
VOID_TEST_SPARSE_DEPTH_150_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'void_test_sparse_depth_150.txt')
VOID_TEST_INTERP_DEPTH_150_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'void_test_interp_depth_150.txt')
VOID_TEST_VALIDITY_MAP_150_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'void_test_validity_map_150.txt')
VOID_TEST_GROUND_TRUTH_150_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'void_test_ground_truth_150.txt')
VOID_TEST_INTRINSICS_150_FILEPATH       = os.path.join(TEST_REFS_DIRPATH, 'void_test_intrinsics_150.txt')
# VOID testing set 500 density
VOID_TEST_IMAGE_500_FILEPATH            = os.path.join(TEST_REFS_DIRPATH, 'void_test_image_500.txt')
VOID_TEST_SPARSE_DEPTH_500_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'void_test_sparse_depth_500.txt')
VOID_TEST_INTERP_DEPTH_500_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'void_test_interp_depth_500.txt')
VOID_TEST_VALIDITY_MAP_500_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'void_test_validity_map_500.txt')
VOID_TEST_GROUND_TRUTH_500_FILEPATH     = os.path.join(TEST_REFS_DIRPATH, 'void_test_ground_truth_500.txt')
VOID_TEST_INTRINSICS_500_FILEPATH       = os.path.join(TEST_REFS_DIRPATH, 'void_test_intrinsics_500.txt')
# VOID testing set 1500 density
VOID_TEST_IMAGE_1500_FILEPATH           = os.path.join(TEST_REFS_DIRPATH, 'void_test_image_1500.txt')
VOID_TEST_SPARSE_DEPTH_1500_FILEPATH    = os.path.join(TEST_REFS_DIRPATH, 'void_test_sparse_depth_1500.txt')
VOID_TEST_INTERP_DEPTH_1500_FILEPATH    = os.path.join(TEST_REFS_DIRPATH, 'void_test_interp_depth_1500.txt')
VOID_TEST_VALIDITY_MAP_1500_FILEPATH    = os.path.join(TEST_REFS_DIRPATH, 'void_test_validity_map_1500.txt')
VOID_TEST_GROUND_TRUTH_1500_FILEPATH    = os.path.join(TEST_REFS_DIRPATH, 'void_test_ground_truth_1500.txt')
VOID_TEST_INTRINSICS_1500_FILEPATH      = os.path.join(TEST_REFS_DIRPATH, 'void_test_intrinsics_1500.txt')


def process_frame(args):
  image_path1, image_path0, image_path2, \
      sparse_depth_path, validity_map_path, ground_truth_path = args

  # Create image composite of triplets
  im1 = cv2.imread(image_path1)
  im0 = cv2.imread(image_path0)
  im2 = cv2.imread(image_path2)
  imc = np.concatenate([im1, im0, im2], axis=1)

  # Create interpolated depth
  sz, vm = data_utils.load_depth_with_validity_map(sparse_depth_path)
  iz = data_utils.interpolate_depth(sz, vm)

  image_ref_path = os.path.join(*image_path0.split(os.sep)[2:])
  sparse_depth_ref_path = os.path.join(*sparse_depth_path.split(os.sep)[2:])
  # Set output paths
  image_output_path = os.path.join(VOID_OUT_DIRPATH, image_ref_path)
  sparse_depth_output_path = sparse_depth_path
  interp_depth_output_path = os.path.join(VOID_OUT_DIRPATH, sparse_depth_ref_path) \
      .replace('sparse_depth', 'interp_depth')
  validity_map_output_path = validity_map_path
  ground_truth_output_path = ground_truth_path

  # Verify that all filenames match
  image_out_dirpath, image_filename = os.path.split(image_output_path)
  sparse_depth_filename = os.path.basename(sparse_depth_output_path)
  validity_map_filename = os.path.basename(validity_map_output_path)
  ground_truth_filename = os.path.basename(ground_truth_output_path)
  assert(image_filename == sparse_depth_filename)
  assert(image_filename == validity_map_filename)
  assert(image_filename == ground_truth_filename)

  # Write to disk
  cv2.imwrite(image_output_path, imc)
  data_utils.save_depth(iz, interp_depth_output_path)

  return (image_ref_path, image_output_path, sparse_depth_output_path,
      interp_depth_output_path, validity_map_output_path, ground_truth_output_path)


parser = argparse.ArgumentParser()

parser.add_argument('--n_thread',  type=int, default=8)
args = parser.parse_args()

for ref_dirpath in [TRAIN_REFS_DIRPATH, TEST_REFS_DIRPATH]:
  if not os.path.exists(ref_dirpath):
    os.makedirs(ref_dirpath)

data_dirpaths = [VOID_DATA_150_DIRPATH, VOID_DATA_500_DIRPATH, VOID_DATA_1500_DIRPATH]

train_output_filepaths = [
    [VOID_TRAIN_IMAGE_150_FILEPATH, VOID_TRAIN_SPARSE_DEPTH_150_FILEPATH,
      VOID_TRAIN_INTERP_DEPTH_150_FILEPATH, VOID_TRAIN_VALIDITY_MAP_150_FILEPATH,
      VOID_TRAIN_GROUND_TRUTH_150_FILEPATH, VOID_TRAIN_INTRINSICS_150_FILEPATH],
    [VOID_TRAIN_IMAGE_500_FILEPATH, VOID_TRAIN_SPARSE_DEPTH_500_FILEPATH,
      VOID_TRAIN_INTERP_DEPTH_500_FILEPATH, VOID_TRAIN_VALIDITY_MAP_500_FILEPATH,
      VOID_TRAIN_GROUND_TRUTH_500_FILEPATH, VOID_TRAIN_INTRINSICS_500_FILEPATH],
    [VOID_TRAIN_IMAGE_1500_FILEPATH, VOID_TRAIN_SPARSE_DEPTH_1500_FILEPATH,
      VOID_TRAIN_INTERP_DEPTH_1500_FILEPATH, VOID_TRAIN_VALIDITY_MAP_1500_FILEPATH,
      VOID_TRAIN_GROUND_TRUTH_1500_FILEPATH, VOID_TRAIN_INTRINSICS_1500_FILEPATH]]
test_output_filepaths = [
    [VOID_TEST_IMAGE_150_FILEPATH, VOID_TEST_SPARSE_DEPTH_150_FILEPATH,
      VOID_TEST_INTERP_DEPTH_150_FILEPATH, VOID_TEST_VALIDITY_MAP_150_FILEPATH,
      VOID_TEST_GROUND_TRUTH_150_FILEPATH, VOID_TEST_INTRINSICS_150_FILEPATH],
    [VOID_TEST_IMAGE_500_FILEPATH, VOID_TEST_SPARSE_DEPTH_500_FILEPATH,
      VOID_TEST_INTERP_DEPTH_500_FILEPATH, VOID_TEST_VALIDITY_MAP_500_FILEPATH,
      VOID_TEST_GROUND_TRUTH_500_FILEPATH, VOID_TEST_INTRINSICS_500_FILEPATH],
    [VOID_TEST_IMAGE_1500_FILEPATH, VOID_TEST_SPARSE_DEPTH_1500_FILEPATH,
      VOID_TEST_INTERP_DEPTH_1500_FILEPATH, VOID_TEST_VALIDITY_MAP_1500_FILEPATH,
      VOID_TEST_GROUND_TRUTH_1500_FILEPATH, VOID_TEST_INTRINSICS_1500_FILEPATH]]

data_filepaths = zip(data_dirpaths, train_output_filepaths, test_output_filepaths)
for data_dirpath, train_filepaths, test_filepaths in data_filepaths:
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
  assert(len(train_image_paths) == len(train_sparse_depth_paths))
  assert(len(train_image_paths) == len(train_validity_map_paths))
  assert(len(train_image_paths) == len(train_ground_truth_paths))
  assert(len(train_image_paths) == len(train_intrinsics_paths))
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
  assert(len(test_image_paths) == len(test_sparse_depth_paths))
  assert(len(test_image_paths) == len(test_validity_map_paths))
  assert(len(test_image_paths) == len(test_ground_truth_paths))
  assert(len(test_image_paths) == len(test_intrinsics_paths))

  # Get test set directories
  test_seq_dirpaths = set(
      [test_image_paths[idx].split(os.sep)[-3] for idx in range(len(test_image_paths))])

  # Initialize placeholders for training output paths
  train_image_output_paths = []
  train_sparse_depth_output_paths = []
  train_interp_depth_output_paths = []
  train_validity_map_output_paths = []
  train_ground_truth_output_paths = []
  train_intrinsics_output_paths = []
  # Initialize placeholders for testing output paths
  test_image_output_paths = []
  test_sparse_depth_output_paths = []
  test_interp_depth_output_paths = []
  test_validity_map_output_paths = []
  test_ground_truth_output_paths = []
  test_intrinsics_output_paths = []

  # For each dataset density, grab the sequences
  seq_dirpaths = glob.glob(os.path.join(data_dirpath, 'data', '*'))
  n_sample = 0
  for seq_dirpath in seq_dirpaths:
    # For each sequence, grab the images, sparse depths and valid maps
    image_paths = sorted(glob.glob(os.path.join(seq_dirpath, 'image', '*.png')))
    sparse_depth_paths = sorted(glob.glob(os.path.join(seq_dirpath, 'sparse_depth', '*.png')))
    validity_map_paths = sorted(glob.glob(os.path.join(seq_dirpath, 'validity_map', '*.png')))
    ground_truth_paths = sorted(glob.glob(os.path.join(seq_dirpath, 'ground_truth', '*.png')))
    absolute_pose_paths = sorted(glob.glob(os.path.join(seq_dirpath, 'absolute_pose', '*.txt')))
    intrinsics_path = os.path.join(seq_dirpath, 'K.txt')
    assert(len(image_paths) == len(sparse_depth_paths))
    assert(len(image_paths) == len(validity_map_paths))
    assert(len(image_paths) == len(ground_truth_paths))
    assert(len(image_paths) == len(absolute_pose_paths))

    # Load intrinsics and process first
    kin = np.loadtxt(intrinsics_path)
    intrinsics_ref_path = os.path.join(*intrinsics_path.split(os.sep)[2:])
    intrinsics_output_path = os.path.join(VOID_OUT_DIRPATH, intrinsics_ref_path[:-3]+'npy')
    image_output_dirpath = os.path.join(os.path.dirname(intrinsics_output_path), 'image')
    interp_depth_output_dirpath = os.path.join(os.path.dirname(intrinsics_output_path), 'interp_depth')

    for output_dirpath in [image_output_dirpath, interp_depth_output_dirpath]:
      if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # Save intrinsics
    np.save(intrinsics_output_path, kin)

    if seq_dirpath.split(os.sep)[-1] in test_seq_dirpaths:
      start_idx = 0
      offset_idx = 0
    else:
      start_idx = 30  # Skip the first 30 stationary frames
      offset_idx = 5  # Temporal window size

    pool_inputs = []
    for idx in range(start_idx, len(image_paths)-offset_idx-start_idx):
      # Find images with enough parallax, pose are from camera to world
      pose0tow = np.loadtxt(absolute_pose_paths[idx])
      pose1tow = np.loadtxt(absolute_pose_paths[idx-offset_idx])
      pose2tow = np.loadtxt(absolute_pose_paths[idx+offset_idx])
      posewto0 = data_utils.inverse_pose(pose0tow)
      pose1to0 = data_utils.compose_pose(posewto0, pose1tow)
      pose2to0 = data_utils.compose_pose(posewto0, pose1tow)
      parallax1to0 = np.linalg.norm(pose1to0[:3, 3])
      parallax2to0 = np.linalg.norm(pose2to0[:3, 3])

      # If translation magnitude less than 1 centimeter, then skip
      if offset_idx > 0 and (parallax1to0 < 0.01 or parallax2to0 < 0.01):
        continue

      pool_inputs.append(
          (image_paths[idx-offset_idx],
           image_paths[idx],
           image_paths[idx+offset_idx],
           sparse_depth_paths[idx],
           validity_map_paths[idx],
           ground_truth_paths[idx]))

    sys.stdout.write('Processing {} examples for sequence={}\r'.format(
        len(pool_inputs), seq_dirpath))
    sys.stdout.flush()

    with mp.Pool(args.n_thread) as pool:
      pool_results = pool.map(process_frame, pool_inputs)

      for result in pool_results:
        image_ref_path, image_output_path, \
            sparse_depth_output_path, interp_depth_output_path, \
            validity_map_output_path, ground_truth_output_path = result

        # Split into training, testing and unused testing sets
        if image_ref_path in train_image_paths:
          train_image_output_paths.append(image_output_path)
          train_sparse_depth_output_paths.append(sparse_depth_output_path)
          train_interp_depth_output_paths.append(interp_depth_output_path)
          train_validity_map_output_paths.append(validity_map_output_path)
          train_ground_truth_output_paths.append(ground_truth_output_path)
          train_intrinsics_output_paths.append(intrinsics_output_path)
        elif image_ref_path in test_image_paths:
          test_image_output_paths.append(image_output_path)
          test_sparse_depth_output_paths.append(sparse_depth_output_path)
          test_interp_depth_output_paths.append(interp_depth_output_path)
          test_validity_map_output_paths.append(validity_map_output_path)
          test_ground_truth_output_paths.append(ground_truth_output_path)
          test_intrinsics_output_paths.append(intrinsics_output_path)

    if offset_idx > 0:
      n_sample = n_sample+len(pool_inputs)

    print('Completed processing {} examples for sequence={}'.format(
        len(pool_inputs), seq_dirpath))

  print('Completed processing {} examples for density={}'.format(n_sample, data_dirpath))

  void_train_image_filepath, void_train_sparse_depth_filepath, \
      void_train_interp_depth_filepath, void_train_validity_map_filepath, \
      void_train_ground_truth_filepath, void_train_intrinsics_filepath = train_filepaths

  print('Storing training image file paths into: %s' % void_train_image_filepath)
  with open(void_train_image_filepath, "w") as o:
    for idx in range(len(train_image_output_paths)):
      o.write(train_image_output_paths[idx]+'\n')
  print('Storing training sparse depth file paths into: %s' % void_train_sparse_depth_filepath)
  with open(void_train_sparse_depth_filepath, "w") as o:
    for idx in range(len(train_sparse_depth_output_paths)):
      o.write(train_sparse_depth_output_paths[idx]+'\n')
  print('Storing training interpolated depth file paths into: %s' % void_train_interp_depth_filepath)
  with open(void_train_interp_depth_filepath, "w") as o:
    for idx in range(len(train_interp_depth_output_paths)):
      o.write(train_interp_depth_output_paths[idx]+'\n')
  print('Storing training validity map file paths into: %s' % void_train_validity_map_filepath)
  with open(void_train_validity_map_filepath, "w") as o:
    for idx in range(len(train_validity_map_output_paths)):
      o.write(train_validity_map_output_paths[idx]+'\n')
  print('Storing training groundtruth depth file paths into: %s' % void_train_ground_truth_filepath)
  with open(void_train_ground_truth_filepath, "w") as o:
    for idx in range(len(train_ground_truth_output_paths)):
      o.write(train_ground_truth_output_paths[idx]+'\n')
  print('Storing training camera intrinsics file paths into: %s' % void_train_intrinsics_filepath)
  with open(void_train_intrinsics_filepath, "w") as o:
    for idx in range(len(train_intrinsics_output_paths)):
      o.write(train_intrinsics_output_paths[idx]+'\n')

  void_test_image_filepath, void_test_sparse_depth_filepath, \
      void_test_interp_depth_filepath, void_test_validity_map_filepath, \
      void_test_ground_truth_filepath, void_test_intrinsics_filepath = test_filepaths

  print('Storing testing image file paths into: %s' % void_test_image_filepath)
  with open(void_test_image_filepath, "w") as o:
    for idx in range(len(test_image_output_paths)):
      o.write(test_image_output_paths[idx]+'\n')
  print('Storing testing sparse depth file paths into: %s' % void_test_sparse_depth_filepath)
  with open(void_test_sparse_depth_filepath, "w") as o:
    for idx in range(len(test_sparse_depth_output_paths)):
      o.write(test_sparse_depth_output_paths[idx]+'\n')
  print('Storing testing interpolated depth file paths into: %s' % void_test_interp_depth_filepath)
  with open(void_test_interp_depth_filepath, "w") as o:
    for idx in range(len(test_interp_depth_output_paths)):
      o.write(test_interp_depth_output_paths[idx]+'\n')
  print('Storing testing validity map file paths into: %s' % void_test_validity_map_filepath)
  with open(void_test_validity_map_filepath, "w") as o:
    for idx in range(len(test_validity_map_output_paths)):
      o.write(test_validity_map_output_paths[idx]+'\n')
  print('Storing testing groundtruth depth file paths into: %s' % void_test_ground_truth_filepath)
  with open(void_test_ground_truth_filepath, "w") as o:
    for idx in range(len(test_ground_truth_output_paths)):
      o.write(test_ground_truth_output_paths[idx]+'\n')
  print('Storing testing camera intrinsics file paths into: %s' % void_test_intrinsics_filepath)
  with open(void_test_intrinsics_filepath, "w") as o:
    for idx in range(len(test_intrinsics_output_paths)):
      o.write(test_intrinsics_output_paths[idx]+'\n')
