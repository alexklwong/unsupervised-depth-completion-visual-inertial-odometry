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
import sys, os, glob, argparse
sys.path.insert(0, 'src')
import numpy as np
import multiprocessing as mp
import cv2, data_utils


'''
  Paths for KITTI dataset
'''
KITTI_RAW_DATA_DIRPATH = os.path.join('data', 'kitti_raw_data')
KITTI_DEPTH_COMPLETION_DIRPATH = os.path.join('data', 'kitti_depth_completion')
KITTI_TRAINVAL_SPARSE_DEPTH_DIRPATH = os.path.join(
    KITTI_DEPTH_COMPLETION_DIRPATH, 'train_val_split', 'sparse_depth')
KITTI_TRAINVAL_SEMI_DENSE_DEPTH_DIRPATH = os.path.join(
    KITTI_DEPTH_COMPLETION_DIRPATH, 'train_val_split', 'ground_truth')
KITTI_VALIDATION_DIRPATH = os.path.join(KITTI_DEPTH_COMPLETION_DIRPATH, 'validation')
KITTI_TESTING_DIRPATH = os.path.join(KITTI_DEPTH_COMPLETION_DIRPATH, 'testing')
KITTI_CALIBRATION_FILENAME = 'calib_cam_to_cam.txt'

# To be concatenated to sequence path
KITTI_TRAINVAL_IMAGE_REFPATH = os.path.join('proj_depth', 'velodyne_raw')
KITTI_TRAINVAL_SPARSE_DEPTH_REFPATH = os.path.join('proj_depth', 'velodyne_raw')
KITTI_TRAINVAL_SEMI_DENSE_DEPTH_REFPATH = os.path.join('proj_depth', 'groundtruth')


'''
  Output paths
'''
KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH = os.path.join(
    'data', 'kitti_depth_completion_voiced')

TRAIN_OUTPUT_REF_DIRPATH = 'training'
VAL_OUTPUT_REF_DIRPATH = 'validation'
TEST_OUTPUT_REF_DIRPATH = 'testing'

TRAIN_IMAGE_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_image.txt')
TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_sparse_depth.txt')
TRAIN_INTERP_DEPTH_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_interp_depth.txt')
TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_validity_map.txt')
TRAIN_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_semi_dense_depth.txt')
TRAIN_INTRINSICS_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_train_intrinsics.txt')

UNUSED_IMAGE_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_image.txt')
UNUSED_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_sparse_depth.txt')
UNUSED_INTERP_DEPTH_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_interp_depth.txt')
UNUSED_VALIDITY_MAP_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_validity_map.txt')
UNUSED_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_semi_dense_depth.txt')
UNUSED_INTRINSICS_OUTPUT_FILEPATH = os.path.join(TRAIN_OUTPUT_REF_DIRPATH, 'kitti_unused_intrinsics_depth.txt')

VAL_IMAGE_OUTPUT_FILEPATH = os.path.join(VAL_OUTPUT_REF_DIRPATH, 'kitti_val_image.txt')
VAL_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(VAL_OUTPUT_REF_DIRPATH, 'kitti_val_sparse_depth.txt')
VAL_INTERP_DEPTH_OUTPUT_FILEPATH = os.path.join(VAL_OUTPUT_REF_DIRPATH, 'kitti_val_interp_depth.txt')
VAL_VALIDITY_MAP_OUTPUT_FILEPATH = os.path.join(VAL_OUTPUT_REF_DIRPATH, 'kitti_val_validity_map.txt')
VAL_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH = os.path.join(VAL_OUTPUT_REF_DIRPATH, 'kitti_val_semi_dense_depth.txt')
VAL_INTRINSICS_OUTPUT_FILEPATH = os.path.join(VAL_OUTPUT_REF_DIRPATH, 'kitti_val_intrinsics.txt')

TEST_IMAGE_OUTPUT_FILEPATH = os.path.join(TEST_OUTPUT_REF_DIRPATH, 'kitti_test_image.txt')
TEST_SPARSE_DEPTH_OUTPUT_FILEPATH = os.path.join(TEST_OUTPUT_REF_DIRPATH, 'kitti_test_sparse_depth.txt')
TEST_INTERP_DEPTH_OUTPUT_FILEPATH = os.path.join(TEST_OUTPUT_REF_DIRPATH, 'kitti_test_interp_depth.txt')
TEST_VALIDITY_MAP_OUTPUT_FILEPATH = os.path.join(TEST_OUTPUT_REF_DIRPATH, 'kitti_test_validity_map.txt')
TEST_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH = os.path.join(TEST_OUTPUT_REF_DIRPATH, 'kitti_test_semi_dense_depth.txt')
TEST_INTRINSICS_OUTPUT_FILEPATH = os.path.join(TEST_OUTPUT_REF_DIRPATH, 'kitti_test_intrinsics.txt')


def process_frame(params):
  image0_path, image1_path, image2_path, \
      sparse_depth_path, semi_dense_depth_path = params

  # Read images and concatenate together
  image0 = cv2.imread(image0_path)
  image1 = cv2.imread(image1_path)
  image2 = cv2.imread(image2_path)
  image = np.concatenate([image1, image0, image2], axis=1)

  sz, vm = data_utils.load_depth_with_validity_map(sparse_depth_path)
  iz = data_utils.interpolate_depth(sz, vm)

  # Create validity map and image output path
  interp_depth_output_path = sparse_depth_path \
      .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH) \
      .replace('sparse_depth', 'interp_depth')
  validity_map_output_path = sparse_depth_path \
      .replace(KITTI_DEPTH_COMPLETION_DIRPATH, KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH) \
      .replace('sparse_depth', 'validity_map')
  image_output_path = validity_map_output_path \
      .replace(os.path.join(os.sep+'proj_depth', 'velodyne_raw'), '') \
      .replace('validity_map', 'image')

  # Create output directories
  for output_path in [image_output_path, interp_depth_output_path, validity_map_output_path]:
    output_dirpath = os.path.dirname(output_path)
    if not os.path.exists(output_dirpath):
      try:
        os.makedirs(output_dirpath)
      except FileExistsError:
        pass

  # Write to disk
  data_utils.save_depth(iz, interp_depth_output_path)
  data_utils.save_validity_map(vm, validity_map_output_path)
  cv2.imwrite(image_output_path, image)

  return (image_output_path, sparse_depth_path,
      interp_depth_output_path, validity_map_output_path, semi_dense_depth_path)


parser = argparse.ArgumentParser()

parser.add_argument('--n_thread',  type=int, default=8)
args = parser.parse_args()

for dirpath in [TRAIN_OUTPUT_REF_DIRPATH, VAL_OUTPUT_REF_DIRPATH, TEST_OUTPUT_REF_DIRPATH]:
  if not os.path.exists(dirpath):
    os.makedirs(dirpath)

# Build a mapping between the camera intrinsics to the directories
intrinsics_files = sorted(glob.glob(os.path.join(KITTI_RAW_DATA_DIRPATH, '*', KITTI_CALIBRATION_FILENAME)))
intrinsics_dkeys = {}
for intrinsics_file in intrinsics_files:
  # Example: data/kitti_depth_completion_voiced/data/2011_09_26/kin2.npy
  intrinsics2_path = intrinsics_file \
      .replace(KITTI_RAW_DATA_DIRPATH, os.path.join(KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH, 'data')) \
      .replace(KITTI_CALIBRATION_FILENAME, 'intrinsics2.npy')
  intrinsics3_path = intrinsics_file \
      .replace(KITTI_RAW_DATA_DIRPATH, os.path.join(KITTI_DEPTH_COMPLETION_OUTPUT_DIRPATH, 'data')) \
      .replace(KITTI_CALIBRATION_FILENAME, 'intrinsics3.npy')

  sequence_dirpath = os.path.split(intrinsics2_path)[0]
  if not os.path.exists(sequence_dirpath):
    os.makedirs(sequence_dirpath)

  calib = data_utils.load_calibration(intrinsics_file)
  intrinsics2 = np.reshape(calib['P_rect_02'], [3, 4])
  intrinsics2 = intrinsics2[:3, :3].astype(np.float32)
  intrinsics3 = np.reshape(calib['P_rect_03'], [3, 4])
  intrinsics3 = intrinsics3[:3, :3].astype(np.float32)

  # Store as numpy
  np.save(intrinsics2_path, intrinsics2)
  np.save(intrinsics3_path, intrinsics3)

  # Add as keys to instrinsics dictionary
  sequence_date = intrinsics_file.split(os.sep)[2]
  intrinsics_dkeys[(sequence_date, 'image_02')] = intrinsics2_path
  intrinsics_dkeys[(sequence_date, 'image_03')] = intrinsics3_path


'''
  Create validity maps and paths for sparse and semi dense depth for training
'''
train_image_output_paths = []
train_sparse_depth_output_paths = []
train_interp_depth_output_paths = []
train_validity_map_output_paths = []
train_semi_dense_depth_output_paths = []
train_intrinsics_output_paths = []
unused_image_output_paths = []
unused_sparse_depth_output_paths = []
unused_interp_depth_output_paths = []
unused_validity_map_output_paths = []
unused_semi_dense_depth_output_paths = []
unused_intrinsics_output_paths = []
# Iterate through train and val directories
for refdir in ['train', 'val']:
  sparse_depth_sequence_dirpath = glob.glob(
      os.path.join(KITTI_TRAINVAL_SPARSE_DEPTH_DIRPATH, refdir, '*/'))
  # Iterate through sequences
  for sequence_dirpath in sparse_depth_sequence_dirpath:
    # Iterate through cameras 02 and 03
    for camera_dirpath in ['image_02', 'image_03']:
      # Contruct sparse depth dirpaths
      sparse_depth_paths = sorted(glob.glob(
          os.path.join(sequence_dirpath, KITTI_TRAINVAL_SPARSE_DEPTH_REFPATH, camera_dirpath, '*.png')))

      # Construct semi dense depth diraths
      semi_dense_depth_sequence_dirpath = sequence_dirpath.replace('sparse_depth', 'ground_truth')
      semi_dense_depth_paths = sorted(glob.glob(
          os.path.join(
            semi_dense_depth_sequence_dirpath, KITTI_TRAINVAL_SEMI_DENSE_DEPTH_REFPATH, camera_dirpath, '*.png')))

      assert(len(sparse_depth_paths) == len(semi_dense_depth_paths))

      # Obtain sequence dirpath in raw data
      sequence = sparse_depth_paths[0].split(os.sep)[5]
      sequence_date = sequence[0:10]
      raw_sequence_dirpath = os.path.join(
          KITTI_RAW_DATA_DIRPATH, sequence_date, sequence, camera_dirpath, 'data')
      image_paths = sorted(glob.glob(os.path.join(raw_sequence_dirpath, '*.png')))

      intrinsics_output_path = intrinsics_dkeys[sequence_date, camera_dirpath]

      print('Processing {} samples using KITTI sequence={} camera={}'.format(
          len(sparse_depth_paths), sequence_dirpath.split(os.sep)[-2], camera_dirpath))

      pool_inputs = []
      # Load sparse depth and save validity map
      for idx in range(len(sparse_depth_paths)):
        sparse_depth_path = sparse_depth_paths[idx]
        semi_dense_depth_path = semi_dense_depth_paths[idx]
        filename0 = os.path.split(sparse_depth_paths[idx])[-1]

        assert(os.path.split(semi_dense_depth_path)[-1] == filename0)

        # Construct image filepaths
        image0_path = os.path.join(raw_sequence_dirpath, filename0)
        image0_path_idx = image_paths.index(image0_path)
        image1_path = image_paths[image0_path_idx-1]
        image2_path = image_paths[image0_path_idx+1]

        pool_inputs.append(
            (image0_path, image1_path, image2_path, sparse_depth_path, semi_dense_depth_path))

      with mp.Pool(args.n_thread) as pool:
        pool_results = pool.map(process_frame, pool_inputs)

        for result in pool_results:
          image_output_path, sparse_depth_path, interp_depth_output_path, \
              validity_map_output_path, semi_dense_depth_path = result

          if refdir == 'train':
            train_image_output_paths.append(image_output_path)
            train_sparse_depth_output_paths.append(sparse_depth_path)
            train_interp_depth_output_paths.append(interp_depth_output_path)
            train_validity_map_output_paths.append(validity_map_output_path)
            train_semi_dense_depth_output_paths.append(semi_dense_depth_path)
            train_intrinsics_output_paths.append(intrinsics_output_path)
          elif refdir == 'val':
            unused_image_output_paths.append(image_output_path)
            unused_sparse_depth_output_paths.append(sparse_depth_path)
            unused_interp_depth_output_paths.append(interp_depth_output_path)
            unused_validity_map_output_paths.append(validity_map_output_path)
            unused_semi_dense_depth_output_paths.append(semi_dense_depth_path)
            unused_intrinsics_output_paths.append(intrinsics_output_path)

      print('Completed processing {} samples using KITTI sequence={} camera={}'.format(
          len(sparse_depth_paths), sequence_dirpath.split(os.sep)[-2], camera_dirpath))

print('Storing training image file paths into: %s' % TRAIN_IMAGE_OUTPUT_FILEPATH)
with open(TRAIN_IMAGE_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(train_image_output_paths)):
    o.write(train_image_output_paths[idx]+'\n')

print('Storing training sparse depth file paths into: %s' % TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH)
with open(TRAIN_SPARSE_DEPTH_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(train_sparse_depth_output_paths)):
    o.write(train_sparse_depth_output_paths[idx]+'\n')

print('Storing training interpolated depth file paths into: %s' % TRAIN_INTERP_DEPTH_OUTPUT_FILEPATH)
with open(TRAIN_INTERP_DEPTH_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(train_interp_depth_output_paths)):
    o.write(train_interp_depth_output_paths[idx]+'\n')

print('Storing training validity map file paths into: %s' % TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH)
with open(TRAIN_VALIDITY_MAP_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(train_validity_map_output_paths)):
    o.write(train_validity_map_output_paths[idx]+'\n')

print('Storing training semi dense depth file paths into: %s' % TRAIN_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH)
with open(TRAIN_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(train_semi_dense_depth_output_paths)):
    o.write(train_semi_dense_depth_output_paths[idx]+'\n')

print('Storing training intrinsics file paths into: %s' % TRAIN_INTRINSICS_OUTPUT_FILEPATH)
with open(TRAIN_INTRINSICS_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(train_intrinsics_output_paths)):
    o.write(train_intrinsics_output_paths[idx]+'\n')

print('Storing unused image file paths into: %s' % UNUSED_IMAGE_OUTPUT_FILEPATH)
with open(UNUSED_IMAGE_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(unused_image_output_paths)):
    o.write(unused_image_output_paths[idx]+'\n')

print('Storing unused sparse depth file paths into: %s' % UNUSED_SPARSE_DEPTH_OUTPUT_FILEPATH)
with open(UNUSED_SPARSE_DEPTH_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(unused_sparse_depth_output_paths)):
    o.write(unused_sparse_depth_output_paths[idx]+'\n')

print('Storing unused interpolated depth file paths into: %s' % UNUSED_INTERP_DEPTH_OUTPUT_FILEPATH)
with open(UNUSED_INTERP_DEPTH_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(unused_interp_depth_output_paths)):
    o.write(unused_interp_depth_output_paths[idx]+'\n')

print('Storing unused validity map file paths into: %s' % UNUSED_VALIDITY_MAP_OUTPUT_FILEPATH)
with open(UNUSED_VALIDITY_MAP_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(unused_validity_map_output_paths)):
    o.write(unused_validity_map_output_paths[idx]+'\n')

print('Storing unused semi dense depth file paths into: %s' % UNUSED_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH)
with open(UNUSED_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(unused_semi_dense_depth_output_paths)):
    o.write(unused_semi_dense_depth_output_paths[idx]+'\n')

print('Storing unused intrinsics file paths into: %s' % UNUSED_INTRINSICS_OUTPUT_FILEPATH)
with open(UNUSED_INTRINSICS_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(unused_intrinsics_output_paths)):
    o.write(unused_intrinsics_output_paths[idx]+'\n')


'''
  Create validity maps and paths for sparse and semi dense depth for validation and testing
'''
val_image_output_paths = []
val_sparse_depth_output_paths = []
val_interp_depth_output_paths = []
val_validity_map_output_paths = []
val_semi_dense_depth_output_paths = []
test_image_output_paths = []
test_sparse_depth_output_paths = []
test_interp_depth_output_paths = []
test_validity_map_output_paths = []
test_semi_dense_depth_output_paths = []
modes = [
    ('validation', KITTI_VALIDATION_DIRPATH, val_image_output_paths,
      val_sparse_depth_output_paths, val_interp_depth_output_paths,
      val_validity_map_output_paths, val_semi_dense_depth_output_paths),
    ('testing', KITTI_TESTING_DIRPATH, test_image_output_paths,
      test_sparse_depth_output_paths, test_interp_depth_output_paths,
      test_validity_map_output_paths, test_semi_dense_depth_output_paths)]

for mode in modes:
  mode_type, kitti_dirpath, image_output_paths, \
      sparse_depth_output_paths, interp_depth_output_paths, \
      validity_map_output_paths, semi_dense_depth_output_paths = mode
  # Iterate through sparse depth and semi dense ground-truth directories
  for refdir in ['image', 'sparse_depth', 'ground_truth']:
    filepaths = sorted(glob.glob(
        os.path.join(kitti_dirpath, refdir, '*.png')))
    # Iterate filepaths
    for idx in range(len(filepaths)):
      path = filepaths[idx]

      if refdir == 'image':
        image = cv2.imread(path)
        image = np.concatenate([image, image, image], axis=1)
        image_output_path = path \
            .replace('kitti_depth_completion', 'kitti_depth_completion_voiced')
        image_output_paths.append(image_output_path)

        if not os.path.exists(os.path.dirname(image_output_path)):
          os.makedirs(os.path.dirname(image_output_path))
        # Write to disk
        cv2.imwrite(image_output_path, image)

      elif refdir == 'sparse_depth':
        # Load sparse depth and save validity map
        sz, vm = data_utils.load_depth_with_validity_map(path)
        iz = data_utils.interpolate_depth(sz, vm)
        # Create validity map output path
        interp_depth_output_path = path \
            .replace('kitti_depth_completion', 'kitti_depth_completion_voiced') \
            .replace('sparse_depth', 'interp_depth')
        validity_map_output_path = path \
            .replace('kitti_depth_completion', 'kitti_depth_completion_voiced') \
            .replace('sparse_depth', 'validity_map')
        sparse_depth_output_paths.append(path)
        interp_depth_output_paths.append(interp_depth_output_path)
        validity_map_output_paths.append(validity_map_output_path)

        for output_path in [interp_depth_output_path, validity_map_output_path]:
          output_dirpath = os.path.dirname(output_path)
          if not os.path.exists(output_dirpath):
            os.makedirs(output_dirpath)
        # Write to disk
        data_utils.save_depth(iz, interp_depth_output_path)
        data_utils.save_validity_map(vm, validity_map_output_path)

      elif refdir == 'ground_truth':
        semi_dense_depth_output_paths.append(path)

      sys.stdout.write(
          'Processed {}/{} {} {} samples \r'.format(
            idx+1, len(filepaths), mode_type, refdir))
      sys.stdout.flush()

    print('Completed generating {} {} {} samples'.format(
        len(filepaths), mode_type, refdir))

print('Storing validation image file paths into: %s' % VAL_IMAGE_OUTPUT_FILEPATH)
with open(VAL_IMAGE_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(val_image_output_paths)):
    o.write(val_image_output_paths[idx]+'\n')

print('Storing validation sparse depth file paths into: %s' % VAL_SPARSE_DEPTH_OUTPUT_FILEPATH)
with open(VAL_SPARSE_DEPTH_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(val_sparse_depth_output_paths)):
    o.write(val_sparse_depth_output_paths[idx]+'\n')

print('Storing validation interpolated depth file paths into: %s' % VAL_INTERP_DEPTH_OUTPUT_FILEPATH)
with open(VAL_INTERP_DEPTH_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(val_interp_depth_output_paths)):
    o.write(val_interp_depth_output_paths[idx]+'\n')

print('Storing validation validity map file paths into: %s' % VAL_VALIDITY_MAP_OUTPUT_FILEPATH)
with open(VAL_VALIDITY_MAP_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(val_validity_map_output_paths)):
    o.write(val_validity_map_output_paths[idx]+'\n')

print('Storing validation semi dense depth file paths into: %s' % VAL_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH)
with open(VAL_SEMI_DENSE_DEPTH_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(val_semi_dense_depth_output_paths)):
    o.write(val_semi_dense_depth_output_paths[idx]+'\n')

print('Storing testing image file paths into: %s' % TEST_IMAGE_OUTPUT_FILEPATH)
with open(TEST_IMAGE_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(test_image_output_paths)):
    o.write(test_image_output_paths[idx]+'\n')

print('Storing testing sparse depth file paths into: %s' % TEST_SPARSE_DEPTH_OUTPUT_FILEPATH)
with open(TEST_SPARSE_DEPTH_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(test_sparse_depth_output_paths)):
    o.write(test_sparse_depth_output_paths[idx]+'\n')

print('Storing testing interpolated depth file paths into: %s' % TEST_INTERP_DEPTH_OUTPUT_FILEPATH)
with open(TEST_INTERP_DEPTH_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(test_interp_depth_output_paths)):
    o.write(test_interp_depth_output_paths[idx]+'\n')

print('Storing testing validity map file paths into: %s' % TEST_VALIDITY_MAP_OUTPUT_FILEPATH)
with open(TEST_VALIDITY_MAP_OUTPUT_FILEPATH, "w") as o:
  for idx in range(len(test_validity_map_output_paths)):
    o.write(test_validity_map_output_paths[idx]+'\n')
