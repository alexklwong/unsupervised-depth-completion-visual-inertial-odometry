# Unsupervised Depth Completion from Visual Inertial Odometry
Project **VOICED**: **De**pth **C**ompletion from **I**nertial **O**dometry and **V**ision

Tensorflow implementation of *Unsupervised Depth Completion from Visual Inertial Odometry*

Published in RA-L January 2020 and ICRA 2020

[[arxiv]](https://arxiv.org/pdf/1905.08616.pdf) [[poster]]()

Model have been tested on Ubuntu 16.04 using Python 3.5, Tensorflow 1.14

If you use this work, please cite our paper:
```
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
```

## About Sparse-to-Dense Depth Completion
In the sparse-to-dense depth completion problem, we seek to infer the dense depth map of a 3-D scene using an RGB image and its associated sparse depth measurements in the form of a sparse depth map, obtained either from computational methods such as SfM (Strcuture-from-Motion) or active sensors such as lidar or structured light sensors.

| *Input RGB image from the VOID dataset*    | *Densified depth map -- colored and back-projected to 3-D* |
| :----------------------------------------- | :--------------------------------------------------------: |
| <img src="figures/void_teaser.jpg" width="400"> | <img src="figures/void_teaser.gif"> |

| *Input RGB image from the KITTI dataset*    | *Densified depth map -- colored and back-projected to 3-D* |
| :------------------------------------------ | :--------------------------------------------------------: |
| <img src="figures/kitti_teaser.jpg" width="400"> | <img src="figures/kitti_teaser.gif"> |

To follow the literature and benchmarks for this task, you may visit:
[Awesome State of Depth Completion](https://github.com/alexklwong/awesome-state-of-depth-completion)

## Setting up your virtual environment
We will create a virtual environment with the necessary dependencies
```
virtualenv -p /usr/bin/python3 voiced-py3env
source voiced-py3env/bin/activate
pip install opencv-python scipy scikit-learn Pillow matplotlib gdown
pip install numpy==1.16.4 gast==0.2.2
pip install tensorflow-gpu==1.14
```

## Setting up your data
For datasets, we will use KITTI for outdoors and VOID for indoors
```
mkdir data
sh bash/setup_dataset_kitti.sh
sh bash/setup_dataset_void.sh
```

## Training VOICED
To train VOICED on the KITTI dataset, you may run
```
sh bash/train_voiced_kitti.sh
```

To train VOICED on the VOID datasets, you may run
```
sh bash/train_voiced_void.sh
```

To monitor your training progress, you may use Tensorboard
```
tensorboard --logdir trained_models/<model_name>
```

## Downloading our Pretrained Models
```
gdown https://drive.google.com/uc?id=18jr9l1YvxDUzqAa_S-LYTdfi6zN1OEE9
unzip pretrained_models.zip
```

## Evaluating VOICED
To evaluate the pretrained VOICED on the KITTI dataset, you may run
```
sh bash/evaluate_voiced_kitti.sh
```

To evaluate the pretrained VOICED on the VOID dataset, you may run
```
sh bash/evaluate_voiced_void.sh
```

You may replace the restore_path and output_path arguments to evaluate your own checkpoints

## Related Projects
You may also find the following projects useful:

- [XIVO][xivo_github]: The Visual-Inertial Odometry system developed at UCLA Vision Lab and used by VOICED.
- [GeoSup][geosup_github]: Geo-Supervised Visual Depth Prediction developed by the authors and awarded *Best Paper in Robot Vision* at ICRA 2019.

[xivo_github]: https://github.com/ucla-vision/xivo
[geosup_github]: https://github.com/feixh/GeoSup
