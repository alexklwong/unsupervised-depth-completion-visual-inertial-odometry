# Unsupervised Depth Completion from Visual Inertial Odometry
Project **VOICED**: **De**pth **C**ompletion from **I**nertial **O**dometry and **V**ision

Tensorflow implementation of *Unsupervised Depth Completion from Visual Inertial Odometry*

Published in RA-L January 2020 and ICRA 2020

[[arxiv]](https://arxiv.org/pdf/1905.08616.pdf) [[poster]]() [[talk]](https://www.youtube.com/watch?v=oBCKO4TH5y0)

Model have been tested on Ubuntu 16.04 using Python 3.5, 3.6 Tensorflow 1.14, 1.15

Authors: [Alex Wong](http://web.cs.ucla.edu/~alexw/), [Xiaohan Fei](https://feixh.github.io/), Stephanie Tsuei

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

### Looking our latest work in unsupervised depth completion?

Check out our RAL 2021 and ICRA 2021 paper, [ScaffNet][scaffnet_github]: *Learning Topology from Synthetic Data for Unsupervised Depth Completion*

[ScaffNet][scaffnet_github] is trained on synthetic data (SceneNet), but is able to generalize to novel real data (VOID and NYUv2)!

Also, our ICCV 2021 oral paper, [KBNet][kbnet_github]: *Unsupervised Depth Completion with Calibrated Backprojection Layers*

[KBNet][kbnet_github] runs at 15 ms/frame (67 fps) and improves over VOICED by 51.7% on indoors (VOID) and 13.7% on outdoors (KITTI)!

### Coming soon!
Thanks for all the interest and inquiries! 

We are planning on releasing PyTorch versions of VOICED and ScaffNet over the coming months!

### Table of Contents
1. [About sparse-to-dense depth completion](#about-sparse-to-dense)
2. [About VOICED](#about-voiced)
3. [Setting up](#setting-up)
4. [Training VOICED](#training-voiced)
5. [Downloading pretrained models](#downloading-pretrained-models)
6. [Evaluating VOICED](#evaluating-voiced)
7. [Related projects](#related-projects)
8. [License and disclaimer](#license-disclaimer)

## About sparse-to-dense depth completion <a name="about-sparse-to-dense"></a>
In the sparse-to-dense depth completion problem, we seek to infer the dense depth map of a 3-D scene using an RGB image and its associated sparse depth measurements in the form of a sparse depth map, obtained either from computational methods such as SfM (Strcuture-from-Motion) or active sensors such as lidar or structured light sensors.

| *Input RGB image from the VOID dataset*    | *Densified depth map -- colored and back-projected to 3-D* |
| :----------------------------------------: | :--------------------------------------------------------: |
| <img src="figures/void_teaser.jpg" width="400"> | <img src="figures/void_teaser.gif"> |

| *Input RGB image from the KITTI dataset*    | *Densified depth map -- colored and back-projected to 3-D* |
| :-----------------------------------------: | :--------------------------------------------------------: |
| <img src="figures/kitti_teaser.jpg" width="400"> | <img src="figures/kitti_teaser.gif"> |

To follow the literature and benchmarks for this task, you may visit:
[Awesome State of Depth Completion](https://github.com/alexklwong/awesome-state-of-depth-completion)

## About VOICED <a name="about-voiced"></a>
VOICED is an unsupervised depth completion method that is built on top of [XIVO][xivo_github]. Unlike previous methods, we build a scaffolding of the scene using the sparse depth measurements (~5% density for outdoors driving scenarios like KITTI and ~0.5% to ~0.05% for indoors scenes like VOID) and refines the scaffolding using a light-weight network.

<p align="center">
<img align="center" src="figures/digest_teaser_horizontal.png" width="800">
</p>

This paradigm allows us to achieve the state-of-the-art on the unsupervised depth completion task while reducing parameters by as much as 80% compared to prior-arts. As an added bonus, our approach does not require top of the line GPUs (e.g. Tesla V100, Titan V) and can be deployed on much cheaper hardware.

## Setting up your virtual environment <a name="setting-up"></a>
We will create a virtual environment with the necessary dependencies
```
virtualenv -p /usr/bin/python3 voiced-py3env
source voiced-py3env/bin/activate
pip install opencv-python scipy scikit-learn Pillow matplotlib gdown
pip install numpy==1.16.4 gast==0.2.2
pip install tensorflow-gpu==1.14
```

## Setting up your datasets
For datasets, we will use KITTI for outdoors and VOID for indoors
```
mkdir data
bash bash/setup_dataset_kitti.sh
bash bash/setup_dataset_void.sh
```

The bash script downloads the VOID dataset using gdown. However, gdown intermittently fails. As a workaround, you may download them via:
```
https://drive.google.com/open?id=1kZ6ALxCzhQP8Tq1enMyNhjclVNzG8ODA
https://drive.google.com/open?id=1ys5EwYK6i8yvLcln6Av6GwxOhMGb068m
https://drive.google.com/open?id=1bTM5eh9wQ4U8p2ANOGbhZqTvDOddFnlI
```
which will give you three files `void_150.zip`, `void_500.zip`, `void_1500.zip`.

Assuming you are in the root of the repository, to construct the same dataset structure as the setup script above:
```
mkdir void_release
unzip -o void_150.zip -d void_release/
unzip -o void_500.zip -d void_release/
unzip -o void_1500.zip -d void_release/
bash bash/setup_dataset_void.sh unpack-only
```
If you encounter `error: invalid zip file with overlapped components (possible zip bomb)`. Please do the following
```
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
```
and run the above again.

For more detailed instructions on downloading and using VOID and obtaining the raw rosbags, you may visit the [VOID][void_github] dataset webpage.

## Training VOICED <a name="training-voiced"></a>
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

## Downloading our pretrained models <a name="downloading-pretrained-models"></a>
To use our KITTI and VOID models, you can download
```
gdown https://drive.google.com/uc?id=18jr9l1YvxDUzqAa_S-LYTdfi6zN1OEE9
unzip pretrained_models.zip
```

Note: `gdown` fails intermittently and complains about permission. If that happens, you may also download the models via:
```
https://drive.google.com/open?id=18jr9l1YvxDUzqAa_S-LYTdfi6zN1OEE9
```

We note that the [VOID][void_github] dataset has been improved (size increased from ~40K to ~47K frames) since this work was published in RA-L and ICRA 2020. We thank the individuals who reached out and gave their feedback. Hence, to reflect the changes, we retrained our model on VOID. We achieve slightly better performance than the reported numbers in the paper.

| Model            | MAE   | RMSE   | iMAE  | iRMSE  |
| :--------------- | :---: | :----: | :---: | :----: |
| VGG11 from paper | 85.05 | 169.79 | 48.92 | 104.02 |
| VGG11 retrained  | 82.27 | 141.99 | 49.23 | 99.67  |

To achieve the results, we trained for 20 epochs and use a starting learning rate of 5 x 10<sup>-5</sup> up to the 12th epoch, then 2.5 x 10<sup>-5</sup> for 4 epochs, and 1.2 x 10<sup>-5</sup> for the remaining 4 epochs. The weight for smoothness (w<sub>sm</sub>) is changed to 0.15. This is reflected in the `train_voiced_void.sh` bash script.

## Evaluating VOICED <a name="evaluating-voiced"></a>
To evaluate the pretrained VOICED on the KITTI dataset, you may run
```
sh bash/evaluate_voiced_kitti.sh
```

To evaluate the pretrained VOICED on the VOID dataset, you may run
```
sh bash/evaluate_voiced_void.sh
```

You may replace the restore_path and output_path arguments to evaluate your own checkpoints

## Related projects <a name="related-projects"></a>
You may also find the following projects useful:

- [KBNet][kbnet_github]: *Unsupervised Depth Completion with Calibrated Backprojection Layers*. A fast (15 ms/frame) and accurate unsupervised sparse-to-dense depth completion method that introduces a calibrated backprojection layer that improves generalization across sensor platforms. This work is published as an *oral* paper in the International Conference on Computer Vision (ICCV) 2021.
- [ScaffNet][scaffnet_github]: *Learning Topology from Synthetic Data for Unsupervised Depth Completion*. An unsupervised sparse-to-dense depth completion method that first learns a map from sparse geometry to an initial dense topology from synthetic data (where ground truth comes for free) and amends the initial estimation by validating against the image. This work is published in the Robotics and Automation Letters (RA-L) 2021 and the International Conference on Robotics and Automation (ICRA) 2021.
- [AdaFrame][adaframe_github]: *Learning Topology from Synthetic Data for Unsupervised Depth Completion*. An adaptive framework for learning unsupervised sparse-to-dense depth completion that balances data fidelity and regularization objectives based on model performance on the data. This work is published in the Robotics and Automation Letters (RA-L) 2021 and the International Conference on Robotics and Automation (ICRA) 2021.
- [VOICED][voiced_github]: *Unsupervised Depth Completion from Visual Inertial Odometry*. An unsupervised sparse-to-dense depth completion method, developed by the authors. The paper introduces Scaffolding for depth completion and a light-weight network to refine it. This work is published in the Robotics and Automation Letters (RA-L) 2020 and the International Conference on Robotics and Automation (ICRA) 2020.
- [VOID][void_github]: from *Unsupervised Depth Completion from Visual Inertial Odometry*. A dataset, developed by the authors, containing indoor and outdoor scenes with non-trivial 6 degrees of freedom. The dataset is published along with this work in the Robotics and Automation Letters (RA-L) 2020 and the International Conference on Robotics and Automation (ICRA) 2020.
- [XIVO][xivo_github]: The Visual-Inertial Odometry system developed at UCLA Vision Lab. This work is built on top of XIVO. The VOID dataset used by this work also leverages XIVO to obtain sparse points and camera poses.
- [GeoSup][geosup_github]: *Geo-Supervised Visual Depth Prediction*. A single image depth prediction method developed by the authors, published in the Robotics and Automation Letters (RA-L) 2019 and the International Conference on Robotics and Automation (ICRA) 2019. This work was awarded **Best Paper in Robot Vision** at ICRA 2019.
- [AdaReg][adareg_github]: *Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction.* A single image depth prediction method that introduces adaptive regularization. This work was published in the proceedings of Conference on Computer Vision and Pattern Recognition (CVPR) 2019.

We also have works in adversarial attacks on depth estimation methods and medical image segmentation:
- [Stereopagnosia][stereopagnosia_github]: *Stereopagnosia: Fooling Stereo Networks with Adversarial Perturbations.* Adversarial perturbations for stereo depth estimation, published in the Proceedings of AAAI Conference on Artificial Intelligence (AAAI) 2021.
- [Targeted Attacks for Monodepth][targeted_attacks_monodepth_github]: *Targeted Adversarial Perturbations for Monocular Depth Prediction.* Targeted adversarial perturbations attacks for monocular depth estimation, published in the proceedings of Neural Information Processing Systems (NeurIPS) 2020.
- [SPiN][spin_github] : *Small Lesion Segmentation in Brain MRIs with Subpixel Embedding.* Subpixel architecture for segmenting ischemic stroke brain lesions in MRI images, published in the Proceedings of Medical Image Computing and Computer Assisted Intervention (MICCAI) Brain Lesion Workshop 2021 as an **oral paper**.

[kitti_dataset]: http://www.cvlibs.net/datasets/kitti/
[nyu_v2_dataset]: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
[void_github]: https://github.com/alexklwong/void-dataset
[voiced_github]: https://github.com/alexklwong/unsupervised-depth-completion-visual-inertial-odometry
[scaffnet_github]: https://github.com/alexklwong/learning-topology-synthetic-data
[adaframe_github]: https://github.com/alexklwong/adaframe-depth-completion
[kbnet_github]: https://github.com/alexklwong/calibrated-backprojection-network
[xivo_github]: https://github.com/ucla-vision/xivo
[geosup_github]: https://github.com/feixh/GeoSup
[adareg_github]: https://github.com/alexklwong/adareg-monodispnet
[stereopagnosia_github]: https://github.com/alexklwong/stereopagnosia
[targeted_attacks_monodepth_github]: https://github.com/alexklwong/targeted-adversarial-perturbations-monocular-depth
[spin_github]: https://github.com/alexklwong/subpixel-embedding-segmentation

## License and disclaimer <a name="license-disclaimer"></a>
This software is property of the UC Regents, and is provided free of charge for research purposes only. It comes with no warranties, expressed or implied, according to these [terms and conditions](license). For commercial use, please contact [UCLA TDG](https://tdg.ucla.edu).
