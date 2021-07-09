
# Context-Aware Grasp Detection for Target Objects in Cluttered Scenes Using Deep Hough Voting

## Introduction
This repository is code release for our VoteGrasp paper.

In this repository, we provide implementation of the proposed method (with Pytorch):
1. VoteGrasp model can be found in [models/votegrasp.py](https://github.com/votegrasp/votegrasp/blob/master/models/votegrasp.py)
2. Context learning module and grasp generation module can be found in [models/proposal_module.py](https://github.com/votegrasp/votegrasp/blob/master/models/proposal_module.py)
3. Loss function can found in [models/loss_helper.py](https://github.com/votegrasp/votegrasp/blob/master/models/loss_helper.py)

## Installation

Install [Pytorch](https://pytorch.org/get-started/locally/) and [Tensorflow](https://github.com/tensorflow/tensorflow) (for TensorBoard). It is required that you have access to GPUs. The code is tested with Ubuntu 16.04, Pytorch v1.1, TensorFlow v1.14, CUDA 10.0 and cuDNN v7.4.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd pointnet2
    python setup.py install

Install the following Python dependencies (with `pip install`):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'

## Training

#### Data preparation

Prepare data by running `python ycbgrasp/ycbgrasp_data.py --gen_data`

#### Train

To train a new VoteGrasp model on the ycbgrasp data (synthetic):

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset ycbgrasp --log_dir log_ycbgrasp

You can use `CUDA_VISIBLE_DEVICES=0,1,2` to specify which GPU(s) to use. Without specifying CUDA devices, the training will use all the available GPUs and train with data parallel (Note that due to I/O load, training speedup is not linear to the nubmer of GPUs used). Run `python train.py -h` to see more training options.
While training you can check the `log_ycbgrasp/log_train.txt` file on its progress.

#### Run predict

    python predict.py

## Create and Train on your own data

If you have your own objects with 3D meshes, you can create a new dataset for your objects using our tools:

1. capturing depth images of a scene with our [blender-scipts](https://github.com/votegrasp/blender-scripts)
2. Generating synthetic point clouds using our [simulator](https://github.com/votegrasp/simulation_grasping)
3. Generating grasps from meshes using [graspit](https://github.com/votegrasp/graspit)
4. data processing in ground-truth data creation stage with our [ros_votegrasp](https://github.com/votegrasp/ros_votegrasp)

## Visualization
Please use our [ros_votegrasp](https://github.com/votegrasp/ros_votegrasp)

## Real robot grasping
Please use our [Robotlab_Franka](https://github.com/votegrasp/real_robot_grasping)

## Dataset and trained model
Please find more information on [our website](https://sites.google.com/view/votegrasp).

## Acknowledgements
Will be available after our paper has been published.

## License
Will be available after our paper has been published.
