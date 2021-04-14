# PoseCNN-PyTorch: A PyTorch Implementation of the PoseCNN Framework for 6D Object Pose Estimation

### Introduction

We implement PoseCNN in PyTorch in this project.

PoseCNN is an end-to-end Convolutional Neural Network for 6D object pose estimation. PoseCNN estimates the 3D translation of an object by localizing its center in the image and predicting its distance from the camera. The 3D rotation of the object is estimated by regressing to a quaternion representation. [arXiv](https://arxiv.org/abs/1711.00199), [Project](https://rse-lab.cs.washington.edu/projects/posecnn/)

Rotation regression in PoseCNN cannot handle symmetric objects very well. Check [PoseRBPF](https://github.com/NVlabs/PoseRBPF) for a better solution for symmetric objects.

The code also supports pose refinement by matching segmented 3D point cloud of an object to its SDF.

<p align="center"><img src="./data/pics/intro.png" width="640" height="320"/></p>

### License

PoseCNN-PyTorch is released under the NVIDIA Source Code License (refer to the LICENSE file for details).

### Citation

If you find the package is useful in your research, please consider citing:

    @inproceedings{xiang2018posecnn,
        Author = {Yu Xiang and Tanner Schmidt and Venkatraman Narayanan and Dieter Fox},
        Title = {{PoseCNN}: A Convolutional Neural Network for {6D} Object Pose Estimation in Cluttered Scenes},
        booktitle = {Robotics: Science and Systems (RSS)},
        Year = {2018}
    }

### Required environment

- Ubuntu 16.04 or above
- PyTorch 0.4.1 or above
- CUDA 9.1 or above

### Installation

Use python3. If ROS is needed, compile with python2.

1. Install [PyTorch](https://pytorch.org/)

2. Install Eigen from the Github source code [here](https://github.com/eigenteam/eigen-git-mirror)

3. Install Sophus from the Github source code [here](https://github.com/yuxng/Sophus)

4. Install python packages
   ```Shell
   pip install -r requirement.txt
   ```

5. Initialize the submodules in ycb_render
   ```Shell
   git submodule update --init --recursive
   ```

6. Compile the new layers under $ROOT/lib/layers we introduce in PoseCNN
    ```Shell
    cd $ROOT/lib/layers
    sudo python setup.py install
    ```

7. Compile cython components
    ```Shell
    cd $ROOT/lib/utils
    python setup.py build_ext --inplace
    ```

8. Compile the ycb_render in $ROOT/ycb_render
    ```Shell
    cd $ROOT/ycb_render
    sudo python setup.py develop
    ```

### Download

- 3D models of YCB Objects we used [here](https://drive.google.com/file/d/1PTNmhd-eSq0fwSPv0nvQN8h_scR1v-UJ/view?usp=sharing) (3G). Save under $ROOT/data or use a symbol link.

- Our pre-trained checkpoints [here](https://drive.google.com/file/d/1-ECAkkTRfa1jJ9YBTzf04wxCGw6-m5d4/view?usp=sharing) (4G). Save under $ROOT/data or use a symbol link.

- Our real-world images with pose annotations for 20 YCB objects collected via robot interation [here](https://drive.google.com/file/d/1cQH_dnDzyrI0MWNx8st4lht_q0F6cUrE/view?usp=sharing) (53G). Check our ICRA 2020 [paper](https://arxiv.org/abs/1909.10159) for details.


### Running the demo

1. Download 3D models and our pre-trained checkpoints first.

2. run the following script
    ```Shell
    ./experiments/scripts/demo.sh
    ```

<p align="center"><img src="./data/pics/posecnn.png" width="640" height="360"/></p>

### Training your own models with synthetic data for YCB objects

1. Download background images, and save to $ROOT/data or use symbol links.

    - Our own images [here](https://drive.google.com/file/d/1Q5VTKHEEejT2lAKwefG00eWcrnNnpieC/view?usp=sharing) (7G)
    - COCO 2014 images [here](https://cocodataset.org/#download)
    - Or use your own background images

2. Download pretrained VGG16 weights: [here](https://drive.google.com/file/d/1tTd64s1zNnjONlXvTFDZAf4E68Pupc_S/view?usp=sharing) (528M). Put the weight file to $ROOT/data/checkpoints. If our pre-trained models are already downloaded, the VGG16 checkpoint should be in $ROOT/data/checkpoints already.

3. Training and testing for 20 YCB objects with synthetic data. Modify the configuration file for training on a subset of these objects.
    ```Shell
    cd $ROOT

    # multi-gpu training, use 1 GPU or 2 GPUs since batch size is set to 2
    ./experiments/scripts/ycb_object_train.sh

    # testing on synthetic data, $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ycb_object_test.sh $GPU_ID

    ```

### Training and testing on the YCB-Video dataset
1. Download the YCB-Video dataset from [here](https://rse-lab.cs.washington.edu/projects/posecnn/).

2. Create a symlink for the YCB-Video dataset
    ```Shell
    cd $ROOT/data/YCB_Video
    ln -s $ycb_data data
    ```

3. Training and testing on the YCB-Video dataset
    ```Shell
    cd $ROOT

    # multi-gpu training, use 1 GPU or 2 GPUs since batch size is set to 2
    ./experiments/scripts/ycb_video_train.sh

    # testing, $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ycb_video_test.sh $GPU_ID

    ```

### Training and testing on the DexYCB dataset
1. Download the DexYCB dataset from [here](https://dex-ycb.github.io/).

2. Create a symlink for the DexYCB dataset
    ```Shell
    cd $ROOT/data/DEX_YCB
    ln -s $dex_ycb_data data
    ```

3. Training and testing on the DexYCB dataset
    ```Shell
    cd $ROOT

    # multi-gpu training for different splits, use 1 GPU or 2 GPUs since batch size is set to 2
    ./experiments/scripts/dex_ycb_train_s0.sh
    ./experiments/scripts/dex_ycb_train_s1.sh
    ./experiments/scripts/dex_ycb_train_s2.sh
    ./experiments/scripts/dex_ycb_train_s3.sh

    # testing, $GPU_ID can be 0, 1, etc.
    # our trained models are in checkpoints.zip
    ./experiments/scripts/dex_ycb_test_s0.sh $GPU_ID $EPOCH
    ./experiments/scripts/dex_ycb_test_s1.sh $GPU_ID $EPOCH
    ./experiments/scripts/dex_ycb_test_s2.sh $GPU_ID $EPOCH
    ./experiments/scripts/dex_ycb_test_s3.sh $GPU_ID $EPOCH

    ```

### Running with ROS on a Realsense Camera for real-world pose estimation

- Python2 is needed for ROS.

- Make sure our pretrained checkpoints are downloaded.

```Shell
# start realsense
roslaunch realsense2_camera rs_aligned_depth.launch tf_prefix:=measured/camera

# start rviz
rosrun rviz rviz -d ./ros/posecnn.rviz

# run posecnn for detection only (20 objects), $GPU_ID can be 0, 1, etc.
./experiments/scripts/ros_ycb_object_test_detection.sh $GPU_ID

# run full posecnn (20 objects), $GPU_ID can be 0, 1, etc.
./experiments/scripts/ros_ycb_object_test.sh $GPU_ID
```

Our example:
<p align="center"><img src="./data/pics/posecnn.gif"/></p>
