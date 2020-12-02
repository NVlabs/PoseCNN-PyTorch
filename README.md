# PoseCNN-PyTorch

PyTorch implementation of the PoseCNN framework.

### Introduction

We implement PoseCNN in PyTorch in this project.

PoseCNN is an end-to-end Convolutional Neural Network for 6D object pose estimation. PoseCNN estimates the 3D translation of an object by localizing its center in the image and predicting its distance from the camera. The 3D rotation of the object is estimated by regressing to a quaternion representation. [arXiv](https://arxiv.org/abs/1711.00199), [Project](https://rse-lab.cs.washington.edu/projects/posecnn/)

The code also supports pose refinement by matching segmented 3D point cloud of an object to its SDF.

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

1. Install [PyTorch](https://pytorch.org/).

2. Install Eigen from the Github source code [here](https://github.com/eigenteam/eigen-git-mirror)

3. Install Sophus from the Github source code [here](https://github.com/yuxng/Sophus)

4. Initialize all submodules
   ```Shell
   git submodule update --init --recursive
   ```

5. Compile the new layers under $ROOT/lib/layers we introduce in PoseCNN.
    ```Shell
    cd $ROOT/lib/layers
    sudo python setup.py install
    ```

6. Compile cython components
    ```Shell
    cd $ROOT/lib/utils
    python setup.py build_ext --inplace
    ```

7. Compile the ycb_render in $ROOT/ycb_render
    ```Shell
    cd $ROOT/ycb_render
    python setup.py develop
    ```

### Download

- 3D models of YCB Objects we used [here](https://drive.google.com/file/d/1PTNmhd-eSq0fwSPv0nvQN8h_scR1v-UJ/view?usp=sharing).

- Our pre-trained models.


### Running the demo

1. Download 3D models and our pre-trained models first.

2. run the following script
    ```Shell
    ./experiments/scripts/demo.sh
    ```

### Training your own models with synthetic data

1. Download background images: our own images [here](https://drive.google.com/file/d/1Q5VTKHEEejT2lAKwefG00eWcrnNnpieC/view?usp=sharing) (7G), COCO 2014 images [here](https://cocodataset.org/#download), or use your own background images.

2. Download pretrained VGG16 weights: [here](https://drive.google.com/file/d/1tTd64s1zNnjONlXvTFDZAf4E68Pupc_S/view?usp=sharing) (528M). Put the weight file to $ROOT/data/checkpoints.

3. Training and testing for 20 YCB objects with synthetic data. Modify the configuration file for training on a subset of these objects.
    ```Shell
    cd $ROOT

    # multi-gpu training, use 1 GPU or 2 GPUs since batch size is set to 2
    ./experiments/scripts/ycb_object_train.sh

    # testing, $GPU_ID can be 0, 1, etc.
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

### Running with ROS on a Realsense Camera for real-world pose estimation
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
