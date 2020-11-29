# posecnn-pytorch

PyTorch implementation of the PoseCNN framework.

### Introduction

We introduce PoseCNN, a new Convolutional Neural Network for 6D object pose estimation. PoseCNN estimates the 3D translation of an object by localizing its center in the image and predicting its distance from the camera. The 3D rotation of the object is estimated by regressing to a quaternion representation. [arXiv](https://arxiv.org/abs/1711.00199), [Project](https://rse-lab.cs.washington.edu/projects/posecnn/)

[![PoseCNN](http://yuxng.github.io/PoseCNN.png)](https://youtu.be/ih0cCTxO96Y)

### License

PoseCNN is released under the MIT License (refer to the LICENSE file for details).

### Citation

If you find the package is useful in your research, please consider citing:

    @inproceedings{xiang2018posecnn,
        Author = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
        Title = {PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes},
        booktitle   = {Robotics: Science and Systems (RSS)},
        Year = {2018}
    }

### Installation

1. Install [PyTorch](https://pytorch.org/).

2. Initialize all submodules
   ```Shell
   git lfs clone
   git submodule update --init --recursive
   ```

3. Install Eigen from the Github source code [here](https://github.com/eigenteam/eigen-git-mirror)

4. Install Sophus from the Github source code [here](https://github.com/strasdat/Sophus)
   Check out this commit "ceb6380a1584b300e687feeeea8799353d48859f"

5. Compile the new layers under $ROOT/lib/layers we introduce in PoseCNN.
    ```Shell
    cd $ROOT/lib/layers
    python setup.py install
    ```

6. Download the VGG16 weights from [here](https://drive.google.com/file/d/1tTd64s1zNnjONlXvTFDZAf4E68Pupc_S/view?usp=sharing) (528M). Put the weight file to $ROOT/data/checkpoints.

7. Compile the ycb_render in $ROOT/ycb_render

8. Compile cython components
    ```Shell
    cd $ROOT/lib
    python setup.py build_ext --inplace
    ```

### Background datasets
- AllenCenter [here](https://drive.google.com/open?id=1YDnGV4poelk9iezxLxYK_zexXugc4Ih1)

### Required environment
- Ubuntu 16.04
- PyTorch 0.4.1 or above
- CUDA 9.1 or above

### Running the demo
1. Download our trained model on five YCB Objects from [here](https://drive.google.com/open?id=1fxfBBCOPqSMYARiJQBc8ZjcWq5LiLHDq), and save it to $ROOT/data/checkpoints.

2. Download the 3D models of the YCB Objects from [here](https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu/view?usp=sharing), and save it to $ROOT/data/YCB_Object.

3. run the following script
    ```Shell
    ./experiments/scripts/demo.sh $GPU_ID
    ```

### Running on the YCB-Video dataset
1. Download the YCB-Video dataset from [here](https://rse-lab.cs.washington.edu/projects/posecnn/).

2. Create a symlink for the YCB-Video dataset
    ```Shell
    cd $ROOT/data/YCB_Video
    ln -s $ycb_data data
    ln -s $ycb_models models
    ```

3. Training and testing on the YCB-Video dataset
    ```Shell
    cd $ROOT

    # multi-gpu training
    ./experiments/scripts/ycb_video_train.sh

    # testing
    ./experiments/scripts/ycb_video_test.sh $GPU_ID

    ```

### Running with ROS
    ```Shell
    # start realsense
    roslaunch realsense2_camera rs_aligned_depth.launch tf_prefix:=measured/camera

    # start rviz
    rosrun rviz rviz -d ./ros/posecnn.rviz

    # run posecnn
    ./experiments/scripts/ros_ycb_object_test.sh $GPU_ID
    ```
