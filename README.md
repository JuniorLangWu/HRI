# Continuous-hand Gesture-based Human-Robot Interaction

The code is divided into two parts: the gesture recognition model and the ROS mapping.Due to WHEELTEC ROS mobile social
robot code is closed source, we only provide our own gesture mapping implementation of the ROS package.

## Gesture recognition model 
### Installation
```shell
git clone https://github.com/JuniorLangWu/HRI
cd HRI/MSGL
# This command runs well with conda 24.5.0, if you are running an early conda version and got some errors, try to update your conda first
# The code is based on the CUDA=11.3 framework.
conda create -n hri python=3.8
conda activate hri
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
MMCV_WITH_OPS=1 pip install mmcv-full==1.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install mediapipe==0.10.11 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install pandas==2.0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn==1.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install fvcore==0.1.5.post20221221  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install yapf==0.32.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -e .

# install rospkg for ROS
pip install rospkg==1.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Isolated Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Ask the author to provide two datasets for processing skeletal data.
# Training
cd HRI/MSGL
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing
cd HRI/MSGL
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy

# Example
cd HRI/MSGL
bash tools/dist_train.sh configs/stgcn++_hand/stfgcn_ipn_No_mediapipe_snip_j.py  1  --validate --test-last --test-best  --deterministic
```

### Continuous testing
```shell
# First, you need to download the two public datasets from  https://github.com/GibranBenitez/IPN-hand and https://github.com/Diananini/LD-ConGR-CVPR2022.
# Then decompress and place them in the video location of the MSGL/configs/online configuration file.

# Testing
cd HRI/MSGL
bash {bash_name} --clf_queue_size {num_queue} --sample_duration_clf {num_window} --set_work

# Example
cd HRI/MSGL
bash tools/online_ipn_test_two.sh --clf_queue_size 38 --sample_duration_clf 76 --set_work
```
## ROS mapping
Using WHEELTEC ROS mobile social  for human-robot interaction
### ROS configuration
```shell

cd HRI/wheeltec/src
catkin_init_workspace

cd ..
catkin_make

# Add environment variables to the .bashrc file. Change ... to your path.
source /.../devel/setup.bash
```

### Local testing
```shell
roscore

conda activate hri
roslaunch wheeltec_robot_rc keyboard_teleop_gesture.launch 
```

### HRI testing
```shell
# First connect the robot.
# Then
roslaunch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch
conda activate hri
roslaunch wheeltec_robot_rc keyboard_teleop_gesture.launch 
```

## Acknowledgement
This project is inspired by many previous works, including:
* [Pyskl: Towards good practices for skeleton action recognition](https://doi.org/10.48550/arXiv.2205.09443), Duan et al, _2022_ [[code](https://github.com/kennymckormick/pyskl)]
* [IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition](https://arxiv.org/abs/2005.02134), Benitez-Garcia et al, _2021_ [[code](https://github.com/GibranBenitez/IPN-hand)]
* [LD-ConGR: A Large RGB-D Video Dataset for Long-Distance Continuous Gesture Recognition](https://ieeexplore.ieee.org/document/9878595), Liu et al, _2022_ [[code](https://github.com/Diananini/LD-ConGR-CVPR2022)]
* [Constructing Stronger and Faster Baselines for Skeleton-Based Action Recognition](https://ieeexplore.ieee.org/document/9729609), Song et al, _2023_ [[code](https://github.com/yfsong0709/EfficientGCNv1)]
* [FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dc1e32dd3eb381dbc71482f6a96cbf86-Abstract-Conference.html), Yi et al, _2023_ [[code](https://github.com/aikunyi/FourierGNN)]
