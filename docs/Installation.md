---
layout: default
title: Installation
nav_order: 1
---
# Install the necessary packages for StarLight
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---



## Installation
- Create a conda envirment. You can use the Tsinghua source for the conda and pip to accelerate installation. 

```shell
  conda create -n starlight python=3.6
  conda activate starlight
```

- Install PyTorch and cuDNN.

```shell
  conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2
  conda install --channel https://conda.anaconda.org/nvidia cudnn=8.0.0
```

- Install TensorRT.

```shell
  # Go to https://developer.nvidia.com/compute/machine-learning/tensorrt
  # Download TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz 
  tar -zxf TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
  cd TensorRT-7.1.3.4/ 
  pip install python/tensorrt-7.1.3.4-cp36-none-linux_x86_64.whl
  pip install uff/uff-0.6.9-py2.py3-none-any.whl
  pip install graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl
  
  # Test if the TensorRT is installed successfully
  python
  import tensorrt 
  # No error mean success, we have summarized two common bugs in the next section
```

- Install PYQT5 and PyQtWebEngine:

```shell
  pip install pyqt5==5.12
  pip install PyQtWebEngine==5.12
```

- Install other packages.

```shell
  pip install easydict opencv-python flask flask_cors gevent imageio pynvml pyyaml psutil matplotlib pycocotools Cython thop schema prettytable
  pip install onnx==1.11.0 pycuda==2019.1.1 tensorboard==2.9.1 tqdm
  pip install opencv-python pdf2image 
```

## Bugs and solutions

- If `libnvinfer.so.7` or `libcudnn.so.8` is missing when you import the tensorrt, simply specify there direction in the `~/.bashrc`:

```shell
# search their direction
find / -name libnvinfer.so.7
find / -name libcudnn.so.8
# for libnvinfer.so.7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/env/TensorRT-7.1.3.4/lib
# for libcudnn.so.8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/envs/starlight/lib
```

- If ImportError: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.27' not found, build GLIBC_2.27 manually:

```shell
# download glibc-2.27
wget http://ftp.gnu.org/gnu/glibc/glibc-2.27.tar.gz
tar -zxf glibc-2.27.tar.gz
cd glibc-2.27
mkdir build
cd build/
../configure --prefix=/opt/glibc-2.17 # <-- where you install glibc-2.27
# if error for gawk/bison, install them using: sudo apt-get install gawk/bison
make -j <number of CPU Cores>  # You can find your <number of CPU Cores> by using `nproc` command
make install
# patch your Python
patchelf --set-interpreter /opt/glibc-2.17/lib/ld-linux-x86-64.so.2 --set-rpath /opt/glibc-2.17/lib/ /root/anaconda3/envs/starlight/bin/python
```