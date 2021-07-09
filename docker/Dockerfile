FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER HDC

# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive

# Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake \
    libopenblas-dev

# Needed for string substitution 
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
	libgl1-mesa-dev

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH


# Python 3.6
# For convenience, alias (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
RUN apt-get install -y --no-install-recommends python3.6 python3.6-dev python3-pip python3-tk && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases
# Pillow and it's dependencies
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && \
    pip3 --no-cache-dir install setuptools Pillow
# Science libraries and other common packages
RUN pip3 --no-cache-dir install \
    numpy scipy sklearn scikit-image imgaug opencv-python IPython[all] matplotlib Cython requests

RUN pip3 --no-cache-dir install \
    plyfile 'trimesh>=2.35.39,<2.35.40' 'networkx>=2.2,<2.3'

# GraspNetAPI
RUN pip3 --no-cache-dir install \
    transforms3d==0.3.1 'open3d>=0.8.0.0' tqdm pywavefront autolab_core autolab-perception cvxopt dill h5py grasp_nms

# Jupyter Notebook
# Allow access from outside the container, and skip trying to open a browser.
# NOTE: disable authentication token for convenience. DON'T DO THIS ON A PUBLIC SERVER.
RUN pip3 --no-cache-dir install jupyter && \
    mkdir /root/.jupyter && \
    echo "c.NotebookApp.ip = '*'" \
         "\nc.NotebookApp.open_browser = False" \
         "\nc.NotebookApp.token = ''" \
         > /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888

# Tensorflow 1.14 - GPU
RUN python3 -m pip install --no-cache-dir tensorflow-gpu==1.14

# Pytorch 1.6
RUN pip install torch==1.6.0 torchvision==0.7.0 

# Expose port for TensorBoard
EXPOSE 6006
