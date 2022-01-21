# check out repo, pull submodules

after checking out, inside repo dir,  give 
```bash
git submodule update --init
```


# docker or not
 it is possible to run the programs with or without docker container.
 docker container standardize env so we can all use the same base system and such base system is reproducible very easily and quickly.


# without docker

install cuda-11.3, libtorch-1.10-cuda11.3, tensorrt 8.0.3, 

get libtorch 1.10-cuda11.3 from https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcu113.zip

## setup venv to run python 
```bash
./create_venv_install_python_deps.sh
```

activate venv 
```bash
source venv/bin/activate
```

set PYTHONPATH to make python code work

```bash
 export PYTHONPATH=/home/<folders>/point_pillar_tensorrt/python:$PYTHONPATH
```

# with docker

## prepare docker env to run nvidia docker images

follow instructions on nvidia website 
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt


### setup to use user and X from docker, run docker as user
```bash
sudo usermod -aG docker `id -un`
xhost +"local:docker@"
sudo chmod 666 /var/run/docker.sock
```
reboot the machine for the changes to take effect


## download libtorch to prepare docker image

```bash
wget -O docker-tensorrt-8-cuda-11.3/libtorch-cxx11-abi-shared-with-deps-1.10.0+cu113.zip https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.10.0%2Bcu113.zip
```

## build docker image
```bash
cd docker-tensorrt-8-cuda-11.3
docker build . -t nvidia-trt-8-cuda-11.3-libtorch-1.10

```


## launch docker env using script
in directory docker-tensorrt-8-cuda-11.3

```bash
./lauch.sh
```

modify the launch script to mount directory according to your system (see flags -v in script)

## setup venv_docker to use trt docker

install python deps in venv_docker directory. this directory is inside home directory, not iside docker. docker launch commad mounts host home directory in docker home directory

inside docker env go to this directory where you find the script
```bash
create_venv_docker_install_python_deps.sh
```

it should be same directory of this README

launch the script to install venv to make python code works
```bash
./create_venv_docker_install_python_deps.sh
```

activate venv 
```bash
source venv_docker/bin/activate
```