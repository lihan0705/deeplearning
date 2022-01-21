# 服务器docker远程开发

## 服务器环境配置

服务器，以下简称host，先装好docker, 显卡驱动， 和nvidia-docker

## 创建一个容器

```
sudo nvidia-docker run -p ContainerPort:22 -itd --name="myname" -v /home/ld_alg_dev/datasets:/workspace/datasets:ro --shm-size 60g --gpus all shawndeu/dl_env:ubuntu18-cuda10.2-cudnn8-py37-torch17
```
-p 指定容器端口

-v 挂载数据集目录

--shm-size 内存60g

## 进入容器

```
sudo docker exec -it myname bash
```

安装ssh

```
apt-get update
apt-get install openssh-server
apt-get install nano
```

允许ssh远程登录容器

```
nano /etc/ssh/sshd_config
```
把 PermitRootLogin prohisbit-password取消注释且改为PermitRootLogin yes

创建root密码
```
passwd root
```

重启ssh

```
service ssh restart
```

保证ssh登录后环境变量不变
```
nano /etc/profile
```
在最后加上
```
export $(cat /proc/1/environ |tr '\0' '\n' | xargs)
```
保存退出

## ssh登录

打开自己电脑的终端，输入
```
ssh -p ContainerPort root@server_ip
```
ContainerPort就是创建容器时设置的端口，server_ip就是服务器的ip

可以用vscode远程开发，参考

https://www.jianshu.com/p/0f2fb935a9a1

## 备注

第一次可以从git上拉取原来的代码，git已经安装在容器了，git ssh key可以按https://docs.gitlab.com/ee/ssh/ 添加

以后可以直接在vscode（容器）里开发和调试，调试好了push到gitlab

vscode的插件需要在容器里重新安装

数据集所有容器共享，只读，容器内路径为/workspace/datasets