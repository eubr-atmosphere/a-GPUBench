#!/bin/bash
"""
Copyright 2018 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#Wait if there is any unattended apt already running
running_apt=`ps aux | grep apt | wc -l`
while [ "$running_apt" -ne "1" ]
do
   echo "Waiting 60 seconds for the end of already running apt"
   sleep 60
   running_apt=`ps aux | grep apt | wc -l`
done

apt update
apt dist-upgrade

apt install -y python3 python3-pip screen wget ssmtp lshw lsb-release

if [ -f a-GPUBench/vm_scripts/ssmtp.conf ]; then
   cp a-GPUBench/vm_scripts/ssmtp.conf /etc/ssmtp/ssmtp.conf
fi
if [ -f a-GPUBench/vm_scripts/revaliases ]; then
   cp a-GPUBench/vm_scripts/revaliases /etc/ssmtp/revaliases
fi

#Install cuda 9.0 since cuda 9.1 is not officially supported by pytorch 0.3.1
wget developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt update
apt install -y cuda-9-0

if [ -f a-GPUBench/vm_scripts/libcudnn7_7.1.3.16-1+cuda9.0_amd64.deb ]; then
   dpkg --install a-GPUBench/vm_scripts/libcudnn7_7.1.3.16-1+cuda9.0_amd64.deb
fi

echo "export PATH=/usr/local/cuda/bin\${PATH:+:\${PATH}}" >> ~/.bashrc
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
CUDA_VERSION=`nvcc --version | grep release | awk '{print $5}' | awk -F',' '{print $1}' | tr -d '.'`
echo "Cuda Version is" $CUDA_VERSION
PYTHON3_VERSION=`python3 --version | awk '{print $2}' | awk -F. '{print $1$2}'`
echo "Python3 version is" $PYTHON3_VERSION
pip3 install http://download.pytorch.org/whl/cu"$CUDA_VERSION"/torch-0.3.1-cp"$PYTHON3_VERSION"-cp"$PYTHON3_VERSION"m-linux_x86_64.whl
pip3 install torchvision
pip3 install dicttoxml xmltodict
pip3 install scipy
pip3 install tensorflow-gpu

#Change owner of mnt
USER=$(stat -c "%U" $0)
chown $USER:$USER /mnt

#Create symbolic link to mnt
ln -s /mnt tmp
