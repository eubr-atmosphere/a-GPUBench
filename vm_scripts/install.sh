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
apt dist-upgrade -y

apt install -y python3 python3-pip screen wget ssmtp lshw lsb-release sox software-properties-common libssl-dev libsox-fmt-mp3

if [ -f a-GPUBench/vm_scripts/ssmtp.conf ]; then
   cp a-GPUBench/vm_scripts/ssmtp.conf /etc/ssmtp/ssmtp.conf
fi
if [ -f a-GPUBench/vm_scripts/revaliases ]; then
   cp a-GPUBench/vm_scripts/revaliases /etc/ssmtp/revaliases
fi

#Install cuda 9.0 since cuda 9.1 is not officially supported by pytorch 0.3.1
if [ ! -f cuda-repo-ubuntu1604_9.0.176-1_amd64.deb ]; then
   wget developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
fi
dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt update
apt install -y cuda-9-0


if [ ! -f libcudnn7_7.1.3.16-1+cuda9.0_amd64.deb ]; then
   wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.1.3.16-1+cuda9.0_amd64.deb
fi

if [ -f libcudnn7_7.1.3.16-1+cuda9.0_amd64.deb ]; then
   dpkg --install libcudnn7_7.1.3.16-1+cuda9.0_amd64.deb
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
pip3 install tensorflow-gpu==1.8.0

#tf_deepspeech modules
add-apt-repository -y ppa:git-core/ppa
apt update
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
apt install -y git-lfs libffi-dev
git lfs install
pip3 install pandas
pip3 install progressbar2
pip3 install python-utils
pip3 install numpy
pip3 install matplotlib
pip3 install scipy
pip3 install paramiko >= 2.1
pip3 install pysftp
pip3 install sox
pip3 install python_speech_features
pip3 install pyxdg
pip3 install bs4
pip3 install six
pip3 install requests
pip3 install deepspeech-gpu==0.1.1

#Change owner of mnt
USER=$(stat -c "%U" $0)
chown $USER:$USER /mnt

#Create symbolic link to mnt
ln -s /mnt tmp

#If UUID file already exists, check that it is still the same
if [ -e /etc/system_uuid ];
then
   stored_sytemd_uuid=$(cat /etc/system_uuid)
   runtime_system_uuid=$(dmidecode | grep UUID)
   if [ "$runtime_system_uuid" != "$stored_sytemd_uuid" ];
   then
      echo "New uuid $runtime_system_uuid (old was $stored_sytemd_uuid)"
      exit 1
   fi
#If it does not yet exist, save system UUID in a file
else
   dmidecode | grep UUID > /etc/system_uuid
fi
