#Dockerfile of the enviroment used for the training session on the target machine
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04


#Copy script from host to cointainer
COPY new_requirements.txt / 
RUN echo 'root:root' | chpasswd
ENV TZ="Europe/Rome" DEBIAN_FRONTEND="noninteractive"

#Install dependencies
RUN apt-get update
RUN apt-get -y install screen wget ssmtp lshw lsb-release sox software-properties-common libssl-dev libsox-fmt-mp3 ffmpeg libsm6 libxext6
RUN apt-get -y install software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get -y install python3.7 python3-pip rsync grsync python3-opencv

#Install python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install setuptools_rust &&\
    python3 -m pip install -r new_requirements.txt  --src /usr/local/src && \
    apt-get -y install python3-opencv

CMD tail -f /dev/null

