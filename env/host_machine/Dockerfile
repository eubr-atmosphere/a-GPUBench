#Enviroment used on the host machine
FROM python:3.7.11-slim-buster

COPY /new_requirements.txt .
RUN apt-get -y update && apt-get install -y openssh-client rsync 

RUN python3 -m pip install --upgrade setuptools wheel
RUN  python3 -m pip install -r new_requirements.txt 

RUN apt-get -y install apt-utils sshpass



CMD tail -f /dev/null