#!/usr/bin/python3
# -*- coding: utf-8 -*-
from subprocess import Popen, PIPE


def get_available_gpus():
    #list only Available NVIDIA GPUs
    list_gpus_command = 'lspci | grep -E "(VGA.*NVIDIA|3D.*NVIDIA)"'

    proc = Popen(list_gpus_command, shell=True, stdout=PIPE)
    lines = proc.stdout.readlines()
    num_gpus = len(lines)
    proc.stdout.close()
    proc.wait()

    return num_gpus