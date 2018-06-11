#!/bin/bash
while true; do (echo "Timestamp $(date +%s%N) ns - $(date '+%Y-%m-%d %H:%M:%S%N')\n%CPU %MEM ARGS" && ps -e -o pcpu,pmem,args --sort=pcpu | cut -d" " -f1-5 | tail -n30); sleep $1; done
