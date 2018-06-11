#!/bin/bash
while true; do (echo "Timestamp $(date +%s%N) ns - $(date '+%Y-%m-%d %H:%M:%S%N')" && nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv); sleep $1; done
