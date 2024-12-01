#!/bin/bash
#podman system prune -a #clean all the containers, images etc
podman container prune
podman run --rm -it --gpus 1 --security-opt=label=disable --device=nvidia.com/gpu=all --shm-size 12G -v /home/lys/Dropbox/PostDoc/dromi/:/opt/project:Z -w /opt/project dromi39 bash