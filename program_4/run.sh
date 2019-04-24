#!/bin/bash
docker run --runtime=nvidia -u $(id -u):$(id -g) -it --rm -v $(realpath ~/git/ecen-489-500/program_4):/tf/notebooks -p 8888:8888 program_4:latest
