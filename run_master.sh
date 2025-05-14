#!/bin/bash
export MASTER_ADDR=[10.0.0.4]
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0

python3 distributed_mnist.py