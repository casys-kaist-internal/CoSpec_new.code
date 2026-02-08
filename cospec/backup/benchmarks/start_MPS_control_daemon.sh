#!/bin/bash
# Make sure that the user is same for server and client 
# Using the provided integer to create unique directories
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log"

# Start the MPS daemon
nvidia-cuda-mps-control -f
