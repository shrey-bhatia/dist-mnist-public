# Distributed Neural Network Training

This project explores scaling performance in distributed neural network training across multiple compute nodes. It is part of my final year project for the University of Leeds.

The code provided in this repository is designed to run on two Azure VMs. Additionally, I have included the code I initially used to run the training on a single node. Regrettably, I had issues with version control so I am unable to provide the original code. I was able to salvage this code from a local backup, as well as the code stored on the actual Azure virtual machines. However, I have included the final version of the code that I used for distributed training.

## Research Question
How does neural network training performance scale across distributed computing nodes, and which factors limit achieving perfect linear performance gains?

## Setup
- 2 Azure VMs running Ubuntu 22.04 communicating using TCP
- PyTorch distributed training
- MNIST dataset
