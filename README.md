# Distributed Neural Network Training

This project explores scaling performance in distributed neural network training across multiple compute nodes. It is part of my final year project for the University of Leeds.

The code provided in this repository is designed to run on two Azure VMs. Additionally, I have included the code I initially used to run the training on a single node. Regrettably, I had issues with version control so I am unable to provide the original code. I was able to salvage this code from a local backup, as well as the code stored on the actual Azure virtual machines. However, I have included the final version of the code that I used for distributed training.

I have included the final output model from the training to demonstrate that the code works. It is saved as the "distribtued_mnist_model.pth" file. The model is a simple neural network trained on the MNIST dataset that can be used to classify handwritten digits. 

## Research Question
How does neural network training performance scale across distributed computing nodes, and which factors limit achieving perfect linear performance gains?

## Setup
- 2 Azure VMs running Ubuntu 22.04 communicating using TCP
- PyTorch distributed training
- MNIST dataset


If you have any questions or need further information, please feel free to reach out to me. 