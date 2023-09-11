# Image Classification with Homeostasis using Spiking Neural Network

This repository presents the final project for the Computational Neuroscience course at the University of Tehran, Department of Computer Science, conducted by **Amir Naderi (Student ID: 610398126)** under the supervision of **M. Ganjtabesh**. The project focuses on feature extraction and image classification from grayscale images using convolution and max-pooling layers within a spiking neural network (SNN).

## Introduction

In this project, we aim to extract meaningful features from images utilizing convolution and max-pooling techniques. The neural network model employed consists of three layers with convolutional connections between the first and second layers, followed by pooling connections between the second and third layers. The key features of this project include:

- Utilization of 5 filters with random weight initialization.
- Training of filter weights using the Spike-Timing-Dependent Plasticity (STDP) learning rule.
- Handling of grayscale images with a single channel.
- Usage of 5x5 kernels for the convolution operation.

## Learning Features with STDP

The project uses grayscale images of faces, converts them to grayscale, and applies Time to First Spike (TTFS) encoding. TTFS encoding focuses on the timing of the first spike to transmit information. Convolutional and pooling connections are established in the model, allowing the weights of the filters to be trained by the STDP rule.

The raster plot displays the simulation results, highlighting the spikes in different layers of the model. The application of STDP enables the filters to learn frequent features from the images.

## Classification Using RSTDP

The project explores Reward-modulated Spike-Timing-Dependent Plasticity (R-STDP), a learning method for Spiking Neural Networks (SNNs). R-STDP combines the advantages of reinforcement learning with the biological plausibility of STDP. While the code for this section is available in the Jupyter notebook, the results are not yet finalized.

For further details, code implementations, and results, please refer to the respective sections and files within this repository.