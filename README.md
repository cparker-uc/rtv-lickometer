# Real-Time Video Lickometer

This repository contains the various modules used in the training and usage of
a machine vision lickometry system. Our use case is in head-fixed mice undergoing
aversion-resistant alcohol drinking trials, but there's no reason it couldn't be
adapted past lickometry!


## Contents:
- rtv-lickometer: this is the Rust binary for running the system after the network has been
fully trained, etc. For now, it's a placeholder.
- rtv-lickometer-training: this is the Rust binary for gathering training data. It presents
a GUI and allows the user to select an ROI in the frame (we only care about a small area
in the frame, but the size can be modified easily).
- rtv-lickometer-network: Python scripts for training/testing the network. The network
architecture is currently based on the paper "Dissected 3D CNNs" 
(https://www.sciencedirect.com/science/article/pii/S1077314221001594)
