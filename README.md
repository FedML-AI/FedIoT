# FedIoT: Federated Learning for Internet of Things
 
<!-- This is FedIoT, an application ecosystem for federated IoT based on FedML framework (https://github.com/FedML-AI/FedML). -->

This repository is the official implementation of Federated Learning for Internet of Things: A Federated Learning Framework for On-device Anomaly Data Detection.

## 1. Introduction

Due to the heterogeneity, diversity, and personalization of IoT networks, Federated Learning (FL) has a promising future in the IoT cybersecurity field. As a result, we present the FedIoT, an open research platform and benchmark to facilitate FL research in the IoT field. In particular, we propose an autoencoder based trainer to IoT traffic data for anomaly detection. In addition, with the application of federated learning approach for aggregating, we propose an efficient and practical model for the anomaly detection in various types of devices, while preserving the data privacy for each device. What is more, our platform supports three diverse computing paradigms: 1) on-device training for IoT edge devices, 2) distributed computing, and 3) single-machine simulation to meet algorithmic and system-level research requirements under different system deployment scenarios. We hope FedIoT could provide an efficient and reproducible means for developing the implementation of FL in the IoT field. 

Check our slides here: https://docs.google.com/presentation/d/1_Y-oqLW8J0uZXRTX23DvuVJAr2TKM61thcXv69wCgXA/edit?usp=sharing

## 2. Installation

<!-- http://doc.fedml.ai/#/installation -->
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
conda create -n fediot python=3.7
conda activate fediot
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt 
git submodule add https://github.com/FedML-AI/FedML
cd FedML; git submodule init; git submodule update; cd ../;
```
For the FedML package installation, please check http://doc.fedml.ai/#/installation-distributed-computing


## 3. Data Preparation

We select the N_BaIoT data as our evaluation dataset. For the detailed of our data, please look at the `data_readme.md` in the data folder.

## 4. Code Structure of FedIoT

- `FedML`: a soft repository link generated using `git submodule add https://github.com/FedML-AI/FedML`.


- `data`: provide data downloading scripts and store the downloaded datasets.
Note that in `FedML/data`, there also exists datasets for research, but these datasets are used for evaluating federated optimizers (e.g., FedAvg) and platforms.
FedNLP supports more advanced datasets and models.

- `data_preprocessing`: data loaders, partition methods and utility functions

- `model`: IoT models. For example, VAE for outlier detection.

- `training`: please define your own `trainer.py` by inheriting the base class in `FedML/fedml-core/trainer/fedavg_trainer.py`.
Some tasks can share the same trainer.

- `experiments/distributed`: 
1. `experiments` is the entry point for training. It contains experiments in different platforms. We start from `distributed`.
2. Every experiment integrates FOUR building blocks `FedML` (federated optimizers), `data_preprocessing`, `model`, `trainer`.
3. To develop new experiments, please refer the code at `experiments/distributed/main_uci_vae.py`.

- `experiments/Raspberry Pi`: 
1. It is the code designed for the implementation on the Raspberry Pi 4b.
2. It contains two blocks, `main_uci_rp.py` should be implemented on the edge device and `app.py` should be implemented on the server.
3. For the more detailed running setup, please look at the `Resberry_Pi Readme.md` at the `experiments/Resberry_Pi`.

- `experiments/centralized`: 
1. please provide centralized training script in this directory. 
2. This is used to get the reference model accuracy for FL. 
3. You may need to accelerate your training through distributed training on multi-GPUs and multi-machines. Please refer the code at `experiments/centralized/ae_cen_glb_test.py`.

## 5. Results

Please read the experiment section in our paper.


## 6. Citation
Please cite our FedIoT and FedML paper if it helps your research.
You can describe us in your paper like this: "We develop our experiments based on FedIoT [1] and FedML [2]".

## 7. Contact

The corresponding author is:

Tuo Zhang
tuozhang@usc.edu

Chaoyang He
chaoyang.he@usc.edu
http://chaoyanghe.com

Special Thanks to Tianhao Ma!
