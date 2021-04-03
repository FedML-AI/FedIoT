# FedIoT: Federated Learning for Internet of Things
 
<!-- This is FedIoT, an application ecosystem for federated IoT based on FedML framework (https://github.com/FedML-AI/FedML). -->

FedIoT is a research-oriented benchmarking framework for advancing *federated learning* (FL) in *Internet of Things* (IoTs).
It uses FedML repository as the git submodule. In other words, FedIoT only focuses on advanced models and dataset, while FedML supports various
federated optimizers (e.g., FedAvg) and platforms (Distributed Computing, IoT/Mobile, Standalone).

## Installation
<!-- http://doc.fedml.ai/#/installation -->
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
conda create -n fediot python=3.7
conda activate fediot
# conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -n fediot
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt 
git submodule add https://github.com/FedML-AI/FedML
cd FedML; git submodule init; git submodule update; cd ../;
```



## Data Preparation

### Option 1: download our processed files from Amazon S3.
Download files for each dataset using this command. For example, we could download UCI-MLR by the following command:
```bash
cd data/UCI-MLR/
sh download.sh
cd ../..
```

## Experiments for Centralized Learning 

## Experiments for Federated Learning

## Update FedML Submodule

<!-- ### Update FedML Submodule 
This is only for internal contributors, can put this kind of info to a seperate readme file.
```
cd FedML
git checkout master && git pull
cd ..
git add FedML
git commit -m "updating submodule FedML to latest"
git push
```  -->

## Code Structure of FedIoT
<!-- Note: The code of FedIoT only uses `FedML/fedml_core` and `FedML/fedml_api`.
In near future, once FedML is stable, we will release it as a python package. 
At that time, we can install FedML package with pip or conda, without the need to use Git submodule. -->

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
3. To develop new experiments, please refer the code at `experiments/distributed/text-classification`.

- `experiments/centralized`: 
1. please provide centralized training script in this directory. 
2. This is used to get the reference model accuracy for FL. 
3. You may need to accelerate your training through distributed training on multi-GPUs and multi-machines. Please refer the code at `experiments/centralized/DDP_demo`.




## Citation
Please cite our FedIoT and FedML paper if it helps your research.
You can describe us in your paper like this: "We develop our experiments based on FedNLP [1] and FedML [2]".

 
