# RUBi
An attempt at the NeurIPS Reproducibility Challenge 2019, reimplementing the RUBi paper: Reducing Unimodal Biases in Visual Question Answering

## Installation

To quickly deploy the master branch (antolu/master) with CPU Tensorflow / Caffe, run the following command in the linux shell (windows/mac not supported by the deploy script). The will ask for your sudo password to install and enable Docker. 

``` bash
curl -OL https://raw.githubusercontent.com/antolu/RUBi/master/deploy.sh && bash deploy.sh
```

For GPU Tensorflow and Caffe, use 

``` bash
curl -OL https://raw.githubusercontent.com/antolu/RUBi/master/deploy.sh && bash deploy.sh --runtime nvidia
```

The deploy script additionally supports the following flags:
* `-r` / `--repo` - the user to pull the repo from, default is antolu.
* `-b` / `--branch` - the branch to checkout, default is master. 
* `--gloud` - Use gcloud servers to download pretrained visual features, which is 40 times faster than getting them from standard imagecaption.
* `--datasets` - Get the datasets (VQA-CP v2 and VQA v2 as well as the COCO 2014 dataset). Requires roughly 100 GBs of free space. 

## Training

Use the following command to start training with mixed precision mode (on Nvidia Volta architechture or later) with the RUBi baseline model and the question-only branch.
```shell script
./main.py --train --rubi --dataset vqa-v2-cp --fp16 rubi
```

