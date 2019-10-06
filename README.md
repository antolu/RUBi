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

