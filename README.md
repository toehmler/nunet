# Brain Tumor Segmentation using a 2D UNet

This project demonstrates the implementation of a 2D UNet Convolution Neural Network to segment regions of High-Grade Glioma brain tumors. The model implemented is based on this [paper](https://arxiv.org/abs/1505.04597).  The model was trained using data from the 2015 MICCAI BRaTS Challenge. For more information please see the [dataset](#dataset) section.

## Background / Overview
### What is brain tumor segmentation?
* What is brain tumor segmentation?
	* Process of separating healthy / normal brain tissues from tumor
	* Difficult because of irregular form and confusing boundaries of tumors
	* Time consuming for a radiologist to manually segment 
	* Segmentation tasks are ripe for machine learning / CNNs
### Data structure
* Explanation of data structure

## Installation
### Requirements
* Python 3.7.9
* Tensorflow 2.4.0
* Keras 2.4.0
* CUDA 11.0
* scitkit-learn
* scikit-image
* SimpleITK
* tqdm


### Setup
1. Install [Tensorflow](tensorflow.org), it is recommended that you have version 2.4.0+
2. Install [Keras](keras.io), it is recommended  that you have version 2.4.0+
3. Install [CUDA](https://www.tensorflow.org/install/gpu) and configure GPU support for tensorflow. It is recommended that you have version 11.0+
4. Clone the project
```
git clone https://github.com/toehmler/nunet.git
```
5. Use pip to install all the dependencies
```
pip3 install -r requirements.txt
```
6. Download the data from the [MICCAI BRaTS Website](https://www.med.upenn.edu/cbica/brats2020/data.html) 
7. Set `path_to_data`  in `config.ini`  to be the full path to the downloaded dataset.
8. Run the preprocessing script to perform N4ITK bias correction and rename the patient directories
```
python3 process.py
```

## Quick Start
* Explanation of how to train
* Explanation of how to test
* Explanation how to test pre-trained models
* Explanation of how to predict specific patient

## Dataset
The MRI data used to train this model was provided by the [2015 MICCAI BRaTS Challenge](http://www.braintumorsegmentation.org) 


## Setup on VM

8vcpus
30gb memory
32gb storage
Ubuntu 18.04 LTS

Installing AWS command line tool (used to download data from S3 storage)
```
sudo apt update
sudo apt install python3-pip awscli
```
Configure AWS credentials and download data
Default reion name and default output format can be left blank
```
aws configure
aws s3 cp s3://midd-brats/brats_data brats_data --recursive
```
Set up configuration for python / tensorflow

​	

```
pip3 install scikit-learn scikit-image SimpleITK tqdm	
```

[unet_v0.2_pat206](./outputs/unet_v0.2_pat206.gif)





TODO
- Automatic vm startup and makefiles for compiling training and predicting
- Visualization of model











