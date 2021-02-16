# Brain Tumor Segmentation using a 2D UNet

This project demonstrates the implementation of a 2D UNet Convolution Neural Network to segment regions of High-Grade Glioma brain tumors. The model implemented is based on this [paper](https://arxiv.org/abs/1505.04597).  The model was trained using data from the 2015 MICCAI BRaTS Challenge. For more information please see the [dataset](#dataset) section.

![unet_v0.2_pat206](./outputs/unet_v0.2_pat206.gif)

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

* GNU Make
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
6. Download the dataset from the [MICCAI BRaTS Website](https://www.med.upenn.edu/cbica/brats2020/data.html) 
7. Run the preprocessing script to perform N4ITK bias correction and rename the patient directories. Bias correction is only performed on the t1 and t1c modalities. The corrected scans are saved with the `_n4` suffix. Set `n4itk=False` to skip bias correction. (Note: the current script expects `.mha` files)
```
python3 process.py <path_to_data> <n4itk=False>
```
8. Set `path_to_data`  in `config.ini`  to be the full path to the downloaded dataset.

## Usage 

The project relies on the values in `config.ini` to operate properly. Updating these values will change which model is being trained, tested or used for predictions. Additional parameters for training can be found here as well (batch size, validation split, etc.).  See [Configuration Options](#Configuration-Options) for a more detailed explanation of each option.

### Quick Start

To perform the entire pipeline on a new model from scratch (compile, train and test), perform the following steps:

1. Make sure `path_to_data` in `config.ini` is set as the full path to the data
2. Set `model_name` and `version_no` in `config.ini` to be the name and version number that new model willbe saved as. (Keeping the default will load a pre-trained model)
3. Update any of the training and testing options as needed
4. Run the pipeline using these options by simply running `make`. 

This will compile a new model and save a summary and its architecture in a new directory in `models/` under the name and version number given in the configuration file. The model will be trained and the results of the testing script will be saved as `.csv` in this directory. Due to the large file size of the MRI scans and limitations on RAM, training is done in groups. See [Training](#Training) for more information.

### Available Commands
* `make`: Runs the entire pipeline (see [Quick Start](#Quick-Start)). 
* `make train`: Runs the training script for the model specified in `config.ini`. Compiles a new model if one is not found using the given name and version number in `config.ini`, otherwise loads the model and resumes training. A training log is kept in the model's directory and training params are appended to this log on each run. See [Training](#Training) for more information.
* `make test`: Runs the test script, saving the result as a `.csv` in the model's directory. Assumes the model specified in `config.ini` can be found in `/models`. See [Testing](#Testing) for more information on the metrics used.
* `make predict`: Runs the prediction script, will show prompt for input of patient number to predict. Creates prediction images using t1c modality as background. Saves predictions as `.png` files and creates a `.gif` from these and saves these outputs in the directory specified as `image_out_path` in `config.ini`.

### Configuration Options

**General**
* `path_to_data` : Path to dataset in full, assumes patient directories have been renamed in the format `pat{x}`.
* `image_out_path` : Path to save prediction images in full.

**Model**
* `name` : Name of the current model being trained, tested or used for predictions. 
* `ver` : The version number of the current model being trained, tested, or used for predictions
If a model of a given name and version already exists when training, that model is loaded, otherwise a new model is compiled.

**Training**   
* `start` : The index of the patient to start training with
* `end` : The index of the patient to end training with
* `interval` : How many patients to train in each group (if not an even divisor of the total number the last group will be the size of the remainder)
* `epochs` : Number of epochs to train for *per group*
* `batch_size` : The batch size to train the model using *per group*   
* `validation_split` : The split to use to create the validation data (0-1), done on *per group* basis

**Testing**
* `start` : The start index of the patient to test on
* `end` : The end index of the patient to test on

## Dataset

The MRI data used to train this model was provided by the [2015 MICCAI BRaTS Challenge](http://www.braintumorsegmentation.org) 

## Training

## Testing

## Results

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

```
pip3 install scikit-learn scikit-image SimpleITK tqdm	
```
TODO
- Automatic vm startup and makefiles for compiling training and predicting
- Visualization of model











