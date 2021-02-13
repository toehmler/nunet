
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

## Usage





TODO
- Fix issues with loading and saving models with custom metrics
- Data / training pipeline 
- Tensorboard integration
- Visualizations of predictions
- Model loader and different models?
- Automatic vm startup and makefiles for compiling training and predicting
- Visualization of model


## Notes
* Use config files for models
* Set current model by setting path to model or something
* Set name to save model under
* Set batch size and all that jazz
* When training, model and weights saved to a folder under the model's name
* Each folder contains an h5 of the model, weights, printout and training params
* When running train, this file is looked for and if found then training continued
* If not found then new model is compiled and folders and files created
* Each model will also have its own folder for outputs visuals 
* Can keep track of training runs and stuff in one centralized place
* Have model name AND version number (kept in seperate folders)
* Results from testing are also tracked in this same folder
* Generate predictions to seperate folder specified in config












