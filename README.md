
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
Enter AWS Access Key ID: AKIAIVH4JK53VC4FH32A
Secret Access Key: 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
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

### Model compilation







TODO
- Fix issues with loading and saving models with custom metrics
- Data / training pipeline 
- Tensorboard integration
- Visualizations of predictions
- Model loader and different models?
- Automatic vm startup and makefiles for compiling training and predicting
- Visualization of model
-









