# Active Label Cleaning Repository
 
#### Table of contents
* Installation
* Datasets
* How to train models
* How to run the main label cleaning simulation
* Code for additional paper experiments

## Installation:

Cloning the repository to the local disk
```
git clone ANONYMIZED_URL
```

Switch directory to place yourself at the root of the repository and set up the python environment
```
conda update -n base -c defaults conda pytorch
python create_environment.py
conda activate DataQuality
pip install -e .
```

Linters:

Please make sure that `Autopep8` and `Flake8` are activated in your IDE (e.g. VS-Code or Pycharm)<br />
<br />

## Datasets
### CIFAR10H
###### About
The CIFAR10H dataset is the CIFAR10 test set but all the samples have been labelled by multiple annotators.
We use the CIFAR training set for validation.
###### How to use it
The dataset will automatically be downloaded to your machine when you run code for the first time
on your machine with the [cifar10h dataset class](DataQuality/datasets/cifar10h.py) 
(or corresponding [PL module](DataQuality/deep_learning/self_supervised/cifar10h_datamodule.py)).

### Chest X-ray datasets
#### Full Kaggle Pneumonia Detection challenge dataset
###### About
For our experiments, in particular for unsupervised pretraining we use the full Kaggle training set (stage 1) from the
[Pneumonia Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). The dataset class for this dataset
can be found in the [kaggle_cxr.py](DataQuality/datasets/kaggle_cxr.py) file. This dataset class loads the full 
set with binary labels based on the bounding boxes provided for the competition.

###### How to use it 
In order to use this dataset, you will have to first download the dataset to your machine. 
The dataset can be downloaded from [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) You will need to update your run configs to point `dataset_dir` field
to the location of the dataset on your machine.

#### Noisy Chest-Xray
The images released as part of the Kaggle Challenge, where originally released as part of the NIH chest x-ray datasets. 
Before starting the competition, 30k images have been selected as the images for competitions. The labels for these images
have then been adjudicated to label them with bounding boxes indicating "pneumonia-life opacities". In order to evaluate 
our label cleaning framework on medical dataset, we have sampled a small subset of the Kaggle dataset (4000 samples, balanced) 
for which we have access to the original labels provided in the NIH dataset. This dataset uses the kaggle dataset with noisy labels
as the original labels from RSNA and the clean labels are the Kaggle labels. Originally the dataset had 14 classes, we 
created a new binary label to label each image as "pneumonia-like" or "non-pneumonia-like" depending on the original label
prior to adjudication. The original (binarized) labels along with their corresponding adjudicated label, can be found in
the [noisy_chestxray_dataset.csv](DataQuality/datasets/noisy_chestxray_dataset.csv) file. The dataset class for this dataset
is the [rsna_cxr.py](DataQuality/datasets/rsna_cxr.py) file. This dataset class will automatically load the labels
from the aforementioned file.

###### How to use it 
The code will assume that the Kaggle dataset is present on your machine (see dataset above for instructions) and that 
your config points to the correct `dataset_dir` location. 

## How to train models
### Supervised models
#### General
The main entry point for training a supervised model (vanilla or coteaching) is [train.py](DataQuality/deep_learning/train.py). 
The code requires you to provide a config file specifying the dataset to use, the training specification (batch size, scheduler etc...),
whether to use vanilla or coteaching training, which augmentation to use ...

Train the model using the command
```python
python DataQuality/deep_learning/train.py --config DataQuality/configs/models/YOUR_CONFIG_CHOICE
```

For each of the dataset used in our experiments, we have defined a config to run training easily off the shelf.

#### CIFAR10H
In order to run: 
* vanilla resnet training please use the [DataQuality/configs/models/cifar10h/resnet.yaml](DataQuality/configs/models/cifar10h/resnet.yaml) config.
* co-teaching resnet training:  [DataQuality/configs/models/cifar10h/resnet_co_teaching.yaml](DataQuality/configs/models/cifar10h/resnet_co_teaching.yaml) config
 
#### Noisy Chest-Xray
To run any model on this dataset, you will need to first make sure you have the dataset uploaded onto your machine (see dataset section).
In order to run:
* Vanilla training on the dataset with 13% noise please use the [DataQuality/configs/models/rsna/resnet.yaml](DataQuality/configs/models/rsna/resnet.yaml) config.
* co-teaching densenet121 training:  [DataQuality/configs/models/rsna/resnet_coteaching.yaml](DataQuality/configs/models/rsna/resnet_coteaching.yaml) config
 
### How to pretrain embeddings with an unsupervised model
#### General
For the unsupervised training of our models, we rely on PyTorch Lightning and Pytorch Lightining bolts. The main entry point
for model training is [DataQuality/deep_learning/self_supervised/main.py](DataQuality/deep_learning/self_supervised/main.py).
You will also need to feed in a config file to specify which dataset to use etc.. 
Command to use run 
```python
DataQuality/deep_learning/self_supervised/main.py --config DataQuality/deep_learning/self_supervised/configs/YOUR_CONFIG_CHOICE
```
#### CIFAR10H
To train embeddings with contrastive learning on CIFAR10H use the 
[DataQuality/deep_learning/self_supervised/configs/cifar10h_byol.yaml](DataQuality/deep_learning/self_supervised/configs/cifar10h_byol.yaml)
config. 

#### Noisy Chest X-ray 
For unsupervised pretraining we use the full NIH dataset 
To train a model on this dataset, please use the [DataQuality/deep_learning/self_supervised/configs/nih_byol.yaml](DataQuality/deep_learning/self_supervised/configs/nih_byol.yaml)
config.

### How to finetune a model based on SSL-pretrained embeddings
#### General
After having trained your unsupervised models, you can learn a simple linear head of top of the (frozen) embeddings. For
this you can again use [train.py](DataQuality/deep_learning/train.py), simply updating your config to specify 
the location of the trained encoder's checkpoint in the `train.self_supervision.checkpoints` field.
#### CIFAR10H
To finetune a linear head on CIFAR10H use the 
[DataQuality/configs/models/cifar10h/resnet_self_supervision.yaml](DataQuality/configs/models/cifar10h/resnet_self_supervision.yaml)
config. 

#### Noisy Chest-Xray
To finetune a linear head on the full Kaggle set use the [DataQuality/configs/models/rsna/self_supervised_nih.yaml](DataQuality/configs/models/rsna/self_supervised_nih.yaml)
config.

To train a co-teaching model from a pretrained set of SSL weights, you can use [DataQuality/configs/models/rsna/resnet_pretrained_coteaching.yaml](DataQuality/configs/models/rsna/resnet_pretrained_coteaching.yaml)
config.

## How to run the main label cleaning simulation
To run the label cleaning simulation you will need to run [main_simulation](DataQuality/main_simulation.py) with
a list of configs in the `--config` arguments as well as a list of seeds to use for sampling in the `--seeds` arguments.

```
python DataQuality/main_simulation.py --config DataQuality/configs/selection/YOU_CONFIG_CHOICE1 DataQuality/configs/selection/YOU_CONFIG_CHOICE2 --seeds 1 2 3
```
You will need to provide a list of `selector_configs` as config arguments. A selector config will allow you to specify which
selector to use and which model config to use for inference. All selectors config can be found in the 
[configs/selection](DataQuality/configs/selection) folder. 

## Code for additional paper experiments
We also provide the code used to run additional experiments in the paper. In particular, we provide the code for the model selection
section in  [DataQuality/model_selection_benchmark.py](DataQuality/model_selection_benchmark.py)