# Seeing more with less: Virtual Labelling 

# Overview

Standard immunofluorescence methods capture just ~4 molecular markers per cell, limiting interrogation of complex cell biology. To overcome this barrier without recourse to relatively inaccessible experimental multiplexing techniques, we ‘computationally multiplex’ an unrestricted marker number per cell, using ResViT. Computational multiplexing thus shifts the paradigm for virtual labelling applications from label-replacement to data-enrichment; seeing more (multi-molecular single-cell biology) with less (standard 4-plex immunofluorescence data). This accessible approach democratises spatially resolved, imaging-based, multi-molecular analyses for any practitioner of standard immunofluorescence.

# Installation

## Hardware Requirements

The package requires a CUDA enabled GPU to run. We suggest a computer with the minimum specs: <br />
RAM: 16+ GB  <br />
CPU: 4+ cores, 3.3+ GHz/core<br />
CUDA GPU: 16+ GB VRAM 

## Software Requirements

Users should install the following packages in a python environment (3.10)
```
torch>=2.2.1
torchvision>=0.17.1
scikit-image
scipy
ml_collections
cuda>=11.2
```

- Download or clone this repo. e.g.
```bash
git clone https://github.com/CancerSystemsMicroscopyLab/VirtualLabelling
```

Installation time ~20 mins on a typical computer with standard internet connection

# Virtual Labelling

## Preprocessing
To use the model as described in the paper, images need to be in 8bit depth with 256x256 resolution.
We have provided a function to make the bit depth conversion called 'convert_folder_to_8bits' in 16bits_convert_to_8bits.py
Label-free background removal can also be done using imagej macro provided in the preprocessing folder

To use the dataloader we have provided, images should be separated by marker and placed in a folder, inside a parent folder and have matching corresponding names, as done in the 'data' folder example we have provided. E.g.

```
Parent
│
└───DAPI
│   │   img1.tif
│   │   img2.tif
│   │   ...
│   
└───GM130
    │   img1.tif
    │   img2.tif
    │   ...
  
```

## Training and applying the model to label 
You will first need to download pre-trained ViT model from Google and place it at - ./model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz

We have provided a simple example script showing how to train a model and apply it to virtually label unseen fields (run_vl.ipynb). The script runs on sample images we have provided in the ./data_example folder.
To modify this script to use with your own images, simply modify the dataroot variable to point to the directory containing your images and change the input and targets to be the markers intended (matching how the folders are named).

Run time can vary depending on hardware and dataset size. ~1hrs-4hrs runtime might be expected. 

Alternatively, we have also provided a script - 'VirtualLabelling_colab_example.ipynb' to run virtual labelling on Google Colab (avoiding any necessary set up). Be ensure a GPU runtime is selected. Data needs to be uploaded to the runtime with modalities placed in separate folders as described. 

## Results
Results can be found in ./results folder. In this folder, the art_pretrain folder contains the model without the trained transformer. 
The resvit folder contains the fully trained model and predictions from the test set in the test_images folder. Images are given a suffix to denote whether they were inputs, predictions or the target. 
The results.csv file contains measured performance metrics for each image in the test set. A list of images in training and test sets are also given.

In the resvit folder, the predictions folder contains prediction for all input image sets, including those without a corresponding target image.
In our toy example, we have provided 10 sets of inputs and 9 target Fibrillarin images. The trained model will finally virtually label all 10 sets (with the limited training provided).

## Downstream processing
Virtual labels can be analysed in the same way as other fluorescence images as needed. 

Results published in "TB"C can be reproduced using the accompanying quantitative data found at tbc.


# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
TBC
```

For any questions, comments and contributions, please contact Dr John Lock (john.lock@unsw.edu.au) <br />

(c) Cancer Systems Microscopy Lab 2024

## Acknowledgments
This code uses libraries from [ResViT](https://github.com/icon-lab/ResViT), and [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.
