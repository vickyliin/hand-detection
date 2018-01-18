# HTC Hand Detection 

## Skin color detector
- Run ``python hand_skin_detector.py``

## Data augmention

### Environments

Linux with python 3.5/3.6

### Install

```
pip install -r requirements.txt
```

And put data as

```
- CycleGAN.ipynb
- ...
- data
    - DeepQ-Vivepaper
    - DeepQ-Synth-Hand-01
    - DeepQ-Synth-Hand-02
```

### Run

Open juypter by `jupyter notebook` first.

### Image Flip
Open `ImageFlip.ipynb` in jupyter, run all blocks

### Generate Image for GAN Training
Open `GenImage.ipynb` in jupyter, run all blocks
Open `MaskComp.ipynb` in jupyter, run all blocks

### Domain Adaption
Open `CycleGAN-keras.ipynb` in jupyter.
Select type of training in first block, and then run all blocks.



## Using single shot multibox detector (SSD) model
- Go to ssd directory `cd ssd/`

- Testing
  - `./test.sh`
  
- Training
  - Prepare dataset for training by data augmentation
  - `DATASET_PATH` should contain DeepQ-Synth-Hand, DeepQ-Vivepaper, DeepQ-Flipped
  - `./preprocessing.sh DATASET_PATH`
  - `./train.sh`
  - `./export_graph.sh`

## Retinanet
Please refer to https://github.com/vickyliin/keras-retinanet
