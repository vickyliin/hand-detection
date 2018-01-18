# hand-detection

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
