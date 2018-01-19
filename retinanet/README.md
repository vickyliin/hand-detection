# Egocentric Hand Detection

- [HTC introduction slides](https://www.csie.ntu.edu.tw/~yvchen/f106-adl/doc/HTCHand.pdf)
- **1/18** [Poster session](https://www.facebook.com/groups/1856571231300201/permalink/1891232677834056/)

## Dataset

- `DeepQ-Synth-Hand-0?`, synthetic data (from 3D model)
- `DeepQ-Vivepaper`, real data 

## Dependencies

With python3.6, 

```
pip install -r requirements3.txt
```

## Training

### Preprocess

With `DeepQ-Synth-Hand-0?` and `DeepQ-Vivepaper` in `[data-dir]`, 
```
python preprocess.py [data-dir]
```

Or to preprocess each subdir,
```
python preprocess.py - data/DeepQ-Vivepaper data/DeepQ-Synth-Hand-01 [...]
```

### Train

```
python train.py --help
```
