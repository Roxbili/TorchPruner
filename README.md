# TorchPruner
TorchPrune is designed to prune PyTorch model.

## Install
pip install -e .

## Usage

### RandomPruner
```python
```

## Feature
1. Support one-shot pruning and AGP iteration pruning
2. Support Pruner:
    - RandomPruner: random pruning
    - LevelPruner: magnitede pruning
    - BlockPruner: block pruning, along in_channel axis