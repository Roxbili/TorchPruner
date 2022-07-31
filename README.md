# TorchPruner
TorchPrune is designed to prune PyTorch model.

## Install
pip install -e .

## Usage
Demo can be found in [examples](./examples/)

### config_list
```python
config_list = [
    {
        'sparsity': 0.25,
        'op_types': ['Conv2d'],
        'op_names': ['block.0']
    }, {
        'sparsity': 0.5,
        'op_types': ['Linear', 'Conv2d'],
    }, {
        'exclude': True,
        'op_names': ['fc']
    }
]
```

- sparsity : This is to specify the sparsity for each layer in this config to be compressed.
- op_types : Operation types to be pruned.
- op_names : Operation names to be pruned.
- exclude : Set True then the layers setting by op_types and op_names will be excluded from pruning.

### one-shot
```python
pruner = {
    'random': RandomPruner(model, config_list),
    'level': LevelPruner(model, config_list),
    'block': BlockPruner(model, config_list, block_size=2)
}['level']

pruner.compress()
pruner.show_sparsity()
pruner.parameters_size()
```

### agp iteration
```python
pruner = {
    'random': RandomPruner(model, config_list),
    'level': LevelPruner(model, config_list),
    'block': BlockPruner(model, config_list, block_size=2)
}['level']

scheduler = AGPScheduler(pruner, config_list, finetuner, evaluator, 
                         total_iteration=args.agp_iteration, finetune_epoch=args.agp_finetune_epoch, lr_scheduler=None)
scheduler.compress()

pruner.show_sparsity()
pruner.parameters_size()
```

`fintuner` gets `epoch` and `model` as inputs, `evaluator` gets `model` as inputs.


## Feature
1. Support Pruner:
- RandomPruner: random pruning
- LevelPruner: magnitede pruning
- BlockPruner: block pruning, along in_channel axis
2. Support one-shot pruning and AGP iteration pruning
3. Support evaluating parameters size of sparse model, including quant size, sparse encoding size and parameters size.