"""训练全精度模型"""

import os, sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.model import (
    Model, ModelBN, ModelLinear, ModelShortCut, ModelBNNoReLU, ModelLayerNorm, 
)
from torchpruner.utils import random_seed
from torchpruner.pruner import LevelPruner, RandomPruner, BlockPruner
from torchpruner.scheduler import AGPScheduler


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', metavar='DIR', default='/tmp',
                    help='path to dataset')
    parser.add_argument('--pruner', default='block',
                    help="'random' or 'level' or 'block'")
    parser.add_argument('--agp-iteration', type=int, default=3, help='AGP pruner iteration, must > 1 (the sparsity of first iteration is 0.)')
    parser.add_argument('--agp-finetune-epoch', type=int, default=1, help='AGP finetune epochs after pruning.)')
    args = parser.parse_args()
    return args


def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))


def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))


def validate(model: nn.Module, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))


def prune(args, model, config_list, finetuner, evaluator, lr_scheduler=None):
    if args.pruner == 'random':
        pruner = RandomPruner(model, config_list)
    elif args.pruner == 'level':
        pruner = LevelPruner(model, config_list)
    elif args.pruner == 'block':
        pruner = BlockPruner(model, config_list)
    else:
        raise Exception("Unsupport pruner {}".format(args.pruner))

    scheduler = AGPScheduler(pruner, config_list, finetuner, evaluator, 
                            total_iteration=args.agp_iteration, finetune_epoch=args.agp_finetune_epoch, lr_scheduler=lr_scheduler)
    scheduler.compress()

    pruner.show_sparsity()
    return pruner


if __name__ == "__main__":
    random_seed(seed=42)
    args = _args()

    config_list = [
        {
            'sparsity': 0.5,
            'op_types': ['Linear', 'Conv2d'],
            'block_size': 4
        }, {
            'sparsity': 0.25,
            'op_types': ['Conv2d'],
            'op_names': ['block.0'],
            'block_size': 2
        }, {
            'exclude': True,
            'op_names': ['fc']
        }
    ]

    # parameters
    batch_size = 64
    test_batch_size = 64
    seed = 1
    finetune_epochs = 1
    lr = 0.01
    momentum = 0.5
    save_model_dir = 'examples/ckpt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dataset_dir, train=True, download=True, 
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dataset_dir, train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    # choose model
    # model = Model()
    model = ModelBN()
    # model = ModelBNNoReLU()
    # model = ModelLinear()
    # model = ModelShortCut()
    # model = ModelLayerNorm()

    model = model.to(device)
    state_dict = torch.load(os.path.join(save_model_dir, f'mnist_{model._get_name()}.pth'), map_location=device)
    model.load_state_dict(state_dict)    

    model.eval()
    full_inference(model, test_loader)  # 测试模型全精度的精度

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    finetuner = lambda epoch, model: train_one_epoch(model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch)
    evaluator = lambda model: validate(model=model, device=device, test_loader=test_loader)
    pruner = prune(args, model, config_list, finetuner, evaluator)

    # finetune
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(1, finetune_epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, epoch)
        validate(model, device, test_loader)

    print("After finetuning sparsity:")
    pruner.show_sparsity()
