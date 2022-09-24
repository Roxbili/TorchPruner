from typing import List, Dict, Optional, Tuple, Any
import math
import torch
import torch.nn as nn
from torch.nn import Module
import logging
import numpy as np
from torchpruner.pruner.compressor import Compressor
from torchpruner.pruner.registry import Pruner

_logger = logging.getLogger(__name__)


@Pruner.register
class LevelPruner(Compressor):
    def prune(self, module: Module, mask, sparsity, prune_bias=False):
        mask_new = {}
        prune_name_list = ['weight', 'bias'] if prune_bias else ['weight']
        for name in prune_name_list:
            if not (hasattr(module, name) and getattr(module, name) is not None):
                mask_new[name] = None
                continue
            param = getattr(module, name).clone().cpu().detach()
            param = torch.where(mask[name].cpu() == 1, param, torch.zeros(param.size()))
            param = torch.abs(param)
            sort_param, indices = torch.sort(param.view(-1), descending=False)
            threshold = sort_param[int(sort_param.size(0) * sparsity)]
            mask_new[name] = param.ge(threshold).detach().type_as(mask[name])

        if not prune_bias:
            # 不剪枝bias的话bias mask直接复制
            mask_new['bias'] = mask['bias']

        return mask_new

    def compress(self, current_config_list: Optional[List[Dict]] = None):
        if current_config_list is None:
            current_config_list = self.config_list

        masks = self.get_masks_from_model()
        # self.debug_mask(masks)

        masks_new = {}
        for name, wrapper in self.get_modules_wrapper().items():
            config = self._select_config(current_config_list, wrapper)
            if config is not None:
                sparsity = config['sparsity']
                mask = self.prune(wrapper.module, masks[name], sparsity)
                masks_new[name] = mask

        self.load_masks_to_model(masks_new)
        return self.bound_model

    def _flash_memory_8bit_offset(self, config, module: nn.Module, part_params_size: dict):
        weight = module.weight
        bias = module.bias
        sparsity = config['sparsity']
        weight_size = int(weight.numel() * (1 - sparsity))
        sparse_encode_size = int(weight.numel() * (1 - sparsity))
        params_flash_memory = weight_size + sparse_encode_size
        part_params_size['weight_size'] = weight_size
        part_params_size['sparse_encode_size'] = sparse_encode_size

        # bias默认不剪枝，因此注释
        # if bias is not None:
        #     params_flash_memory += (4 + 1) * int(bias.numel() * (1 - sparsity))     # bias has 32 bit = 4 bytes, 1 for index
        return params_flash_memory

    def _cal_multi_bit_offset_memory(self, tensor):
        """
        calculate offset size(byte)
        """
        non_zero_offset = self._nonzero_offset_1d(tensor)
        memory = 0
        for offset in non_zero_offset:
            if offset < 2**2:
                memory += 2
            elif offset < 2**4:
                memory += 4
            elif offset < 2**8:
                memory += 8
            elif offset < 2**16:
                memory += 16
            else:
                raise ValueError(f'Invalid offset: {offset}')
        return memory

    def _flash_memory_multi_bit_offset(self, config, module: nn.Module, part_params_size: dict):
        weight = module.weight
        bias = module.bias
        sparsity = config['sparsity']
        quantize_type = self._get_quantize_type(type(module).__name__)

        qparams_flash_memory = self._qparams_flash_memory(weight, quantize_type)
        params_flash_memory = (1 + 0.25) * int(weight.numel() * (1 - sparsity)) # weight + type
        offset_memory = self._cal_multi_bit_offset_memory(self._weight_to_1d(weight, type(module).__name__))
        params_flash_memory += offset_memory / 8 # bit -> byte
        return math.ceil(qparams_flash_memory + params_flash_memory)
