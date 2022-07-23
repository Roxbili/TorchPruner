from typing import List, Dict, Optional, Tuple, Any
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import logging
_logger = logging.getLogger(__name__)

from .compressor import Compressor, PrunerModuleWrapper
from tools import LayerInfo

class BlockPruner(Compressor):
    def __call__(self, current_config_list: Optional[List[Dict]]):
        return self.compress(current_config_list)

    def __init__(self, model: Optional[Module], config_list: Optional[List[Dict]], block_size: int):
        """
        Parameters
        ----------
        model
            The model under compressed.
        config_list
            The config list used by compressor, usually specifies the 'op_types' or 'op_names' that want to compress.
        block_size
            The block window, Linear: input_feature direction, Conv: channel direction
        """
        self.block_size = block_size
        super().__init__(model, config_list)
    
    def _generate_block_size(self, module: Module) -> Tuple:
        """
        根据Module类型生成block_size
        """
        module_name = module._get_name()
        if module_name == 'Linear':
            block_size = (1, self.block_size)   # in_feature方向上
        elif module_name == 'Conv2d':
            # conv在in_channel方向上
            # depthwise conv在out_channel方向上
            if module.groups == module.weight.shape[0]:    # depthwise
                block_size = (self.block_size, 1, 1, 1)
            elif module.groups == 1:    # 普通conv
                block_size = (1, self.block_size, 1, 1)
            else:
                raise ValueError('Unsupported group convolution')
        else:
            raise ValueError(f'Invalid op_types: {module_name}')
        return block_size
    
    def _block_avg_2d(self, data, block_size) -> torch.Tensor:
        """
        二维block平均，不够block_size步长的时候会取剩下的并做平均
        """
        # torch pool
        metric = F.avg_pool2d(data.abs().unsqueeze(0), kernel_size=block_size,
                        ceil_mode=True, count_include_pad=False).squeeze(0)

        # 手写的，效率低
        # metric = []
        # for i in range(0, data.shape[0], block_size[0]):
        #     line_metrics = []
        #     for j in range(0, data.shape[1], block_size[1]):
        #         block_mean = data[i:i+block_size[0], j:j+block_size[1]].mean()
        #         line_metrics.append(block_mean)
        #     metric.append(line_metrics)
        # metric = torch.from_numpy(np.array(metric, dtype=np.float32))

        return metric.to(data.device)

    def _block_avg_4d(self, data, block_size) -> torch.Tensor:
        """
        四维block平均，不够block_size步长的时候会取剩下的并做平均
        """
        # 寻找非1的维度
        block_size = np.array(block_size)
        non_one_dim = np.argwhere(block_size != 1)
        if len(non_one_dim) == 1:   # 只有一个维度上需要压缩
            dim = non_one_dim[0][0]
            metric = None
            for start in range(0, data.shape[dim], block_size[dim]):
                end = min(start + block_size[dim], data.shape[dim])
                selected_data = data.index_select(dim=dim, index=torch.arange(start, end, device=data.device))
                selected_data = selected_data.abs().mean(dim=dim, keepdims=True)
                
                metric = torch.cat((metric, selected_data), dim=dim) if metric is not None else selected_data

        else:   # 多个维度上都需要压缩
            metric = []
            for i in range(0, data.shape[0], block_size[0]):
                m1 = []
                for j in range(0, data.shape[1], block_size[1]):
                    m2 = []
                    for k in range(0, data.shape[2], block_size[2]):
                        m3 = []
                        for l in range(0, data.shape[3], block_size[3]):
                            block_mean = data[i:i+block_size[0], j:j+block_size[1], k:k+block_size[2], l:l+block_size[3]].abs().mean()
                            m3.append(block_mean.cpu())
                        m2.append(m3)
                    m1.append(m2)
                metric.append(m1)
            metric = torch.from_numpy(np.array(metric, dtype=np.float32))
        return metric.to(data.device)

    def _compress_mask(self, mask, block_size) -> torch.Tensor:
        """
        依照block_size压缩mask
        """
        if len(mask.shape) == 2:
            return self._block_avg_2d(mask, block_size)
        elif len(mask.shape) == 4:
            return self._block_avg_4d(mask, block_size)
        else:
            raise ValueError('Invalid mask shape, expected 2d or 4d')

    def _get_metric(self, wrapper: PrunerModuleWrapper, block_size: Tuple) -> torch.Tensor:
        """
        按照block_size把weight进行avg_pool，bias暂时不剪枝;
        Linear在输入特征的方向做均值，Conv在通道方向做均值

        Return
        ----------
        metric: 返回评估指标，即 block_reduce 后又做了上一轮的mask
        """
        module = wrapper.module
        module_name = module._get_name()
        metric = getattr(module, 'weight').clone().cpu().detach()

        if module_name == 'Linear':
            # unsqueeze 和 squeeze 的目的是加入batch维度，否则pool无法调用
            _logger.debug(f'{wrapper.name} weight shape: {module.weight.data.shape}')   # (out_feature, in_feature)
            metric = self._block_avg_2d(metric, block_size)
            _logger.debug(f'{wrapper.name} metric shape: {metric.shape}')

        elif module_name == 'Conv2d':
            _logger.debug(f'{wrapper.name} weight shape: {module.weight.data.shape}')   # (out_channel, in_channel, kernel_w, kernel_h)
            metric = self._block_avg_4d(metric, block_size)
            _logger.debug(f'{wrapper.name} metric shape: {metric.shape}')
        else:
            raise ValueError(f'Invalid op_types: {module_name}')
        return metric.to(module.weight.device)

    def _expand_mask(self, mask: torch.Tensor, block_size: Tuple, target_shape) -> torch.Tensor:
        """
        根据block_size扩展mask
        """
        expand_size = list(mask.size())
        reshape_size = list(mask.size())
        for i, block_width in reversed(list(enumerate(block_size))):
            mask = mask.unsqueeze(i + 1)
            expand_size.insert(i + 1, block_width)
            reshape_size[i] *= block_width
        mask = mask.expand(expand_size).reshape(reshape_size)

        # 裁剪mask至target_shape
        for i, target_shape_i in enumerate(target_shape):
            mask = torch.index_select(mask, i, torch.arange(target_shape_i).to(mask.device))
        assert mask.shape == target_shape, f'mask shape: {mask.shape}, target shape: {target_shape}'
        return mask.to(mask.device)

    def prune(self, wrapper: PrunerModuleWrapper, mask: Dict[str, torch.Tensor], sparsity):
        if sparsity == 0:   # 不需要进行剪枝
            return mask

        module = wrapper.module
        mask_new = {}
        # 根据module获得block_size
        block_size = self._generate_block_size(module)

        # 获得metric作为评估
        metric = self._get_metric(wrapper, block_size)   # 获得本轮metric
        metric *= self._compress_mask(mask['weight'], block_size)   # 导入上一轮的mask
        
        # 计算需要剪枝的block个数
        prune_num = int(sparsity * metric.numel())
        _logger.debug(f"{wrapper.name} sparsity: {sparsity}, metric_num: {metric.numel()}, prune_num: {prune_num}")

        if prune_num == 0:  # 也就是说本次无需剪枝
            mask_new['weight'] = mask['weight']
        else:
            # 剪枝
            threshold = torch.topk(metric.view(-1), prune_num, largest=False)[0].max()
            mask_new['weight'] = torch.gt(metric, threshold).type_as(mask['weight'])
            mask_new['weight'] = self._expand_mask(mask_new['weight'], block_size, target_shape=module.weight.shape)    # 扩展mask
            mask_new['weight'] = mask_new['weight'].to(mask['weight'].device)

        # bias mask直接复制
        mask_new['bias'] = mask['bias']
        return mask_new

    def compress(self, current_config_list: Optional[List[Dict]]):
        masks = self.get_masks_from_model()
        # self.debug_mask(masks)

        masks_new = {}
        for name, wrapper in self.get_modules_wrapper().items():
            config = self._select_config(current_config_list, wrapper)
            if config is not None:
                sparsity = config['sparsity']
                mask = self.prune(wrapper, masks[name], sparsity)
                masks_new[name] = mask

        self.load_masks_to_model(masks_new)
        return self.bound_model

    def _flash_memory_8bit_offset(self, sparsity, module: nn.Module):
        weight = module.weight
        bias = module.bias
        quantize_type = self._get_quantize_type(type(module).__name__)

        qparams_flash_memory = self._qparams_flash_memory(weight, quantize_type)
        # (1 + 1 / self.block_size) 是考虑块内接下来的元素都是非零，因此列号只需要block首地址，剩下的当作是block_size连续
        params_flash_memory = (1 + 1 / self.block_size) * int(weight.numel() * (1 - sparsity))
        
        # bias默认不剪枝，因此注释
        # if bias is not None:
        #     params_flash_memory += (4 + 1 / self.block_size) * int(bias.numel() * (1 - sparsity))     # bias has 32 bit = 4 bytes, 1 / self.block_size for index
        return qparams_flash_memory + params_flash_memory

    def _cal_multi_bit_offset_memory(self, tensor):
        """
        calculate offset size(byte)
        """
        non_zero_offset = self._nonzero_offset_1d(tensor)[::self.block_size]
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

    def _flash_memory_multi_bit_offset(self, sparsity, module: nn.Module):
        weight = module.weight
        bias = module.bias
        quantize_type = self._get_quantize_type(type(module).__name__)

        qparams_flash_memory = self._qparams_flash_memory(weight, quantize_type)
        params_flash_memory = (1 + 0.25 / self.block_size) * int(weight.numel() * (1 - sparsity)) # weight + type
        offset_memory = self._cal_multi_bit_offset_memory(self._weight_to_1d(weight, type(module).__name__))
        params_flash_memory += offset_memory / 8 # bit -> byte
        return math.ceil(qparams_flash_memory + params_flash_memory)