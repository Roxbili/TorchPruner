import collections
from typing import List, Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
import pandas as pd
# set pandas to print all data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 180)    # 设置打印宽度(**重要**)

import logging
_logger = logging.getLogger(__name__)

from torchpruner.utils import get_module_by_name


weighted_modules = [
    'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'Linear', 'Bilinear',
    'PReLU',
    'Embedding', 'EmbeddingBag',
]


class LayerInfo:
    def __init__(self, name: str, module: nn.Module):
        self.module = module
        self.name = name
        self.type_ = type(module).__name__


def _setattr(model: Module, name: str, module: Module):
    parent_module, _ = get_module_by_name(model, name)
    if parent_module is not None:
        name_list = name.split(".")
        setattr(parent_module, name_list[-1], module)
    else:
        raise '{} not exist.'.format(name)

class Compressor:
    """
    The abstract base pytorch compressor.
    """

    def __init__(self, model: Optional[Module], config_list: Optional[List[Dict]]):
        """
        Parameters
        ----------
        model
            The model under compressed.
        config_list
            The config list used by compressor, usually specifies the 'op_types' or 'op_names' that want to compress.
        """
        self.is_wrapped = False
        if model is not None:
            self.reset(model=model, config_list=config_list)
        else:
            _logger.warning('This compressor is not set model and config_list, waiting for reset() or pass this to scheduler.')

    def reset(self, model: Module, config_list: List[Dict]):
        """
        Reset the compressor with model and config_list.

        Parameters
        ----------
        model
            The model under compressed.
        config_list
            The config list used by compressor, usually specifies the 'op_types' or 'op_names' that want to compress.
        """
        assert isinstance(model, Module), 'Only support compressing pytorch Module, but the type of model is {}.'.format(type(model))
        self.bound_model = model
        self.config_list = config_list

        self._unwrap_model()

        self._modules_to_compress = None
        self.modules_wrapper = collections.OrderedDict()
        for layer, config in self._detect_modules_to_compress():
            wrapper = self._wrap_modules(layer, config)
            self.modules_wrapper[layer.name] = wrapper

        self._wrap_model()

    def _detect_modules_to_compress(self) -> List[Tuple[LayerInfo, Dict]]:
        """
        Detect all modules should be compressed, and save the result in `self._modules_to_compress`.
        The model will be instrumented and user should never edit it after calling this method.
        """
        if self._modules_to_compress is None:
            self._modules_to_compress = []
            for name, module in self.bound_model.named_modules():
                if module == self.bound_model:
                    continue
                layer = LayerInfo(name, module)
                config = self._select_config(self.config_list, layer)
                if config is not None:
                    self._modules_to_compress.append((layer, config))
        return self._modules_to_compress

    def _select_config(self, config_list: Optional[List[Dict]], layer: Optional[LayerInfo]) -> Optional[Dict]:
        """
        Find the configuration for `layer` by parsing `self.config_list`.

        Parameters
        ----------
        layer
            The layer that need to check if has compression configuration.
            Or PrunerModuleWrapper.

        Returns
        -------
        Optional[Dict]
            The retrieved configuration for this layer, if None, this layer should not be compressed.
        """
        ret = None
        for config in config_list:
            config = config.copy()
            # expand config if key `default` is in config['op_types']
            if 'op_types' in config and 'default' in config['op_types']:
                expanded_op_types = []
                for op_type in config['op_types']:
                    if op_type == 'default':
                        expanded_op_types.extend(weighted_modules)
                    else:
                        expanded_op_types.append(op_type)
                config['op_types'] = expanded_op_types

            # check if condition is satisified
            if 'op_types' in config and layer.type_ not in config['op_types']:
                continue
            if 'op_names' in config:
                assert isinstance(config['op_names'], list), 'op_names must be a list of strings'
                skip_layer = True
                for name in config['op_names']:
                    if layer.name.startswith(name): # start with op_names
                        skip_layer = False
                        break
                if skip_layer:
                    continue

            ret = config
        if ret is None or 'exclude' in ret:
            return None
        return ret

    def get_modules_wrapper(self) -> Dict[str, Module]:
        """
        Returns
        -------
        OrderedDict[str, Module]
            An ordered dict, key is the name of the module, value is the wrapper of the module.
        """
        return self.modules_wrapper

    def _wrap_modules(self, layer: LayerInfo, config: Dict):
        """
        Create a wrapper module to replace the original one.

        Parameters
        ----------
        layer
            The layer to instrument the mask.
        config
            The configuration for generating the mask.
        """
        _logger.debug("Module detected to compress : %s.", layer.name)
        wrapper = PrunerModuleWrapper(layer.module, layer.name, layer.type_, config, self)
        assert hasattr(layer.module, 'weight'), "module %s does not have 'weight' attribute" % layer.name
        # move newly registered buffers to the same device of weight
        wrapper.to(layer.module.weight.device)
        return wrapper

    def _unwrap_model(self):
        """
        Unwrap all modules that needed to be compressed.
        """
        if self.is_wrapped:
            for _, wrapper in self.get_modules_wrapper().items():
                _setattr(self.bound_model, wrapper.name, wrapper.module)
            self.is_wrapped = False

    def _wrap_model(self):
        """
        Wrap all modules that needed to be compressed.
        """
        if not self.is_wrapped:
            for _, wrapper in reversed(self.get_modules_wrapper().items()):
                _setattr(self.bound_model, wrapper.name, wrapper)
            self.is_wrapped = True

    def load_masks_to_model(self, masks: Dict[str, Dict[str, Tensor]]):
        """
        Load an exist masks on the wrapper. You can train the model with an exist masks after load the masks.

        Parameters
        ----------
        masks
            The masks dict with format {'op_name': {'weight': mask, 'bias': mask}}.
        """
        wrappers = self.get_modules_wrapper()
        for name, layer_mask in masks.items():
            assert name in wrappers, '{} is not in wrappers of this pruner, can not apply the mask.'.format(name)
            if layer_mask.get('weight') is not None:
                assert hasattr(wrappers[name], 'weight_mask'), 'There is no attribute weight_mask in wrapper.'
                setattr(wrappers[name], 'weight_mask', layer_mask.get('weight'))
            if layer_mask.get('bias') is not None:
                assert hasattr(wrappers[name], 'bias_mask'), 'There is no attribute bias_mask in wrapper.'
                setattr(wrappers[name], 'bias_mask', layer_mask.get('bias'))

    def get_masks_from_model(self) -> Dict[str, Dict[str, Tensor]]:
        """
        Get masks from the model.

        Return
        ----------
        masks
            The masks dict with format {'op_name': {'weight': mask, 'bias': mask}}.
        """
        masks = {}
        wrappers = self.get_modules_wrapper()
        for name, wrapper in wrappers.items():
            masks[name] = {'weight': None, 'bias': None}
            if hasattr(wrapper, 'weight_mask') is not None:
                masks[name]['weight'] = getattr(wrapper, 'weight_mask')
            if hasattr(wrapper, 'bias_mask') is not None:
                masks[name]['bias'] = getattr(wrapper, 'bias_mask')
        return masks

    def compress(self) -> Module:
        """
        Compress the model with algorithm implemented by subclass.

        The model will be instrumented and user should never edit it after calling this method.
        `self._modules_to_compress` records all the to-be-compressed layers.

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        return self.bound_model

    def show_sparsity(self):
        """
        Show sparsity of the model, layer sparsity, total sparsity

        masks
            The masks dict with format {'op_name': {'weight': mask, 'bias': mask}}.
        """
        def calc_sparsity(t: torch.Tensor):
            return (1 - t.count_nonzero() / t.numel()).item()

        masks = self.get_masks_from_model()
        res = []    # [[name, weight_sparsity, bias_sparsity]]
        for name, value in masks.items():
            weight_sparsity = calc_sparsity(value['weight'])
            bias_sparsity = calc_sparsity(value['bias']) if value['bias'] != None else None
            res.append([name, value['weight'].shape, weight_sparsity, None if value['bias'] is None else value['bias'].shape, bias_sparsity])
        
        df = pd.DataFrame(res, columns=['op_name', 'weight_shape', 'weight_sparsity', 'bias_shape', 'bias_sparsity'])
        _logger.info(df)

        total_weight_sparsity = df['weight_sparsity'].mean()
        total_bias_sparsity = df['bias_sparsity'].mean()
        _logger.info(f'total weight sparsity: {total_weight_sparsity}, total bias sparsity: {total_bias_sparsity}')

    def _get_quantize_type(self, module_type):
        return {
            'Linear': 'per_tensor',
            'Conv2d': 'per_channel',
            'LayerNorm': 'per_tensor'  # 其实不准确，但是基本没有太大差别
        }[module_type]

    def _qparams_flash_memory(self, weight: torch.Tensor, quantize_type='per_channel'):
        """
        evaluate quantization parameters flash memory for tiny device

        input_zero_point: int8 + output_zero_point: int8 + self.M(multiplier: int32 + shift: int8)

        Args
        -------
        weight: the weight of the module
        quantize_type: 'per_channel' or 'per_tensor', the flash memory will take quantization parameters size into account

        return
        -------
        flash_memory: the number of Byte
        """
        # conv weight shape: (out_channel, in_channel, kernel_size_h, kernel_size_w)
        # conv with groups=in_channel=out_channel(depthwise): (in_channel, 1, kernel_size_h, kernel_size_w)
        # linear weight shape: (out_channel, in_channel)

        # scale = multiplier / 2**shift, len(scale) == out_channel in 'per_channel'
        # multiplier: int32, shift: int8
        input_output_flash_memory = 2     # zero_point  这里input、output全估稍微有点不准确，但是无伤大雅

        if quantize_type == 'per_channel':
            weight_flash_memory = (4 + 1) * weight.shape[0]
        elif quantize_type == 'per_tensor':
            weight_flash_memory = 4 + 1
        else:
            raise ValueError(f'Unsupported quantize type {quantize_type}, expected per_channel or per_tensor')
        return weight_flash_memory + input_output_flash_memory

    def _flash_memory_8bit_offset(self, sparsity, module: nn.Module, part_params_size: dict):
        """
        evaluate flash memory for tiny device

        Args
        -------
        sparsity: the sparsity of the module to the weight
        module: nn.Module to be counted
        part_params_size: a dict to save the parameters size for each part

        return
        -------
        flash_memory: the number of Byte
        """
        raise NotImplementedError('_flash_memory_8bit_offset should be implemented.') 
    
    def _flash_memory_multi_bit_offset(self, sparsity, module: nn.Module, part_params_size: dict):
        raise NotImplementedError('_flash_memory_multi_bit_offset should be implemented.') 

    def _weight_to_1d(self, weight: torch.Tensor, module_type: str):
        """
        According to module type, reshape weight to 1d as memory layerout in Tiny device
        """
        if module_type == 'Conv2d':
            tensor = weight.permute(0, 2, 3, 1).reshape(-1)
        elif module_type == 'Linear':
            tensor = weight.reshape(-1)
        else:
            raise ValueError(f'Unsupported module type: {module_type}')
        return tensor

    def _nonzero_offset_1d(self, tensor: torch.Tensor):
        """
        Calculate 1D tensor offset between nonzero values

        e.g.
        tensor [1 0 0 2 2]
        offset [0 2 0]
        """
        non_zero_index = torch.nonzero(tensor).reshape(-1)
        non_zero_index -= non_zero_index[0].clone() # 首位置0，所以所有元素坐标都偏移
        for i in range(non_zero_index.numel() - 1, 0, -1):  # 从后往前
            non_zero_index[i] = non_zero_index[i] - non_zero_index[i - 1] - 1
        return non_zero_index

    def parameters_size(self, eval_for_tiny=True, sparse_format=0) -> int:
        """
        网络参数大小统计，目前网络由Conv、Linear、BN组成，BN不剪枝，之后会融合，因此只需要统计Conv和Linear

        Args
        -------
        eval_for_tiny: if True, evaluate how many flash memory of tiny device the pruned model will take.
        sparse_format: 0-方案1 offset和value都使用8bit存储, 1-方案2 offset采用动态bit的方式进行存储
        
        Return
        -------
        params_size: the space the pruned model is used.(Byte)
        """
        # config
        _flash_memory = {
            0: self._flash_memory_8bit_offset,
            1: self._flash_memory_multi_bit_offset
        }[sparse_format]

        self.layer_parameters_size = {}
        params_size = 0
        skip_layer_name = None
        module_list = list(self.bound_model.named_modules())
        for idx, (name, module) in enumerate(module_list):
            layer = LayerInfo(name, module)
            part_params_size = {'quant_param_size': 0, 'sparse_encode_size': 0, 'weight_size': 0, 'bias_size': 0}

            if layer.name == skip_layer_name:   # 跳过被PruneModuleWrapper封装过且已经统计过的层
                continue
            elif layer.type_ == 'PrunerModuleWrapper':  # sparse layer
                # PrunerModuleWrapper下一层封装已经统计过了，这里记录名字，之后跳过
                skip_layer_name = layer.name + '.module'

                if not eval_for_tiny:
                    part_params_size['weight_size'] = int((1 - module.config['sparsity']) * module.module.weight.numel())  # 剪枝后非零元素个数，也是量化后的字节数
                else:
                    # quant size
                    quantize_type = self._get_quantize_type(type(module.module).__name__)
                    qparams_flash_memory = self._qparams_flash_memory(module.module.weight, quantize_type)
                    part_params_size['quant_param_size'] = qparams_flash_memory
                    # sparse_encode_size & weight_size
                    _flash_memory(module.config['sparsity'], module.module, part_params_size)
                # 这里默认bias未剪枝
                if hasattr(module.module, 'bias') and module.module.bias is not None:
                    part_params_size['bias_size'] = 4 * module.module.bias.numel()
                elif type(module.module).__name__ == 'Conv2d':   # bias False, 检查是否下面有BN层，算子融合后的整个算子是有bias的
                    for new_name, new_module in module_list[idx + 1:]:
                        new_layer = LayerInfo(new_name, new_module)
                        if new_layer.name == skip_layer_name:
                            continue
                        if new_layer.type_ == 'BatchNorm2d':
                            part_params_size['bias_size'] = 4 * module.module.out_channels
                        elif new_layer.type_ in ['Conv2d', 'Linear', 'LayerNorm']:
                            break
                sparse_size = part_params_size['weight_size'] + part_params_size['sparse_encode_size']
                if sparse_size > module.module.weight.numel():  # 稀疏后的大小大于dense，那么就不稀疏
                    part_params_size['sparse_encode_size'] = 0
                    part_params_size['weight_size'] = module.module.weight.numel()  # 按照dense推理
                self.layer_parameters_size.update({name: part_params_size})

            elif layer.type_ == 'Linear' or layer.type_ == 'Conv2d' or layer.type_ == 'LayerNorm':    # 说明未剪枝的Linear或者Conv2d层
                # quant size
                quantize_type = self._get_quantize_type(type(module).__name__)
                qparams_flash_memory = self._qparams_flash_memory(module.weight, quantize_type)
                part_params_size['quant_param_size'] = qparams_flash_memory
                # weight and bias size
                part_params_size['weight_size'] = module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    part_params_size['bias_size'] = 4 * module.bias.numel()
                elif type(module).__name__ == 'Conv2d':   # bias False, 检查是否下面有BN层，算子融合后的整个算子是有bias的
                    for new_name, new_module in module_list[idx + 1:]:
                        new_layer = LayerInfo(new_name, new_module)
                        if new_layer.name == skip_layer_name:
                            continue
                        if new_layer.type_ == 'BatchNorm2d':
                            part_params_size['bias_size'] = 4 * module.out_channels
                        elif new_layer.type_ in ['Conv2d', 'Linear', 'LayerNorm']:
                            break
                self.layer_parameters_size.update({name: part_params_size})
        params_size = sum([sum(part_params_size.values()) for part_params_size in self.layer_parameters_size.values()])
        return params_size
    
    def get_layer_parameters_size(self) -> dict:
        """Return parameters size of each layer

            {
                'layer_name': {
                    'quant_param_size' : size1,
                    'sparse_encode_size': size2,
                    'weight_size': size3    # sparse or dense
                }
            }
        """
        assert len(self.layer_parameters_size) != 0, \
            "parameters_size should be implemented before calling this function"
        return self.layer_parameters_size

    def debug_mask(self, masks):
        """
        Show mask shape if value is not None.
        """
        for key, value in masks.items():
            _logger.debug(f"{key}, weight, {value['weight'].shape}")
            try:
                _logger.debug(f"{key}, bias, {value['bias'].shape}")
            except:
                _logger.debug(f"{key}, bias, {value['bias']}")


class PrunerModuleWrapper(Module):
    def __init__(self, module: Module, module_name: str, module_type: str, config: Dict, pruner: Compressor):
        """
        Wrap a module to enable data parallel, forward method customization and buffer registeration.

        Parameters
        ----------
        module
            The module user wants to compress.
        config
            The configurations that users specify for compression.
        module_name
            The name of the module to compress, wrapper module shares same name.
        module_type
            The type of the module to compress
        pruner
            The pruner used to calculate mask.
        """
        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        self.type_ = module_type
        # config and pruner
        self.config = config
        self.pruner = pruner

        # register buffer for mask
        self.register_buffer("weight_mask", torch.ones(self.module.weight.shape))
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.register_buffer("bias_mask", torch.ones(self.module.bias.shape))
        else:
            self.register_buffer("bias_mask", None)
        
    def forward(self, *inputs):
        # apply mask to weight, bias
        self.module.weight.data = self.module.weight.data.mul_(self.weight_mask)
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            self.module.bias.data = self.module.bias.data.mul_(self.bias_mask)
        return self.module(*inputs)
