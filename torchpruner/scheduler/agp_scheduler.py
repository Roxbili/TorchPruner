from typing import List
import copy
import logging
_logger = logging.getLogger(__name__)

from compression.pruning.pruner.compressor import Compressor

class AGPScheduler:
    def __init__(self, pruner: Compressor, config_list: List[dict], finetuner, evaluator, total_iteration, 
                    start_iteration=0, initial_sparsity=0., finetune_epoch=1, lr_scheduler=None):
        '''
            Args:
                pruner: pruner to prune weights of model
                config_list:
                    - sparsity: This is to specify the sparsity for each layer in this config to be compressed.
                    - op_types: Operation types to be pruned.
                    - op_names: Operation names to be pruned.
                    - exclude: Set True then the layers setting by op_types and op_names will be excluded from pruning.
                finetuner: input model
                evaluator: evaluate model
                total_iteration: how many pruning iterations to prune, not including the training step
                start_iteration: when to start pruning
                initial_sparsity: the initial sparsity
                finetune_epoch: how many epochs between pruning iterations
                lr_scheduler: lr_scheduler for finetuner, None means not to use lr_scheduler
        '''
        self.pruner = pruner
        self.config_list = config_list
        self.finetuner = finetuner
        self.evaluator = evaluator
        self.start_iteration = start_iteration
        self.total_iteration = total_iteration
        self.initial_sparsity = initial_sparsity
        self.finetune_epoch = finetune_epoch
        self.lr_scheduler = lr_scheduler

    def step(self, current_iteration) -> List[dict]:
        current_config_list = copy.deepcopy(self.config_list)
        for i in range(len(self.config_list)):
            if self.config_list[i].get('sparsity') is None: # 没有sparsity则直接跳过
                continue

            final_sparsity = self.config_list[i]['sparsity']

            assert final_sparsity >= self.initial_sparsity
            if current_iteration < self.start_iteration:
                return 0.
            
            span = self.total_iteration - self.start_iteration - 1
            target_sparsity = (final_sparsity +
                                (self.initial_sparsity - final_sparsity) *
                                (1.0 - ((current_iteration - self.start_iteration) / span))**3)
            
            current_config_list[i]['sparsity'] = target_sparsity
        return current_config_list
    
    def compress(self):
        for current_iteration in range(self.total_iteration):
            current_config_list = self.step(current_iteration)
            _logger.info(current_config_list)   # 打印当前配置

            model = self.pruner(current_config_list)
            # self.pruner.show_sparsity()     # 打印稀疏度

            for epoch in range(self.finetune_epoch):
                current_epoch = current_iteration * self.finetune_epoch + epoch
                # finetune的时候，前传需要考虑mask，否则剪枝就没有意义(采用PrunerModuleWrapper封装需要剪枝的层)
                self.finetuner(epoch=current_epoch, model=model)

                eval_metrics = self.evaluator(model=model)
                _logger.info(
                    'Loss: {loss:>6.4f}  '
                    'Acc@1: {top1:>7.4f}  '
                    'Acc@5: {top5:>7.4f}\n'.format(
                        loss=eval_metrics['loss'], top1=eval_metrics['top1'], top5=eval_metrics['top5']))

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(current_epoch, eval_metrics['top1'])
        return eval_metrics

    def get_best_result(self):
        return self.pruner.bound_model, self.pruner.get_masks_from_model()
