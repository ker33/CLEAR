import math

class LossWeightScheduler:
    """
    动态调整Loss权重的调度器，帮助模型在训练不同阶段学习不同的东西
    """
    def __init__(self, initial_weights, total_steps=50000, warmup_steps=5000):
        self.initial_weights = initial_weights
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def get_weights(self, step):
        """
        step: 当前训练步数
        
        设计逻辑：
        - 早期 (warmup): 主要学习 causal loss，稳定LLM
        - 中期 (main): 逐步增加 contrast 和 ortho，学习解耦
        - 后期 (finetune): 高强度的 ortho + contrast，强化对比
        """
        if step < self.warmup_steps:
            # 预热阶段：主要依赖 causal loss
            progress = step / self.warmup_steps
            return {
                'causal': 1.0,
                'intervene': 0.3 + 0.5 * progress,
                'ortho': 0.1 + 0.2 * progress,
                'contrast': 0.05 + 0.15 * progress,
            }
        
        elif step < self.total_steps * 0.8:
            # 主训练阶段
            progress = (step - self.warmup_steps) / (self.total_steps * 0.8 - self.warmup_steps)
            return {
                'causal': max(0.7, 1.0 - 0.3 * progress),
                'intervene': 0.8 + 0.2 * progress,
                'ortho': 0.3 + 0.4 * progress,
                'contrast': 0.2 + 0.3 * progress,
            }
        
        else:
            # 后期微调阶段：强化对比学习
            return {
                'causal': 0.7,
                'intervene': 1.0,
                'ortho': 0.7,
                'contrast': 0.5,
            }
