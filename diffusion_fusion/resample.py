from abc import ABC, abstractmethod

import numpy as np
import torch


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC): # 对应为一个时间步采样器的抽象父类；
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights() # 缩放过的形式了； 也就是原本可能是50步 但是现在我们从里面抽取了 10 步
        p = w / np.sum(w) # 转换为概率分布
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)# 按照整体的概率求取对应的时间步内容；
        indices = torch.from_numpy(indices_np).long().to(device)# 这是抽到的时间下表  0-T-1
        weights_np = 1 / (len(p) * p[indices_np]) # 特殊的矫正操作；
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights # 一次性采样得到batch_size 时间步和对应的权重内容；



class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])# 比方说是从 250中抽样得到的 100个 indices


    def weights(self):
        return self._weights
