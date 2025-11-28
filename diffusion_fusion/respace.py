import numpy as np
import torch

from .gaussian_diffusion import GaussianDiffusion

''' 对应为稀疏的时间步内容； '''


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up. 也即是250
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. 
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        # 首先切分为相同的数量的块，然后在块中等比例抽样对应的内容；
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)  # 多余的内容；
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)  # 将extra 部分的也是带上了的；
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )  # 要是比需要抽取的要少就直接报错处理
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
        # 本质上是为了去除重复的操作 所以转换为set
    return set(all_steps)  # 转换为set 集合的形式；
# python 中的set并不是有序的 ； 原本构建的list 中的内容将会被打乱；


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    '''
    use_timesteps=space_timesteps(steps, timestep_respacing),# 总的数量为从0开始的 后面为分块和块中取到的位置数量的内容了；
        betas=betas,# 从其中等间隔抽取到指定的内容； 
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,# 是否将参数重新映射到 0-1000 之间 
    '''

    def __init__(self, use_timesteps, **kwargs):
        '''
        :param use_timesteps: 使用的时间步的集合内容 从0->T-1 抽样得到的这些内容；
        betas 生成的噪声水平参数；
        model_mean_type
        model_var_type
        loss_type
        rescale_timesteps 是否将时间步重新映射到 0-1000 之间的内容；
        转换为使用统一的时间尺度内容；

        '''
        # 从0-250 之间抽取选择的内容；
        self.use_timesteps = set(use_timesteps)  # 对应为需要使用的use_timesteps
        self.timestep_map = []  # 记录新的时间轴对应的时间步的编号内容；
        self.original_num_steps = len(kwargs["betas"])  # 总的长度；250
        # 原始的采样的步数内容；betas 的长度和step 为一样的是 250 的

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                # 新的beta 需要重建出来原始的diffusion 递增的过程方可；
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)  # 新的被纳入的时间步 i 位置；
        kwargs["betas"] = np.array(new_betas)
        # self.time_step 从0-250之中抽样的标签位置情况；
        super().__init__(**kwargs) #  num_timesteps 为 250-100 也就是其中的100的内容了；

    # orginal_num_steps  就是按照给定的 x0 内容计算相应后验概率的分布的情况内容；
    def p_mean_variance_org(
            self, diffusion_stage1, diffusion_stage2, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance_org(self._wrap_diffusion_stage1(diffusion_stage1), # warp 使用的时间t也是scale 处理之后的形式；
                                           self._wrap_diffusion_stage2(diffusion_stage2), *args, **kwargs)

    def p_mean_variance(
            self, diffusion_stage1, diffusion_stage2, Fusion_Control_Model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_diffusion_stage1(diffusion_stage1),
                                       self._wrap_diffusion_stage2(diffusion_stage2), Fusion_Control_Model, *args,
                                       **kwargs)
    def p_mean_variance_fuse(
            self,diffusion_stage1_vis,diffusion_stage2_vis,diffusion_stage1_ir,diffusion_stage2_ir,
            Fusion_Control_Model, *args, **kwargs
    ):
        return super().p_mean_variance_fuse(
            self._wrap_diffusion_stage1(diffusion_stage1_vis),
            self._wrap_diffusion_stage2(diffusion_stage2_vis),
            self._wrap_diffusion_stage1(diffusion_stage1_ir),
            self._wrap_diffusion_stage2(diffusion_stage2_ir),
            Fusion_Control_Model, *args, **kwargs)

    def train_FCM_loss_fuse(
            self,diffusion_stage1_vis,diffusion_stage2_vis,diffusion_stage1_ir,diffusion_stage2_ir,
            Fusion_Control_Model, *args, **kwargs
    ):
        return super().train_FCM_loss_fuse(
            self._wrap_diffusion_stage1(diffusion_stage1_vis),
            self._wrap_diffusion_stage2(diffusion_stage2_vis),
            self._wrap_diffusion_stage1(diffusion_stage1_ir),
            self._wrap_diffusion_stage2(diffusion_stage2_ir),
            Fusion_Control_Model, *args, **kwargs)


    def train_FCM_loss(
            self, diffusion_stage1, diffusion_stage2, Fusion_Control_Model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().train_FCM_loss(self._wrap_diffusion_stage1(diffusion_stage1),
                                      self._wrap_diffusion_stage2(diffusion_stage2), Fusion_Control_Model, *args,
                                      **kwargs)

    def modulated_loss(
            self, diffusion_stage1, diffusion_stage2, Fusion_Control_Model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().modulated_loss(self._wrap_diffusion_stage1(diffusion_stage1),
                                      self._wrap_diffusion_stage2(diffusion_stage2), Fusion_Control_Model, *args,
                                      **kwargs)

    def training_losses(
            self, diffusion_stage1, diffusion_stage2, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_diffusion_stage1(diffusion_stage1),
                                       self._wrap_diffusion_stage2(diffusion_stage2), *args, **kwargs)

    def _wrap_diffusion_stage1(self, model):
        if isinstance(model, _WrappedModel_diffusion_stage1):
            return model  # 给定的参数就是diffusion_stage1
        return _WrappedModel_diffusion_stage1(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _wrap_diffusion_stage2(self, model):
        if isinstance(model, _WrappedModel_diffusion_stage2):
            return model
        return _WrappedModel_diffusion_stage2(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel_diffusion_stage1:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs): # ts 为时间采样的下标位置了；
        # 这里给定的ts的范围就是从 0-19 的了；
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        # 在这里从下标位置映射转换为了原始的时间步的位置内容 从 0-249 随后rescale_timesteps 再将其映射到 1000的范围中
        # self.original_num_steps
        new_ts = map_tensor[ts]  # 这里的ts 就是 0-50根据一种权重采样得到的内容
        if self.rescale_timesteps:  # 随后的new_ts 为将其映射回到 0-1000 之间的内容； 随后同时进行缩放的操作
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
            # 映射的参数应该就是从 1000和 original_num_steps 上计算得到的了；
            # 将原始的 0 4 17 35 47  0-50 映射到0-999 的浮点数的内容 ；
        return self.model(x, new_ts, **kwargs)


class _WrappedModel_diffusion_stage2:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, feature_hidden, feature_skip_list, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        # map_tensor 内容就是被选中的 20 个在于 0-100 之间的数据
        # map_tensor
        # ts 内容是介于 0-19 之间的整数的形式；
        new_ts = map_tensor[ts]
        if self.rescale_timesteps: # 1000 100 20  计算scale 使用的就是这个 1000/100 得到的 ；
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, feature_hidden, feature_skip_list, **kwargs)
