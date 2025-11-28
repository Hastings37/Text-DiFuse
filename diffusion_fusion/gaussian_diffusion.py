import enum
import math
import torch.nn as nn
import numpy as np
import torch
import cv2
from PIL import Image
from pyexpat import features
from torch.optim import AdamW
from .losses import discretized_gaussian_log_likelihood, normal_kl
from .nn_util import mean_flat
import torch.nn.functional as F
from math import exp
import torchvision.transforms.functional as Ft
from .util import to_numpy_image, Sobelxy, SSIM, high_freq_enhancement_loss, local_contrast_enhancement_loss
import os


# def Get_feature_all(feature_hidden_1, feature_skip_list_1):
#     #
#     B, C_, H, W = feature_hidden_1.shape
#     skip_channel_list = [2, 2, 2, 6, 6, 4, 16, 16, 8, 32, 32, 32, 128, 128, 128]
#     for index in range(len(skip_channel_list)):
#         feature_middle = feature_skip_list_1.pop().view(B, skip_channel_list[index], H * 16,
#                                                         W * 16) if index != 2 else torch.cat(
#             (feature_skip_list_1.pop(), torch.full((B, 128, H, W), 1).to(feature.device)), dim=1).view(B,
#                                                                                                        skip_channel_list[
#                                                                                                            index],
#                                                                                                        H * 16, W * 16)
#         feature = torch.cat((feature, feature_middle), dim=1) if index != 0 else feature_middle
#     feature = torch.cat((feature, feature_hidden_1.view(B, 2, H * 16, W * 16)), dim=1)
#     return feature
def Get_feature_all(feature_hidden_1, feature_skip_list_1):
    """
    将 bottleneck 特征和 skip connection 特征全部重排(PixelShuffle逆操作)到统一的空间尺寸并拼接。

    Args:
        feature_hidden_1: [B, 512, 8, 8]
        feature_skip_list_1: List of tensors, 长度12
                             从 Layer 0 (128,64,64) 到 Layer 11 (512,8,8)

    Returns:
        feature: [B, Total_Channels, 64, 64]
    """

    # 1. 获取基础维度
    B, C_hidden, H_s, W_s = feature_hidden_1.shape
    # H_s=8, W_s=8

    # 2. 设定目标尺寸 (根据你的要求: H*8, W*8 -> 64x64)
    # 注意：这里的 8 是基于 feature_hidden_1 的尺寸放大的倍数
    H_target = H_s * 8
    W_target = W_s * 8

    # 用于收集所有处理后的特征图
    processed_features = []

    # 3. 处理 Hidden Feature (Bottleneck)
    # [B, 512, 8, 8] -> [B, ?, 64, 64]
    # 通道计算: 512 * (8*8) / (64*64) = 512 / 64 = 8
    C_hidden_new = int(C_hidden * H_s * W_s / (H_target * W_target))
    feature_hidden_reshaped = feature_hidden_1.view(B, C_hidden_new, H_target, W_target)

    # 将 Hidden 层先加入列表 (或者你想把它放在最后也可以，这里默认放最前)
    processed_features.append(feature_hidden_reshaped)

    # 4. 循环处理 Skip List (倒序: Layer 11 -> Layer 0)
    # 使用 while 循环配合 pop，直到列表为空
    while feature_skip_list_1:
        # 弹出最后一个元素 (例如 Layer 11)
        current_feat = feature_skip_list_1.pop()

        # 获取当前特征的形状
        # e.g., Layer 11: [B, 512, 8, 8]
        # e.g., Layer 9:  [B, 384, 8, 8]
        # e.g., Layer 0:  [B, 128, 64, 64]
        b, c, h, w = current_feat.shape

        # 自动计算重排后的新通道数
        # 原理：体积守恒 (C * H * W) = (C_new * H_target * W_target)
        c_new = (c * h * w) / (H_target * W_target)

        # 检查是否能整除 (为了安全性)
        if c_new % 1 != 0:
            raise ValueError(f"Feature shape ({c},{h},{w}) cannot be reshaped to ({H_target},{W_target})")

        c_new = int(c_new)

        # 执行 View 操作 (Depth-to-Space)
        feat_reshaped = current_feat.view(B, c_new, H_target, W_target)

        # 加入列表
        processed_features.append(feat_reshaped)

    # 5. 最终拼接
    # 列表里的 tensor 现在都是 [B, C_n, 64, 64]，在通道维 dim=1 拼接
    feature = torch.cat(processed_features, dim=1)

    return feature


# def Spit_feature_all(feature_hidden_fusion):
#     B, C_, H, W = feature_hidden_fusion.shape
#     skip_channel_list = [2, 2, 2, 6, 6, 4, 16, 16, 8, 32, 32, 32, 128, 128, 128]
#     skip_channel_list_next = [512, 512, 512, 384, 384, 256, 256, 256, 128, 128, 128, 128, 128, 128, 128]
#     skip_H_W_list = [16, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2, 1, 1, 1]
#     C_UP, C_DOWN = C_, C_ - 2
#     feature_hidden = feature_hidden_fusion[:, C_DOWN:C_UP, :, :].view(B, 512, int(H / (16)), int(W / (16)))
#     feature_skip_list = []
#     for index in range(len(skip_channel_list)):
#         C_UP, C_DOWN = C_DOWN, C_DOWN - skip_channel_list[len(skip_channel_list) - index - 1]
#         # 也是反向逐渐划分获取到需要的通道的内容的操作；
#         feature_skip = feature_hidden_fusion[:, C_DOWN:C_UP, :, :].view(B, skip_channel_list_next[  # 需要转换回去的维度情况；
#             len(skip_channel_list) - index - 1], int(H / skip_H_W_list[len(skip_channel_list) - index - 1]),
#                                                                         int(W / skip_H_W_list[
#                                                                             len(skip_channel_list) - index - 1])) if index != 12 else feature_hidden_fusion[
#                                                                                                                                       :,
#                                                                                                                                       C_DOWN:C_UP,
#                                                                                                                                       :,
#                                                                                                                                       :].view(
#             B, skip_channel_list_next[len(skip_channel_list) - index - 1],
#             int(H / skip_H_W_list[len(skip_channel_list) - index - 1]),
#             int(W / skip_H_W_list[len(skip_channel_list) - index - 1]))[:, :384, :, :]
#         feature_skip_list.append(feature_skip)
#     return feature_hidden, feature_skip_list

def Split_feature_all(feature_hidden_fusion):
    """
    将融合后的特征图 [B, 638, 64, 64] 拆解还原回 Hidden 和 Skip List。

    Args:
        feature_hidden_fusion: [B, 638, 64, 64]

    Returns:
        feature_hidden: [B, 512, 8, 8]
        feature_skip_list: List, 包含 Layer 0 到 Layer 11 的特征图 (正序)
    """

    B, Total_C, H, W = feature_hidden_fusion.shape
    # H=64, W=64

    # 指针：记录当前切分到了哪个通道，从 0 开始
    current_channel_ptr = 0

    # ==========================================
    # 1. 还原 Feature Hidden
    # ==========================================
    # Hidden 原始形状: [512, 8, 8]
    # 压缩后通道数: 512 * (8*8) / (64*64) = 8
    c_hidden_compressed = 8

    # 切片
    hidden_slice = feature_hidden_fusion[:, current_channel_ptr: current_channel_ptr + c_hidden_compressed, :, :]
    # 还原形状 [B, 512, 8, 8]
    feature_hidden = hidden_slice.view(B, 512, 8, 8)

    # 更新指针
    current_channel_ptr += c_hidden_compressed

    # ==========================================
    # 2. 还原 Skip List
    # ==========================================
    # 注意：在 Get 阶段，我们是 pop() 出来的，所以拼接顺序是 Layer 11 -> Layer 10 -> ... -> Layer 0
    # 所以这里定义的配置表必须也是这个顺序

    # 定义每一层的原始形状 (Channel, H, W)
    # 顺序：Layer 11 -> Layer 0
    layers_config_reverse = [
        (512, 8, 8),  # Layer 11
        (512, 8, 8),  # Layer 10
        (384, 8, 8),  # Layer 9  (现在可以完美还原，无特殊操作)
        (384, 16, 16),  # Layer 8
        (384, 16, 16),  # Layer 7
        (256, 16, 16),  # Layer 6
        (256, 32, 32),  # Layer 5
        (256, 32, 32),  # Layer 4
        (128, 32, 32),  # Layer 3
        (128, 64, 64),  # Layer 2
        (128, 64, 64),  # Layer 1
        (128, 64, 64)  # Layer 0
    ]

    temp_skip_list = []

    for (org_c, org_h, org_w) in layers_config_reverse:
        # 1. 计算这一层在 64x64 画布上占用了多少通道
        # 公式: c_new = (C * H * W) / (64 * 64)
        c_compressed = int((org_c * org_h * org_w) / (H * W))

        # 2. 切片提取
        feat_slice = feature_hidden_fusion[:, current_channel_ptr: current_channel_ptr + c_compressed, :, :]

        # 3. 还原形状 (Space-to-Depth)
        feat_restored = feat_slice.view(B, org_c, org_h, org_w)

        # 4. 加入临时列表
        temp_skip_list.append(feat_restored)

        # 5. 更新指针
        current_channel_ptr += c_compressed

    # ==========================================
    # 3. 整理返回结果
    # ==========================================
    # temp_skip_list 目前是 [Layer 11, Layer 10, ..., Layer 0]
    # 通常网络层级列表都是正序的 [Layer 0, ..., Layer 11]，所以这里反转一下
    feature_skip_list = temp_skip_list[::-1]

    return feature_hidden, feature_skip_list


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(  # 步数情况和综合起来的内容；
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )  # 其中的s为偏移量内容；
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


# 其中alpha_bar 为一个指定的连续函数，实现的功能是定义在t 时刻还保留的信息量； 其实就是 a_cumprod 的一个连续版本；
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    # 对应一个连续的函数内容，定义扩散的过程中的t 时刻其中还在保留的信息内容数量；
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


# 模型预测目标的一个枚举类内容
class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    # .name .value 获取到枚举类中的单个内容的属性信息；


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()  # 预测逆向过程中的方差
    FIXED_SMALL = enum.auto()  # 固定的小方差
    FIXED_LARGE = enum.auto()  # 固定的大方差
    LEARNED_RANGE = enum.auto()  # 学习范围内的方差


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like Ho et al's 
                              diffusion models (0 to 1000).
    """

    def __init__(
            self,
            *,
            betas,  # 重新调整后的少量内容；
            model_mean_type,
            model_var_type,
            loss_type,
            rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        # 这里的betas 就是选择确定的  25 个时间步长了；
        self.num_timesteps = int(betas.shape[0])  # num_timesteps 实际上就是我们重新采样之后的时间步的数量；
        # 指定的20个 1000 50 20 这样的修改处理；
        # 从250 中抽取到的 100 这样的情况的了；

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )
        self.sobelconv = Sobelxy()
        self.SSIM = SSIM()

    '''
    存在的问题： 
        为什么这里的i x_F_t 需要被两次进行encoder的操作
        这里的output_GT内容是什么 
        model_kwargs 内容是什么 
    '''

    @torch.enable_grad()
    def train_FCM_loss_fuse(self,diffusion_stage1_vis,diffusion_stage2_vis,diffusion_stage1_ir,diffusion_stage2_ir,
            Fusion_Control_Model, x_F_t, t, output_GT1, output_GT2,
            clip_denoised=True, denoised_fn=None, model_kwargs=None, model_kwargs1=None
    ):
        B,C= x_F_t.shape[:2]
        assert t.shape == (B,),print(f't shape:{t.shape},B:{B} 两者并不匹配出现错误 ！！！！！')
        features_hidden_1, feature_skip_list_1 = diffusion_stage1_vis(x_F_t, self._scale_timesteps(t), **model_kwargs)
        features_hidden_2, feature_skip_list_2 = diffusion_stage1_ir(x_F_t, self._scale_timesteps(t), **model_kwargs1)
        feature_all_1 = Get_feature_all(features_hidden_1, feature_skip_list_1)
        feature_all_2 = Get_feature_all(features_hidden_2, feature_skip_list_2)
        feature_all_fusion = Fusion_Control_Model(feature_all_1, feature_all_2)

        feature_hidden_fusion, feature_skip_list_fusion = Split_feature_all(feature_all_fusion)


        pass

    @torch.enable_grad()
    def train_FCM_loss(
            self, diffusion_stage1, diffusion_stage2, Fusion_Control_Model, x_F_t, t, output_GT1, output_GT2,
            clip_denoised=True, denoised_fn=None, model_kwargs=None, model_kwargs1=None
    ):
        B, C = x_F_t.shape[:2]
        assert t.shape == (B,)
        # 这里的数值t内容
        feature_hidden_1, feature_skip_list_1 = diffusion_stage1(x_F_t, self._scale_timesteps(t), **model_kwargs)
        feature_hidden_2, feature_skip_list_2 = diffusion_stage1(x_F_t, self._scale_timesteps(t), **model_kwargs1)
        feature_all_1 = Get_feature_all(feature_hidden_1, feature_skip_list_1)
        feature_all_2 = Get_feature_all(feature_hidden_2, feature_skip_list_2)
        feature_all_fusion = Fusion_Control_Model(feature_all_1, feature_all_2)  # 这里使用原始的退化图像内容作为condition内容；
        # 明白了原来是在这里的操作来的啊；
        # 融合之后再分离的操 最后得到需要的图像特征内容
        feature_hidden_fusion, feature_skip_list_fusion = Split_feature_all(feature_all_fusion)
        # 我感觉这里应该cross attention 融合输出一下指定的内容把；
        # 其训练的时候使用是对应的 vis 和 ir 的各自的内容啊；  用CLIP动态结合起来吗？？？


        model_output = diffusion_stage2(x_F_t, self._scale_timesteps(t), feature_hidden_fusion,
                                        feature_skip_list_fusion, **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x_F_t.shape[2:])  # 同样也是需要学习var 的情况下
            model_output, model_var_values = torch.split(model_output, C, dim=1)  # split 指定的是块的数量  chunk 指定的是块的大小；
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:  # ModelVarType.LEARNED_RANGE
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x_F_t.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x_F_t.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2  # 这样预测的一个系数内容就是从 -1  1 之间的
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:  # FIXED_SMALL  or  FIXED_LARGE
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x_F_t.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x_F_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)  # 去除噪声的fn操作
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:  # PREVIOUS_X
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x_F_t, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            # X_START or EPSILON
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x_F_t, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_F_t, t=t
            )  # 根据X_0 X_t t 计算X_t-1 将其内容作为model_mean
        else:
            raise NotImplementedError(self.model_mean_type)

        x0t_gt1 = (output_GT1 + 1) / 2
        x0t_gt2 = (output_GT2 + 1) / 2
        x0t = (pred_xstart + 1) / 2  # 对应的gt 内容只是用来计算图像信息的；

        x0t_max = torch.max(x0t_gt1, x0t_gt2)
        gt1_grad = self.sobelconv(x0t_gt1)
        gt2_grad = self.sobelconv(x0t_gt2)
        x0t_grad = self.sobelconv(x0t)
        x0t_grad_max = torch.maximum(gt1_grad, gt2_grad)
        loss_int = F.l1_loss(x0t, x0t_max)
        loss_grad = F.l1_loss(x0t_grad, x0t_grad_max)
        # loss_ssim=0.3*(1-self.SSIM(x0t,x0t_gt2))+0.7*(1-self.SSIM(x0t,x0t_gt1))
        loss = loss_int + loss_grad  # 强度内容和grad 梯度内容的实现了；
        print("timestep:", t, "total_loss", loss, "loss_int:", loss_int, "loss_grad:",
              loss_grad)  # ,"loss_ssim:",loss_ssim)
        return loss, {
            "MAX": x0t_max,
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    @torch.enable_grad()
    def modulated_loss(
            self, diffusion_stage1, diffusion_stage2, Fusion_Control_Model, x_F_t, t, output_GT, target_modulated,
            clip_denoised=True, denoised_fn=None, model_kwargs=None, model_kwargs1=None
    ):
        B, C = x_F_t.shape[:2]
        assert t.shape == (B,)
        feature_hidden_1, feature_skip_list_1 = diffusion_stage1(x_F_t, self._scale_timesteps(t), **model_kwargs)
        feature_hidden_2, feature_skip_list_2 = diffusion_stage1(x_F_t, self._scale_timesteps(t), **model_kwargs1)
        feature_all_1 = Get_feature_all(feature_hidden_1, feature_skip_list_1)
        feature_all_2 = Get_feature_all(feature_hidden_2, feature_skip_list_2)
        feature_all_fusion = Fusion_Control_Model(feature_all_1, feature_all_2)
        feature_hidden_fusion, feature_skip_list_fusion = Split_feature_all(feature_all_fusion)

        model_output = diffusion_stage2(x_F_t, self._scale_timesteps(t), feature_hidden_fusion,
                                        feature_skip_list_fusion, **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x_F_t.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x_F_t.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x_F_t.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x_F_t.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x_F_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x_F_t, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x_F_t, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_F_t, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        x0t_gt = (output_GT + 1) / 2
        x0t = (pred_xstart + 1) / 2

        gt_grad = self.sobelconv(x0t_gt)
        x0t_grad = self.sobelconv(x0t)
        loss_int = F.l1_loss(x0t, x0t_gt)
        loss_grad = F.l1_loss(x0t_grad, gt_grad)
        loss_en = high_freq_enhancement_loss(x0t_grad, target_modulated)
        loss_contrast = local_contrast_enhancement_loss(x0t, target_modulated, kernel_size=5)
        loss = loss_int + loss_grad + 100 * loss_en + 3 * loss_contrast
        print("timestep:", t)
        return loss, {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    #  根据起始的x0和时间步t，获取到对应的扩散后的 mean var log(var)
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    # 根据起始的x0和时间步t，获取到对应的扩散后的样本内容；前向扩散的操作；
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        # 这里的t是那100个内容的下标记情况  从0-99 但是其保存的数据内容的范围是 0-250的
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    # 这里计算的并不是实际上的均值和方差内容，（使用的x0为真实的x0 这里的内容在我们实际上是不存在的）
    # 计算的是扩散过程中的后验分布 q(xt-1 | xt, x0) 的均值和方差内容；这里的是实际上的均值和方差内容；
    # 并不是我们后面需要借助网络预测的内容了； q就是对应的前向的过程
    # 用来和后面我们预测反向的 时候进行对比的；
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    '''
    modelmean_type  
    直接预测xt-1 均值直接使用预测的就行 x0从xt-1 反向推导得到； 
    直接预测为x0  使用标准的公式求取xt-1 的均值；直接使用模型的输出作为x0 
    预测噪声 使用噪声预测x0 结合后验分布计算xt-1 的均值； 通过前向的公式推导得到 x0 内容
    
    '''

    # stage1 和 stage2 作为额外的参数传递进来的操作；
    # 对应就是我们的反向采样的一个过程 得到对应的 mean var log var X0Pred 等等
    def p_mean_variance_org(
            self, diffusion_stage1, diffusion_stage2, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}  # 给定神经网路的额外参数内容，用于实现条件化
        B, C = x.shape[:2]
        assert t.shape == (B,)
        feature_hidden, feature_skip_list = diffusion_stage1(x, self._scale_timesteps(t), **model_kwargs)
        # h hs  我感觉我需要看看就是使用一个test 看看其中的内容怎样；
        model_output = diffusion_stage2(x, self._scale_timesteps(t), feature_hidden, feature_skip_list, **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:  # 后面为限定在 大 小 范围中的方差预测；
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            # split 为块的大小情况 chunk为块的数量情况；dim 指定维度；
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values  # 模型预测的是后验方差的对数值；
                model_variance = torch.exp(model_log_variance)
            else:  # LEARNED_RANGE
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                    # 对应在时间步为t是时刻上的数据内容；
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )  # 这里的均值实际上就是xt-1 的内容；  使用xt-1反向推理x_0 的内容 讲究人
            model_mean = model_output  # 使用模型的结果作为均值并且使用这个数值得到 x0

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)  # 直接预测x0
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )  # 使用噪声预测x0 结合后验分布
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )  # 计算指定的内容的分布信息；

        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    # mean variance log_variance pred_xstart
    # 这也就是为什么这里是不训练对应的参数的内容的；
    @torch.no_grad()
    def p_mean_variance_fuse(
            self,diffusion_stage1_vis,diffusion_stage2_vis,diffusion_stage1_ir,diffusion_stage2_ir,
            Fusion_Control_Model, x, t, couple_single=True,
            clip_denoised=True, denoised_fn=None, model_kwargs=None, model_kwargs1=None
    ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        if couple_single == True:  # 为一对信号的情况下；
            # 这里将t转换为了对应的比例的操作 ；
            # 我们先将 t 从指定的范围 str 等间隔拿到之后，然后重新计算其betas内容
            feature_hidden_1, feature_skip_list_1 = diffusion_stage1_vis(x, self._scale_timesteps(t), **model_kwargs)
            # 这两个内容我们直接使用干净的图像内容不就行了？？？
            # 而且需要train 两条内容一个是红外的 另一个是可见光的内容了；
            feature_hidden_2, feature_skip_list_2 = diffusion_stage1_ir(x, self._scale_timesteps(t), **model_kwargs1)
            # 使用的是双分支的结构
            feature_all_1 = Get_feature_all(feature_hidden_1, feature_skip_list_1)
            feature_all_2 = Get_feature_all(feature_hidden_2, feature_skip_list_2)
            feature_all_fusion = Fusion_Control_Model(feature_all_1, feature_all_2)
            # 融合控制模型的特征内容；

            feature_hidden_fusion, feature_skip_list_fusion = Split_feature_all(feature_all_fusion)

            # 进一步将这里的内容进行分解处理
            # h hs 内容
            # 将我们的CLIP模型的参数引入到这里来使用；也是可以的把 这里只用给定相应的文本内容就行了对吧；
            model_output = diffusion_stage2_vis(x, self._scale_timesteps(t), feature_hidden_fusion,
                                            feature_skip_list_fusion,
                                            **model_kwargs)  # vis 为引导的内容；# 将上面每一步得到的干净的图像内容作为 这个的生成操作了属于是；
        else:
            feature_hidden, feature_skip_list = diffusion_stage1_vis(x, self._scale_timesteps(t), **model_kwargs)
            model_output = diffusion_stage2_vis(x, self._scale_timesteps(t), feature_hidden, feature_skip_list,
                                            **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)

        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }


    @torch.no_grad()
    def p_mean_variance(  # 默认的情况就是True的 ；
            self, diffusion_stage1, diffusion_stage2, Fusion_Control_Model, x, t, couple_single=True,
            clip_denoised=True, denoised_fn=None, model_kwargs=None, model_kwargs1=None
    ):  # 融合控制结构
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        if couple_single == True:  # 为一对信号的情况下；
            # 这里将t转换为了对应的比例的操作 ；
            # 我们先将 t 从指定的范围 str 等间隔拿到之后，然后重新计算其betas内容
            feature_hidden_1, feature_skip_list_1 = diffusion_stage1(x, self._scale_timesteps(t), **model_kwargs)
            # 这两个内容我们直接使用干净的图像内容不就行了？？？
            # 而且需要train 两条内容一个是红外的 另一个是可见光的内容了；
            feature_hidden_2, feature_skip_list_2 = diffusion_stage1(x, self._scale_timesteps(t), **model_kwargs1)
            # 使用的是双分支的结构
            feature_all_1 = Get_feature_all(feature_hidden_1, feature_skip_list_1)
            feature_all_2 = Get_feature_all(feature_hidden_2, feature_skip_list_2)
            feature_all_fusion = Fusion_Control_Model(feature_all_1, feature_all_2)
            # 融合控制模型的特征内容；

            feature_hidden_fusion, feature_skip_list_fusion = Split_feature_all(feature_all_fusion)

            # 进一步将这里的内容进行分解处理
            # h hs 内容
            model_output = diffusion_stage2(x, self._scale_timesteps(t), feature_hidden_fusion,
                                            feature_skip_list_fusion,
                                            **model_kwargs)  # vis 为引导的内容；# 将上面每一步得到的干净的图像内容作为 这个的生成操作了属于是；
        else:
            feature_hidden, feature_skip_list = diffusion_stage1(x, self._scale_timesteps(t), **model_kwargs)
            model_output = diffusion_stage2(x, self._scale_timesteps(t), feature_hidden, feature_skip_list,
                                            **model_kwargs)
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)

        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }  # 这里其实预测的就是对应的融合图像对的内容了不是吗；

    # 从噪声中预测x0 的内容； 适用于模型预测的是噪声的情况下；对应使用的公式就是最原始的推导公式；从x0到xt 的公式反推；
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    # 模型预测的内容就是xt-1 的内容；  通过xt-1 反向推导x0 的内容；
    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
                * x_t
        )

    # 从x0 预测噪声的内容；也是使用最原始的推导公式的形式就行了；
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # 保证时间步被缩放到指定的范围中的情况；
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        # 这里的num_timesteps 指的就是我们的 100
        return t

    '''
    得到对应的反向扩散中的参数内容并且从其中采样得到一个xt-1
    '''

    @torch.no_grad()
    def p_sample(
            self, diffusion_stage1, diffusion_stage2, Fusion_Control_Model, x, t, couple_single, clip_denoised=True,
            denoised_fn=None, model_kwargs=None, model_kwargs1=None  # 选择是不是要用两个信息的操作；
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            diffusion_stage1,
            diffusion_stage2,
            Fusion_Control_Model,
            x,
            t,
            couple_single,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            model_kwargs1=model_kwargs1,
        )
        sample = out["mean"]  # 使用均值作为采样内容；
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"]}

    # 预测的xt_1 x_0 均值内容；使用其均值作为sample 内容；

    def p_sample_loop(
            self,
            diffusion_stage1,
            diffusion_stage2,
            Fusion_Control_Model,
            shape,
            couple_single=True,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            model_kwargs1=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                diffusion_stage1,
                diffusion_stage2,
                Fusion_Control_Model,
                shape,
                couple_single,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                model_kwargs1=model_kwargs1,
                device=device,
                progress=progress,
        ):
            final = sample  # 每次得到一个对应的yeild 出来的内容最终得到 需要的干净的图像信息；
        return final["sample"]

    def p_sample_loop_progressive(
            self,
            diffusion_stage1,
            diffusion_stage2,
            Fusion_Control_Model,
            shape,
            couple_single,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            model_kwargs1=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:  # 从模型推断其设备
            device = next(diffusion_stage1.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]  # 构建反向的时间步列表
        # T-1 .。。。。 0 这样的过程；

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:  # T-1 到 0 的过程；
            t = torch.tensor([i] * shape[0], device=device)  # B, 形成对应的时间标签内容；
            with torch.no_grad():  # 前向的推导其中没有梯度的内容存在；
                out = self.p_sample(
                    diffusion_stage1,
                    diffusion_stage2,
                    Fusion_Control_Model,
                    img,
                    t,
                    couple_single,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    model_kwargs1=model_kwargs1,
                )
                yield out
                # 每次都使用的是 均值作为其采样；
                img = out["sample"]

    '''
    计算扩散模型的变分下界的 对应时间步的 KL NLL 项
    '''

    def _vb_terms_bpd(
            self, diffusion_stage1, diffusion_stage2, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # 首先根据 x_t t 预测到我们需要的内容；
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )  # 计算真实的后验概率分布；这里是根据定义的超参数的内容得到的；
        out = self.p_mean_variance_org(
            diffusion_stage1, diffusion_stage2, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )  # 计算模型预测的后验概率分布；
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        # 从batch 到后面全都是取平均值的操作；
        kl = mean_flat(kl) / np.log(2.0)
        # 转化为对 2 取对数

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
            # 获取到的均值就认为是需要的 x0内容；
        )  # 计算−logpθ​(x0​∣x1​) 对应扩散模型的最后一步从 x1 到 x0
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)  # 对应计算的是从1000到这里的缩放的位置的操作了；
        # 0-19 中抽出indice 随后得到抽选的 0-50范围内的内容 最后将其根据scale缩放到1000的范围中去；
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    # model_kwargs: HR——Y内容；
    def training_losses(self, diffusion_stage1, diffusion_stage2, x_start, t, model_kwargs=None, noise=None):
        # x_start 为 HR model_kwargs 为 LR 内容；
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        # 从x_t 得到的内容；

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                diffusion_stage1=diffusion_stage1,
                diffusion_stage2=diffusion_stage2,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
                # 不这样放大将会导致 KL损失的内容比MSE 小两个数量级 ；
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:  # 需要同时学习 方差的时候；

            feature_hidden, feature_skip_list = diffusion_stage1(x_t, self._scale_timesteps(t), **model_kwargs)

            model_output = diffusion_stage2(x_t, self._scale_timesteps(t), feature_hidden, feature_skip_list,
                                            **model_kwargs)
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,  # 需要同时去学习其var内容的时候；
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:]), print(
                    f'x_t.shape:{x_t.shape},model_output.shape:{model_output.shape}')
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # split 需要进行指定的是大小 chunk 需要指定的是块数量
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                # 进行了拼接得到的大型的tensor 内容两次；
                # MSE 用来去训练 mean 正常进行反向传播操作  KL Loss 用来训练 variance mean 不被KL影响到；
                terms["vb"] = self._vb_terms_bpd(
                    diffusion_stage1=lambda *args, r=frozen_out: (r, r),  # 忽略对应的参数内容直接返回固定的rr 和r 内容；
                    diffusion_stage2=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]  # 计算得到的梯度并不回传到 mean部分因为其已经被detach 处理过了；
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0  # 其中为betas 的数量；
                    # 一个进行过缩放之后的KL损失项内容；
                    # 主要是因为后面这一项的内容数值往往很大导致的情况了；

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]  # 这是根据后验概率计算得到的内容；
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            # 计算均方误差内容；
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms  # 其中的loss为最终的训练的时候需要用到的损失内容；

    def p_sample_loop_single(self,
                             diffusion_stage1,
                             diffusion_stage2,
                             shape,
                             noise=None,
                             clip_denoised=True,
                             denoised_fn=None,
                             model_kwargs=None,
                             device=None,
                             progress=False,
                             ):
        final = None
        for sample in self.p_sample_loop_progressive_single(
                diffusion_stage1,
                diffusion_stage2,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample  # 每次得到一个对应的yeild 出来的内容最终得到 需要的干净的图像信息；
        return final["sample"]
        # for sample in self.p_sample_loop_progressive_single()

    @torch.no_grad()
    def p_sample_loop_progressive_single(self,
                                         diffusion_stage1,
                                         diffusion_stage2,
                                         shape,
                                         noise=None,
                                         clip_denoised=True,
                                         denoised_fn=None,
                                         model_kwargs=None,
                                         device=None,
                                         progress=False,
                                         ):
        if device is None:  # 从模型推断其设备
            device = next(diffusion_stage1.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise  # 输入的开始噪声内容；
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]  # 构建反向的时间步列表
        # T-1 .。。。。 0 这样的过程；

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)  # 默认展示其进度条；

        for i in indices:  # T-1 到 0 的过程；
            t = torch.tensor([i] * shape[0], device=device)  # B, 形成对应的时间标签内容；
            with torch.no_grad():  # 前向的推导其中没有梯度的内容存在；
                out = self.p_sample_single(
                    diffusion_stage1,
                    diffusion_stage2,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                # 每次都使用的是 均值作为其采样；
                img = out["sample"]

    @torch.no_grad()
    def p_sample_single(self,
                        diffusion_stage1,
                        diffusion_stage2,
                        x,
                        t,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None
                        ):
        out = self.p_mean_variance_single(
            diffusion_stage1,
            diffusion_stage2,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        sample = out["mean"]  # 使用均值作为采样内容；
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean": out["mean"]}

    def p_mean_variance_single(self,
                               diffusion_stage1,
                               diffusion_stage2,
                               x, t,
                               clip_denoised=True,
                               denoised_fn=None,
                               model_kwargs=None, ):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        feature_hidden, feature_skip_list = diffusion_stage1(x, self._scale_timesteps(t), **model_kwargs)
        model_output = diffusion_stage2(x, self._scale_timesteps(t), feature_hidden, feature_skip_list, **model_kwargs)
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)

        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }  # 这里其实预测的就是对应的融合图像对的内容了不是吗


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
