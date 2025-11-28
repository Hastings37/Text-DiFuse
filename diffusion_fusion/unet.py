import math
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_util import (SiLU, avg_pool_nd, checkpoint, conv_nd, linear,
                      normalization, timestep_embedding, zero_module) # 自定义其中的checkpoint 前向和反向传播操作；
def group_norm(channels):
    return nn.GroupNorm(32, channels)



class TimeBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        null
        """

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
    # 瓶颈结构


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        # H/W+2*padding-kernel_size / stride +1 = H/W

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        # 对应最大位置出现的位置的向量内容；
        out = torch.cat([avgout, maxout], dim=1)
        # B 2 H W -> B 1 H W
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

'''
自定义的具有条件注入能力的Sequential 容器模块 
构建起来一个模块链，将时间或者是条件信息自动传递给序列中需的特定层中； 

'''
class TimeSequential(nn.Sequential, TimeBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimeBlock):
                x = layer(x, emb) # 需要接受时间/条件注入的部分
            else:
                x = layer(x)
        return x


def time_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [batch_size x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# 只要有时间信息的注入就必须要集成实现这里的内容；
class ResBlock1(TimeBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(), # x*sigmoid(x)
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            group_norm(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            # 恒等映射操作；

    def forward(self, x, t): # 这里的t 就是对应的emb 嵌入的内容
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x) # 将in_dim 映射到了 out_dim 上
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None] # batch_size x  1 1
        h = self.conv2(h)
        return h + self.shortcut(x) # 最后进行一个残差连接的操作；

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3: # NCDHW
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels # 输入的图像的通道维度
        self.emb_channels = emb_channels # 时间信息的维度
        self.dropout = dropout # dropout 的概率
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        # 将通道维度映射到 embed_dim 上
        self.in_layers = nn.Sequential(
            normalization(channels), # 通道维度分割为32个进行归一化 进行分组归一化处理；
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
            # 这里的3为卷积核的大小为3 dim 为控制其中的卷积都是 2d卷积的形式；
        )
        # 需要scale norm 映射到两倍的通道上去 不需要的时候 直接映射到out_channels 上去
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        ) # 使用scale_shift_norm 就是需要将时间内容映射到两倍的输出通道上 ；
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ), # conv_nd 指定为3 BCDHW DHW 上进行滑动操作；
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1 # kernel_size
            )
        else: # 第一个参数指定其进行的是nd 的卷积操作；
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x) # 转换到 output_dim  上去；
        emb_out = self.emb_layers(emb).type(h.dtype) # 映射到out_dim/out_dim*2 BC / B2*C
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            # scale shift 参数；
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """

    def __init__(
        self, channels, 
        num_heads=1, 
        num_head_channels=-1, 
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels # 通道数量
        self.num_heads = num_heads # 注意力头的数量
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels  # 通过通道的数量进行实际上的推导的操作；
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads 先分割出来qkv然后分割得到heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv 分割出来heads 然后分割出来qkv
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
        # 这里真是奇妙，还可以自定义这里的形式；

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape # 收集剩余的维度并且构成一个list的形式；
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial) #残差连接过了；


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape # 批次大小 总的通道数量 序列的长度
        assert width % (3 * self.n_heads) == 0 #B C H*W
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        ) 
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_head_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param num_heads: the number of attention heads in each attention layer.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels, # unet 第一个卷积快，对应最高层分辨率的输出通道的数量情况
        out_channels, # 预测的是噪声或者是干净的数据内容
        num_res_blocks,
        attention_resolutions, # 注意力机制作用的分辨率 通常是进行缩小到原始的 1/4 之后进行；
        dropout=0,
        channel_mult=(1, 2, 4, 8), # 通道扩展的速度
        conv_resample=True, # 使用卷积上下采样，其中存在可更新的参数；
        dims=2, # 信号的维度
        use_checkpoint=False, #梯度检查点，节省内存的技术
        num_heads=1,# 注意力头的数量
        num_head_channels=-1, # 每个头的通道数量
        num_heads_upsample=-1, # 模型上采样中注意力块使用的注意力头的数量；
        use_scale_shift_norm=False, # 使用FILM的条件注入机制
        use_new_attention_order=False,# 是否使用新的注意力拆分顺序
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads # 默认和下采样保持一致；注意力头的数量；

        self.in_channels = in_channels
        self.model_channels = model_channels # 模型的基准通道数量；
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks # 每个下采样级别中的残差块的数量
        self.attention_resolutions = attention_resolutions # 注意力机制作用的分辨率
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample # 使用卷积上下采样，其中存在可学习参数
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample # 模型上采样中注意力块使用的注意力头的数量；
        self.dtype = torch.float32

        time_embed_dim = model_channels * 4 #unet的基准通道数量；
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList( # 只是将维度转换到model_channels 上去；
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels] # 存储用于跳跃连接的通道数量
        ch = model_channels # 当前块的输入通道数量
        ds = 1 # 当前分贝率下的下采样倍数
        for level, mult in enumerate(channel_mult):# 每一个下采样级别  1 2 4 8 为其通道数量需要扩展的倍数情况；
            # 同时也是其进行下采样的级别；
            for _ in range(num_res_blocks):# 每次下采样中的残差块的数量；
                layers = [
                    ResBlock(
                        ch, # 相当于每次其通道的维度都提高到到一定的程度 也不一定是这样的形式的；
                        time_embed_dim,# model_channels * 4
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,# FILM 条件注入的操作；
                    )
                ]
                ch = mult * model_channels # 新的通道的数量情况 进行了Res连接的操作；
                if ds in attention_resolutions:# 其下采样的倍率了 1 2 4 8 的时候使用 attention 处理；
                    layers.append(
                        AttentionBlock( # 维度并不发生变化
                            ch, 
                            use_checkpoint=use_checkpoint, 
                            num_heads=num_heads, 
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                # 基本上就是每个res的后面都有一个 attention的机制；
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch) # 对应的多个层次的维度情况；
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock( # 要是没有指定out_channels 就是保持不变的状态的；
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch, 
                use_checkpoint=use_checkpoint, 
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)





class diffusion_stage_Encode(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_head_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param num_heads: the number of attention heads in each attention layer.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,# 默认的情况下我们使用的维度是 1 这样的；
        model_channels,# model_channels =128
        out_channels,# 要是需要学习sigma 就设定为2
        num_res_blocks,# 2
        attention_resolutions,# 128//16==8 128//8==16 使用注意力机制的分辨率
        dropout=0, #图像总的尺寸位置是 128*128 下采样16 之后的维度就是 8*8
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,    
        num_heads=1,
        num_head_channels=-1, # 计算自动得到
        num_heads_upsample=-1,# 为上采样路径中的注意力特别指定的注意力头数量
        use_scale_shift_norm=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads #注意力头的数量；
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dtype = torch.float32

        time_embed_dim = model_channels * 4 # 512
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels] # [128] B 128 128 128
        ch = model_channels
        ds = 1 # 现在的分辨率对应进行的下采样的倍数  达到 8 16 的时候需要使用注意力机制；
        for level, mult in enumerate(channel_mult):#  1 1 2 3 4
            for _ in range(num_res_blocks):# 2
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,# 将对应的时间的信息内容，通过FiLM的形式调制注入到其中的操作；
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, 
                            use_checkpoint=use_checkpoint, 
                            num_heads=num_heads, 
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),  # 没有指定out_dim 就是保持 ch 不变的了；
            AttentionBlock(
                ch, 
                use_checkpoint=use_checkpoint, 
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )



    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)  # 32 32 32 32 64 64·
        h = self.middle_block(h, emb) # 将时间信息内容再一次嵌入其中；
        # 通过了bottlelack 进行了进一步的调制之后得到的内容 作为最终的输出encoder的内容；
        return h, hs # 最终的输出特征+多层的中间特征列表内容；





class diffusion_stage_Decode(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_head_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param num_heads: the number of attention heads in each attention layer.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,# 要是需要学习sigma 为 2 否则为1
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),# 1 1 2 4 8
        conv_resample=True,
        dims=2,
        use_checkpoint=False,    
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dtype = torch.float32

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )


        input_block_chans = [model_channels]# [128] 这样的维度；
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):# 1 1 2 3 4
            for _ in range(num_res_blocks):
       
                ch = mult * model_channels
               
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                input_block_chans.append(ch)
                ds *= 2 # 计算其下采样的最后的倍数；


        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:# 4 3 2 1 1
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps,h , hs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
           
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)




class diffusion_stage1(diffusion_stage_Encode):
    """
    Expects an extra kwarg `low_light` in `forward` to condition on a normal-light image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)
        # 位置参数和名称参数；

    def forward(self, x, timesteps, condition, **kwargs):
        if condition is not None:
            x = torch.cat([x, condition], dim=1) # 将condition cat上去就行了
        return super().forward(x, timesteps, **kwargs)
    # 带有噪声的图像内容来实现去噪信息的先验扩展操作；




class diffusion_stage2(diffusion_stage_Decode):
    """
    Expects an extra kwarg `low_light` in `forward` to condition on a normal-light image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs) # 也是在这里自动* 2 处理过了；

    def forward(self, x, timesteps,h,hs, condition, **kwargs):
        if condition is not None:
            x = torch.cat([x, condition], dim=1)
        return super().forward(x, timesteps,h,hs, **kwargs)
        



# FCM
class Get_Fusion_Control_Model(nn.Module):
    def group_norm_29(self,num_channels):
        return nn.GroupNorm(num_groups=29, num_channels=num_channels)

    def __init__(self,
                 in_channels=1276,
                 model_channels=638,
                 ):
        super(Get_Fusion_Control_Model,self).__init__()

   
        self.in_channels = in_channels
        self.model_channels = model_channels
       

        self.inBlock = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        self.middleBlock1 = nn.Sequential(
            self.group_norm_29(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
        )
        self.CBMA1=CBAM(channel=model_channels)
        
        self.middleBlock2 = nn.Sequential(
            self.group_norm_29(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
        )
        self.middleBlock3 = nn.Sequential(
            self.group_norm_29(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
        )
        self.middleBlock4 = nn.Sequential(
            self.group_norm_29(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
        )   
        
        self.CBMA2=CBAM(channel=model_channels)
        self.CBMA3=CBAM(channel=model_channels)
        self.CBMA4=CBAM(channel=model_channels)
        
        
        self.outBlock1= nn.Sequential(
            self.group_norm_29(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
      
        self.outBlock2 = nn.Sequential(
            self.group_norm_29(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.outBlock3 = nn.Sequential(
            self.group_norm_29(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )


    def forward(self, h1,h2):
    
        h=torch.cat([h1,h2],  dim=1)   
        middle1=self.inBlock(h)  # 映射回到指定的维度上的操作；model_channels
        middle2=self.middleBlock1(middle1) # 非线性映射操作；通道上的维度并不变化；

        m22=self.CBMA1(middle2)
        middle3=self.middleBlock2(m22)
        middle4=self.middleBlock3(m22)
        middle5=self.middleBlock4(m22)
        m1=self.CBMA2(middle3)
        m2=self.CBMA3(middle4)
        m3=self.CBMA4(middle5)

        w_v=self.outBlock1(m1) # 对应生成的是逐个位置上的权重内容；
        w_i = self.outBlock2(m2)
        bais=self.outBlock3(m3)
        out=(w_v*h2)+(w_i*h1)
        return out




