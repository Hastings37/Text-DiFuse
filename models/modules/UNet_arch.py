import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools
from models.modules.module_util import (
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, Residual, Identity)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ch=64, ch_mult=[1, 2, 4, 4], embed_dim=4):
        super().__init__()
        self.depth = len(ch_mult)

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_ch, ch, 3)

        # layers
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        ch_mult = [1] + ch_mult # 4 8 8 16
        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i + 1]
            self.encoder.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in),
                block_class(dim_in=dim_in, dim_out=dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if i == (self.depth - 1) else Identity(),
                Downsample(dim_in, dim_out) if i != (self.depth - 1) else default_conv(dim_in, dim_out)
            ]))

            self.decoder.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if i == (self.depth - 1) else Identity(),
                # 就只有在对应的底层进行 相应的操作； 我直接在这个上面测试的不行吗？？？？也是可以的对吧；
                Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = ch * ch_mult[-1]
        self.latent_conv = default_conv(mid_dim, embed_dim, 1)
        self.post_latent_conv = default_conv(embed_dim, mid_dim, 1)
        self.final_conv = nn.Conv2d(ch, out_ch, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))  # 16
        mod_pad_h = (s - h % s) % s  # 需要添加的内容；
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def encode(self, x):
        self.H, self.W = x.shape[2:]
        x = self.check_image_size(x, self.H, self.W)

        x = self.init_conv(x)  # 转换到embed_dim 中
        h = [x]
        for b1, b2, attn, downsample in self.encoder:
            x = b1(x)
            h.append(x)

            x = b2(x)
            x = attn(x)
            h.append(x)  #

            x = downsample(x)

        x = self.latent_conv(x)
        # print(f'x.shape:{x.shape}')
        # for i,layer in enumerate(h):
        #     print(f'h[{i}].shape:{layer.shape}')
        return x, h

    def decode(self, x, h):
        x = self.post_latent_conv(x)
        # print(f'x.shape after post_latent_conv:{x.shape}')
        for i, (b1, b2, attn, upsample) in enumerate(self.decoder):
            x = torch.cat([x, h[-(i * 2 + 1)]], dim=1)
            x = b1(x)
            # print(f'x.shape after b1 at decoder level {i}: {x.shape}')

            x = torch.cat([x, h[-(i * 2 + 2)]], dim=1)
            x = b2(x)
            x = attn(x)
            # print(f'x.shape before upsample at decoder level {i}: {x.shape}')

            x = upsample(x)

        x = self.final_conv(x + h[0])
        # print(f'final x.shape before cropping: {x.shape}')
        return x[..., :self.H, :self.W]

    def forward(self, x):
        x, h = self.encode(x)
        x = self.decode(x, h)

        return x


if __name__ == "__main__":
    model = UNet(
        in_ch=3,
        out_ch=3,
        ch=64,
        ch_mult=[1, 2, 4, 4],
        embed_dim=4
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ---------------------------------------------------
    # 2. 构造测试输入
    # （这里使用一个不能整除 16 的尺寸，用于测试 padding）
    # ---------------------------------------------------
    H, W = 233, 307
    x = torch.randn(1, 3, H, W).to(device)



    print("Input size:", x.shape)

    # ---------------------------------------------------
    # 3. 前向推理
    # ---------------------------------------------------
    with torch.no_grad():
        h,hs=model.encode(x)
        print(f'Encoded h shape: {h.shape}')

        y = model(x)

    print("Output size:", y.shape)

    # ---------------------------------------------------
    # 4. 检查是否与原始尺寸一致
    # ---------------------------------------------------
    if y.shape[2] == H and y.shape[3] == W:
        print("输出尺寸正确，与输入一致！")
    else:
        print("输出尺寸不一致，需要检查 decode 截取部分。")

    #  需要的形式就是参考这里构建的 潜在的特征内容是如何构成后面的扩散模型的输入的内容的；
    # 扩散的时候使用分割图和干净的图像内容作为条件扩散的操作；
