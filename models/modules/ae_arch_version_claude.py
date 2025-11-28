import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
from einops import rearrange
import clip
from models.modules.ae_arch import *


# ============================================================================
# Part 1: 核心组件保持不变
# ============================================================================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# [前面的所有辅助类保持不变: CBAM, Attention, TransformerBlock等]
# ... (这里省略重复代码，使用您原有的实现)

class FeatureWiseAffine(nn.Module):
    """文本特征条件注入模块"""

    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        if text_embed.dim() == 3:
            text_embed = text_embed.squeeze(1)

        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            beta = self.MLP(text_embed).view(batch, -1, 1, 1)
            x = x + beta
        return x


class FusionModule(nn.Module):
    """多模态融合模块：CBAM + Cross Attention + Affine"""

    def __init__(self, feature_dim, context_dim=768, reduction_ratio=8, kernel_size=7):
        super(FusionModule, self).__init__()

        # 1. CBAM注意力
        self.cbam = CBAM(feature_dim, reduction_ratio, kernel_size)

        # 2. 跨模态注意力（可选）
        self.cross_attention = Cross_attention(feature_dim, n_head=1)

        # 3. 文本条件注入
        self.affine = FeatureWiseAffine(context_dim, feature_dim, use_affine_level=True)

        # 4. 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, 1, 1),
            nn.GroupNorm(8, feature_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, feat_ir, feat_vis, text_embed):
        """
        Args:
            feat_ir: 红外特征 [B, C, H, W]
            feat_vis: 可见光特征 [B, C, H, W]
            text_embed: 文本特征 [B, 768]
        Returns:
            融合后的特征 [B, C, H, W]
        """
        # 1. CBAM增强
        feat_ir = self.cbam(feat_ir)
        feat_vis = self.cbam(feat_vis)

        # 2. 交叉注意力
        feat_ir_enhanced, feat_vis_enhanced = self.cross_attention(feat_ir, feat_vis)

        # 3. 文本条件注入
        feat_ir_cond = self.affine(feat_ir_enhanced, text_embed)
        feat_vis_cond = self.affine(feat_vis_enhanced, text_embed)

        # 4. 特征融合
        fused = torch.cat([feat_ir_cond, feat_vis_cond], dim=1)
        fused = self.fusion_conv(fused)

        return fused


# ============================================================================
# Part 2: 改进的 TransformerUNet
# ============================================================================

class ImprovedTransformerUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ch=64, ch_mult=[1, 2, 2, 4],
                 embed_dim=16, context_dim=768, use_fusion=True,
                 ffn_expansion_factor=2.0, bias=False, LayerNorm_type='WithBias'):
        """
        改进的TransformerUNet，支持多模态融合和文本条件

        Args:
            use_fusion: 是否使用融合模块（双模态输入时启用）
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fusion = use_fusion
        self.context_dim = context_dim

        # CLIP模型
        self.clip_model, _ = clip.load("ViT-L/14@336px", device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.depth = len(ch_mult)
        self.init_conv = OverlapPatchEmbed(in_c=in_ch, embed_dim=ch)

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.fusion_modules = nn.ModuleList([])  # 每层的融合模块
        self.affine_modules = nn.ModuleList([])  # 每层的文本条件注入

        ch_mult = [1] + ch_mult

        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i + 1]

            # Encoder
            self.encoder.append(
                nn.ModuleList([
                    TransformerBlock(dim_in=dim_in, dim_out=dim_in,
                                     num_heads=max(1, dim_in // 64),
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias, LayerNorm_type=LayerNorm_type),
                    TransformerBlock(dim_in=dim_in, dim_out=dim_in,
                                     num_heads=max(1, dim_in // 64),
                                     ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias, LayerNorm_type=LayerNorm_type),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))) if i == (self.depth - 1) else Identity(),
                    Downsample(dim_in, dim_out) if i != (self.depth - 1) else default_conv(dim_in, dim_out)
                ])
            )

            # Decoder
            self.decoder.insert(0, nn.ModuleList([
                TransformerBlock(dim_in=dim_out + dim_in, dim_out=dim_out,
                                 num_heads=max(1, dim_out // 64),
                                 ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type),
                TransformerBlock(dim_in=dim_out + dim_in, dim_out=dim_out,
                                 num_heads=max(1, dim_out // 64),
                                 ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if i == 0 else Identity(),
                Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
            ]))

            # 文本条件注入模块（每层都有）
            self.affine_modules.append(
                FeatureWiseAffine(context_dim, dim_in, use_affine_level=True)
            )

            # 融合模块（如果启用双模态）
            if use_fusion:
                self.fusion_modules.append(
                    FusionModule(dim_in, context_dim)
                )

        mid_dim = ch * ch_mult[-1]
        self.latent_conv = default_conv(mid_dim, embed_dim, kernel_size=1)
        self.post_latent_conv = default_conv(embed_dim, mid_dim, kernel_size=1)

        # Bottleneck的文本条件注入
        self.latent_affine = FeatureWiseAffine(context_dim, embed_dim, use_affine_level=True)

        self.final_conv = nn.Conv2d(ch, out_ch, 3, 1, 1)

    @torch.no_grad()
    def get_text_feature(self, text):
        """获取文本CLIP特征"""
        if isinstance(text, str):
            text = [text]
        text_tokens = clip.tokenize(text).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        return text_features.float()

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth - 1))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def encode(self, x, context):
        """编码过程，注入文本条件"""
        h, w = x.shape[2:]
        self.H, self.W = h, w
        x = self.check_image_size(x, h, w)
        x = self.init_conv(x)

        skip_connections = [x]

        for i, (b1, b2, attn, downsample) in enumerate(self.encoder):
            # 文本条件注入
            x = self.affine_modules[i](x, context)

            x = b1(x)
            skip_connections.append(x)

            x = b2(x)
            x = attn(x)
            skip_connections.append(x)

            x = downsample(x)

        # Latent空间的文本条件
        x = self.latent_conv(x)
        x = self.latent_affine(x, context)

        return x, skip_connections

    def decode(self, x, skip_connections):
        """解码过程"""
        x = self.post_latent_conv(x)

        for i, (b1, b2, attn, upsample) in enumerate(self.decoder):
            x = torch.cat([x, skip_connections[-(i * 2 + 1)]], dim=1)
            x = b1(x)

            x = torch.cat([x, skip_connections[-(i * 2 + 2)]], dim=1)
            x = b2(x)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x + skip_connections[0])
        return x[..., :self.H, :self.W]

    def forward(self, x, context=None, x_ir=None, x_vis=None):
        """
        前向传播

        Args:
            x: 单模态输入 [B, 3, H, W] 或 None
            context: 文本特征 [B, 768] 或文本字符串
            x_ir: 红外图像 [B, 3, H, W] (双模态模式)
            x_vis: 可见光图像 [B, 3, H, W] (双模态模式)
        """
        # 处理文本条件
        if context is None:
            context = torch.zeros(x.shape[0], self.context_dim).to(x.device)
        elif isinstance(context, str) or isinstance(context, list):
            context = self.get_text_feature(context)

        # 双模态融合模式
        if self.use_fusion and x_ir is not None and x_vis is not None:
            # 分别编码两个模态
            latent_ir, skips_ir = self.encode(x_ir, context)
            latent_vis, skips_vis = self.encode(x_vis, context)

            # 在每个尺度融合特征
            fused_skips = []
            for i, (skip_ir, skip_vis) in enumerate(zip(skips_ir, skips_vis)):
                if i < len(self.fusion_modules):
                    fused = self.fusion_modules[i](skip_ir, skip_vis, context)
                else:
                    fused = (skip_ir + skip_vis) / 2
                fused_skips.append(fused)

            # 融合latent
            fused_latent = (latent_ir + latent_vis) / 2

            # 解码
            return self.decode(fused_latent, fused_skips)

        # 单模态模式
        else:
            latent, skips = self.encode(x, context)
            return self.decode(latent, skips)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== 测试单模态模式 ===")
    model_single = ImprovedTransformerUNet(
        in_ch=3, out_ch=3, ch=64, ch_mult=[1, 2, 2, 4],
        embed_dim=16, context_dim=768, use_fusion=False
    ).to(device)

    x = torch.randn(2, 3, 256, 256).to(device)
    text = ["a clear image", "enhance the details"]

    output = model_single(x, context=text)
    print(f"输入: {x.shape} -> 输出: {output.shape}")

    print("\n=== 测试双模态融合模式 ===")
    model_fusion = ImprovedTransformerUNet(
        in_ch=3, out_ch=3, ch=64, ch_mult=[1, 2, 2, 4],
        embed_dim=16, context_dim=768, use_fusion=True
    ).to(device)

    x_ir = torch.randn(2, 3, 256, 256).to(device)
    x_vis = torch.randn(2, 3, 256, 256).to(device)
    text = ["fuse infrared and visible images"]

    output_fused = model_fusion(x=None, context=text, x_ir=x_ir, x_vis=x_vis)
    print(f"红外: {x_ir.shape} + 可见光: {x_vis.shape} -> 融合: {output_fused.shape}")

    print("\n✓ 所有测试通过！")