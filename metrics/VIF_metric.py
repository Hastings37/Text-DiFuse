import torch
from sympy.abc import sigma

# from basicivif.utils.registry import METRIC_REGISTRY
import torch.nn.functional as F
'''
可视信息保真度
VIF（Visual Information Fidelity）是一种用于评估图像质量的指标，基于人类视觉系统的信息处理方式。它通过比较参考图像和失真图像在不同空间频率下的信息保真度来衡量图像质量。VIF的计算涉及以下几个关键步骤：
评估的是参考图像和失真图之间的“可视信息量”的比值 
观察者可以从失真图像中获取到的视觉信息占据源图像的内容的多少 

VIF数值越高代表其和参考图越是一致的，视觉信息越是接近 
'''
def fspecial_gaussian(size,sigma):
    x=torch.linspace(-size[0]//2,size[0]//2,size[0])
    y=torch.linspace(-size[1]//2,size[1]//2,size[1])
    x,y=torch.meshgrid(x,y)
    '''
    类似是在XY坐标系中构建起来的形式； 
    '''
    g=torch.exp(-(x**2+y**2)/(2*sigma**2))
    return g/g.sum()

def convolve2d(input,kernel):
    kernel=kernel.unsqueeze(0).unsqueeze(0).to(input.device)
    # 对应边缘的填充设定为same填充的操作，保持其对应的输出尺寸和输入尺寸一样；
    return F.conv2d(input.unsqueeze(0).unsqueeze(0),kernel,padding=kernel.shape[-1]//2).squeeze()


# 没看明白这个实现的细节含义是什么 暂时不纠结这个。。。。。。。。。。。。。
def vifp_mscale(ref, dist):
    sigma_nsq = 2          # 噪声方差常数（模型中的观测噪声项），论文里固定经验值
    num = 0                # 分子累加器（跨尺度求和）
    den = 0                # 分母累加器（跨尺度求和）
    for scale in range(1, 5):                    # 4 个金字塔尺度：1,2,3,4
        N = 2 ** (4 - scale + 1) + 1             # 高斯窗大小：scale=1→17, 2→9, 3→5, 4→3
        win = fspecial_gaussian((N, N), N / 5)   # 生成高斯权重窗口

        if scale > 1:                # 从第二层起：先做平滑再下采样 2×
            ref = convolve2d(ref, win)
            dist = convolve2d(dist, win)
            ref = ref[::2, ::2]      # 双线性金字塔的等效：高斯低通 + 2 倍降采样
            dist = dist[::2, ::2]

        # 估计局部均值（低频）
        mu1 = convolve2d(ref, win)   # ref 的局部均值
        mu2 = convolve2d(dist, win)  # dist 的局部均值
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        # 估计局部方差与协方差（高频能量/结构）
        sigma1_sq = convolve2d(ref * ref, win) - mu1_sq    # Var(ref)
        sigma2_sq = convolve2d(dist * dist, win) - mu2_sq  # Var(dist)
        sigma12   = convolve2d(ref * dist, win) - mu1_mu2  # Cov(ref, dist)

        # 数值稳定：理论上非负，数值误差导致的负值截断为 0
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        # 线性退化模型参数：dist ≈ g * ref + v，g 为增益，sv_sq 为残差噪声方差
        g     = sigma12 / (sigma1_sq + 1e-10)          # g = Cov/Var
        sv_sq = sigma2_sq - g * sigma12                # sv^2 = Var(dist) - g*Cov

        # 退化情形的边界处理（避免除零和无意义的 g）
        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0

        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0

        # 若 g<0（非物理/回归异常），将 g 置 0，残差方差回退
        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0

        # 再次数值稳定，避免分母为 0
        sv_sq[sv_sq <= 1e-10] = 1e-10

        # 分子：log10(1 + SNR_after_degradation)，其中 SNR ~ g^2 * Var(ref) / (sv^2 + σ_n^2)
        num += torch.sum(torch.log10(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq)))
        # 分母：log10(1 + 原始信号 SNR)，SNR ~ Var(ref)/σ_n^2
        den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den   # 跨尺度加权后的总体 VIFP：∈[0, +∞)，越大代表失真越小/质量越好
    return vifp


# @METRIC_REGISTRY.register()
def VIF(ir,vis,fuse):
    vif=vifp_mscale(ir,fuse)+vifp_mscale(vis,fuse)
    return {'VIF':vif.detach().cpu().item()}
# 返回的是单个的数值；
