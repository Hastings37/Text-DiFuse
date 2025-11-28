import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
# from basicivif.utils.registry import METRIC_REGISTRY


def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            perms = list(range(win.ndim))
            perms[2 + i] = perms[-1]
            perms[-1] = 2 + i
            out = conv(out, weight=win.permute(perms), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, K=(0.01, 0.03)):
    K1, K2 = K
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.type_as(X)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim_tensor(X, Y, data_range=255, size_average=True, win_size=11,
                win_sigma=1.5, win=None, K=(0.01, 0.03), nonnegative_ssim=False):
    """
    直接处理tensor的SSIM函数

    Args:
        X, Y: [H, W] 或 [C, H, W] 或 [B, C, H, W] 的tensor, 值域 0-255
        data_range: 数据范围，默认255

    Returns:
        SSIM值
    """
    # 确保是4D tensor [B, C, H, W]
    if X.dim() == 2:  # [H, W]
        X = X.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        Y = Y.unsqueeze(0).unsqueeze(0)
    elif X.dim() == 3:  # [C, H, W]
        X = X.unsqueeze(0)  # [1, C, H, W]
        Y = Y.unsqueeze(0)

    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")

    if win is not None:
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
        win = win.to(X.device)

    ssim_per_channel, _ = _ssim(X, Y, data_range=data_range, win=win, K=K)

    if nonnegative_ssim:
        ssim_per_channel = F.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(dim=1)


def ms_ssim_tensor(X, Y, data_range=255, size_average=True, win_size=11,
                   win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)):
    """
    直接处理tensor的MS-SSIM函数

    Args:
        X, Y: [H, W] 或 [C, H, W] 或 [B, C, H, W] 的tensor, 值域 0-255

    Returns:
        MS-SSIM值
    """
    # 确保是4D tensor [B, C, H, W]
    if X.dim() == 2:
        X = X.unsqueeze(0).unsqueeze(0)
        Y = Y.unsqueeze(0).unsqueeze(0)
    elif X.dim() == 3:
        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)

    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")

    avg_pool = F.avg_pool2d

    if win is not None:
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4), \
        f"Image size should be larger than {(win_size - 1) * (2 ** 4)} due to 4 downsamplings in ms-ssim"

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.tensor(weights, dtype=X.dtype, device=X.device)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))
        win = win.to(X.device)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, K=K)

        if i < levels - 1:
            mcs.append(F.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = F.relu(ssim_per_channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.reshape((-1, 1, 1)), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(dim=1)


# 修正后的评估函数
# @METRIC_REGISTRY.register()
def SSIM(ir, vis, fuse):
    """
    计算SSIM指标

    Args:
        ir: [H, W] tensor, 值域 0-255
        vis: [H, W] tensor, 值域 0-255
        fuse: [H, W] tensor, 值域 0-255

    Returns:
        dict: 包含 SSIM_ir, SSIM_vis, SSIM_total
    """
    # 问题1修正: tensor没有to_numpy()方法，应该用cpu().numpy()
    # 问题2修正: 但实际上不需要转换，直接用tensor计算

    # 确保在同一设备上
    device = ir.device
    vis = vis.to(device)
    fuse = fuse.to(device)

    # 直接使用tensor计算
    ssim_ir = ssim_tensor(ir, fuse, data_range=255).item()
    ssim_vis = ssim_tensor(vis, fuse, data_range=255).item()
    ssim_total = (ssim_ir + ssim_vis) / 2

    return {
        'SSIM_ir': ssim_ir,
        'SSIM_vis': ssim_vis,
        'SSIM_total': ssim_total
    }

# @METRIC_REGISTRY.register()
def MS_SSIM(ir, vis, fuse):
    """
    计算MS-SSIM指标

    Args:
        ir: [H, W] tensor, 值域 0-255
        vis: [H, W] tensor, 值域 0-255
        fuse: [H, W] tensor, 值域 0-255

    Returns:
        dict: 包含 MS_SSIM_ir, MS_SSIM_vis, MS_SSIM_total
    """
    # 确保在同一设备上
    device = ir.device
    vis = vis.to(device)
    fuse = fuse.to(device)

    # 检查图像尺寸是否满足MS-SSIM要求
    min_size = min(ir.shape[-2:])
    required_size = (11 - 1) * (2 ** 4)  # 默认win_size=11, 4次下采样

    if min_size <= required_size:
        # 如果图像太小，使用普通SSIM
        warnings.warn(
            f"Image size {min_size} is too small for MS-SSIM (requires > {required_size}). "
            f"Using regular SSIM instead."
        )
        return SSIM(ir, vis, fuse)

    # 直接使用tensor计算
    ms_ssim_ir = ms_ssim_tensor(ir, fuse, data_range=255).item()
    ms_ssim_vis = ms_ssim_tensor(vis, fuse, data_range=255).item()
    ms_ssim_total = (ms_ssim_ir + ms_ssim_vis) / 2

    return {
        'MS_SSIM_ir': ms_ssim_ir,
        'MS_SSIM_vis': ms_ssim_vis,
        'MS_SSIM_total': ms_ssim_total
    }


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    print("=== 测试SSIM指标 ===")
    ir = torch.randint(0, 256, (256, 256), dtype=torch.float32)
    vis = torch.randint(0, 256, (256, 256), dtype=torch.float32)
    fuse = (ir + vis) / 2

    # 测试SSIM
    ssim_result = SSIM(ir, vis, fuse)
    print(f"SSIM结果: {ssim_result}")

    # 测试MS-SSIM
    ms_ssim_result = MS_SSIM(ir, vis, fuse)
    print(f"MS-SSIM结果: {ms_ssim_result}")

    # 测试小图像
    print("\n=== 测试小图像 ===")
    small_ir = torch.randint(0, 256, (64, 64), dtype=torch.float32)
    small_vis = torch.randint(0, 256, (64, 64), dtype=torch.float32)
    small_fuse = (small_ir + small_vis) / 2

    ms_ssim_small = MS_SSIM(small_ir, small_vis, small_fuse)
    print(f"小图像MS-SSIM结果: {ms_ssim_small}")

    # 测试GPU
    if torch.cuda.is_available():
        print("\n=== 测试GPU ===")
        ir_gpu = ir.cuda()
        vis_gpu = vis.cuda()
        fuse_gpu = fuse.cuda()

        ssim_gpu = SSIM(ir_gpu, vis_gpu, fuse_gpu)
        print(f"GPU SSIM结果: {ssim_gpu}")