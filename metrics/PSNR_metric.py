# from basicivif.utils.registry import METRIC_REGISTRY
import torch
import math

# @METRIC_REGISTRY.register()
def PSNR(ir, vis, fuse):
    """
    计算图像融合任务的 PSNR (Peak Signal-to-Noise Ratio)

    Args:
        ir: torch.Tensor [H, W], 值域 0–255
        vis: torch.Tensor [H, W], 值域 0–255
        fuse: torch.Tensor [H, W], 值域 0–255

    Returns:
        dict:
            {
                'PSNR_ir': 与 IR 图的 PSNR,
                'PSNR_vis': 与 VIS 图的 PSNR,
                'PSNR_total': 平均 PSNR
            }
    """
    # 转为 float32 并归一化到 [0, 1]
    ir = ir.float() / 255.0
    vis = vis.float() / 255.0
    fuse = fuse.float() / 255.0

    # 确保张量在同一设备上
    device = fuse.device
    ir = ir.to(device)
    vis = vis.to(device)

    # 计算 MSE
    mse_ir = torch.mean((fuse - ir) ** 2)
    mse_vis = torch.mean((fuse - vis) ** 2)

    mse_avg = 0.5 * (mse_ir + mse_vis)

    # 避免 log(0)
    eps = 1e-10
    psnr_ir = 20 * torch.log10(1.0 / torch.sqrt(mse_ir + eps))
    psnr_vis = 20 * torch.log10(1.0 / torch.sqrt(mse_vis + eps))
    psnr_total = 20 * torch.log10(1.0 / torch.sqrt(mse_avg + eps))

    return {
        'PSNR_ir': psnr_ir.detach().cpu().item(),
        'PSNR_vis': psnr_vis.detach().cpu().item(),
        'PSNR_total': psnr_total.detach().cpu().item()
    }