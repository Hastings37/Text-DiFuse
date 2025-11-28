import os
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# ====== 按需导入你已有的 metric 函数 ======
from AG_metric import AG
from EN_metric import EN
from SD_metric import SD
from SF_metric import SF
from Qabf_metric import Qabf  # 假设函数名为 Qabf
from SSIM_metric import SSIM  # 假设函数名为 SSIM
from VIF_metric import VIF  # 假设函数名为 VIF
from MI_metric import MI  # 假设函数名为 MI
from PSNR_metric import PSNR  # 若有
from SCD_metric import SCD  # 若有

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_rgb_tensor(path: str, device: torch.device) -> torch.Tensor:
    """读取图像为 RGB tensor: [3,H,W], float32, [0,1]."""
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img) * 255.0 # 将图像转换到 [0,255] 范围
    t=t.to(device)  # [C,H,W], 0-1'
    return t


def collect_common_filenames(ir_dir, vis_dir, fuse_dir):
    ir_files = {f for f in os.listdir(ir_dir) if f.lower().endswith(IMG_EXTS)}
    vis_files = {f for f in os.listdir(vis_dir) if f.lower().endswith(IMG_EXTS)}
    fuse_files = {f for f in os.listdir(fuse_dir) if f.lower().endswith(IMG_EXTS)}

    common = sorted(ir_files & vis_files & fuse_files)
    if not common:
        raise RuntimeError("三个文件夹没有共同文件名，请检查路径和命名。")
    return common




def metric_dir(ir_dir,vis_dir,fuse_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    filenames = collect_common_filenames(ir_dir, vis_dir, fuse_dir)
    num_imgs = len(filenames)
    print(f"Found {num_imgs} matched images.")

    # 想统计的所有指标（键名与各 metric 返回的 key 对应）
    metric_names = [
        "AG", "EN", "SD", "SF",
        "Qabf", "SSIM", "VIF", "MI",
        # 如有再加："PSNR", "SCD"
    ]

    metric_sums = {name: 0.0 for name in metric_names}

    for idx, name in enumerate(filenames, 1):
        ir_path = os.path.join(ir_dir, name)
        vis_path = os.path.join(vis_dir, name)
        fuse_path = os.path.join(fuse_dir, name)

        ir = load_rgb_tensor(ir_path, device)  # [3,H,W]
        vis = load_rgb_tensor(vis_path, device)  # [3,H,W]
        fuse = load_rgb_tensor(fuse_path, device)  # [3,H,W]
        # Y = 0.299 * R + 0.587 * G + 0.114 * B
        ir = ir[0] * 0.299 + ir[1] * 0.587 + ir[2] * 0.114  # [H,W]
        vis = vis[0] * 0.299 + vis[1] * 0.587 + vis[2] * 0.114
        fuse = fuse[0] * 0.299 + fuse[1] * 0.587 + fuse[2] * 0.114

        # ===== 调用各个 metric 函数 =====
        # 约定：每个 metric 函数返回 dict，如 {'AG': value}
        metrics = {}

        # 只依赖 fused 的指标
        metrics.update(AG(fuse))
        metrics.update(EN(fuse))
        metrics.update(SD(fuse))
        metrics.update(SF(fuse))

        # 依赖 ir / vis / fused 的指标（你按自己实际接口调整）
        metrics.update(Qabf(ir, vis, fuse))
        metrics.update(SSIM(ir, vis, fuse))
        metrics.update(VIF(ir, vis, fuse))
        metrics.update(MI(ir, vis, fuse))
        metrics.update(PSNR(ir, vis, fuse))
        metrics.update(SCD(ir, vis, fuse))

        # ===== 累加 =====
        for k, v in metrics.items():
            if k not in metric_sums:
                # 如果某个 metric 返回了你没在 metric_names 里登记的 key，可以打印看一下
                print(f"[Warn] metric key '{k}' not in metric_sums, skip.")
                continue
            metric_sums[k] += float(v)

        if idx % 50 == 0 or idx == num_imgs:
            print(f"Processed {idx}/{num_imgs}")
        return metrics # dict 内容；

    # ===== 计算平均值 =====
    print("\n=== Average Metrics over {} images ===".format(num_imgs))
    for k in metric_names:
        avg = metric_sums[k] / num_imgs
        print(f"{k:>6}: {avg:.6f}")


def metric_batch_images(ir, vis, fuse):
    """
    计算一个 Batch 内图像的 IVIF 指标总和。

    Args:
        ir (torch.Tensor): 红外图像, [B, 3, H, W], 范围 0-1 或 0-255
        vis (torch.Tensor): 可见光图像, [B, 3, H, W], 范围 0-1 或 0-255
        fuse (torch.Tensor): 融合图像, [B, 3, H, W], 范围 0-1 或 0-255

    Returns:
        返回的内容是一个综合的数值内容；
        dict: 包含该 batch 所有图像各指标累加和的字典, 例如 {'AG': 120.5, 'VIF': 3.4 ...}
    """

    # 1. 开启无梯度模式，节省显存
    with torch.no_grad():

        # 2. 灰度转换 (RGB -> Gray)
        # 输入为 [B, 3, H, W]，转换公式: Y = 0.299*R + 0.587*G + 0.114*B
        # 结果维度变为 [B, H, W]
        ir_gray = ir[:, 0] * 0.299 + ir[:, 1] * 0.587 + ir[:, 2] * 0.114
        vis_gray = vis[:, 0] * 0.299 + vis[:, 1] * 0.587 + vis[:, 2] * 0.114
        fuse_gray = fuse[:, 0] * 0.299 + fuse[:, 1] * 0.587 + fuse[:, 2] * 0.114

        # 3. 数值范围归一化 (Ensure 0-255)
        # 假设如果最大值 <= 1.0，则认为是 0-1 范围，需要扩展到 0-255
        # 这一步是为了满足 AG, EN, SF 等传统指标通常基于 0-255 计算的要求
        if ir_gray.max() <= 1.0:
            ir_gray = ir_gray * 255.0
        if vis_gray.max() <= 1.0:
            vis_gray = vis_gray * 255.0
        if fuse_gray.max() <= 1.0:
            fuse_gray = fuse_gray * 255.0

        # 4. 初始化累加字典
        batch_size = ir.shape[0]
        # 定义你想统计的指标列表（需确保这些函数在上下文已定义）
        metric_keys = [
            "AG", "EN", "SD", "SF",
            "Qabf", "SSIM", "VIF", "MI",
            "PSNR", "SCD"  # 根据需要增减
        ]
        total_metrics = {k: 0.0 for k in metric_keys}

        # 5. 遍历 Batch (因为底层指标函数只支持单张 2D 输入)
        for b in range(batch_size):
            # 取出单张图像 [H, W]
            img_ir = ir_gray[b]
            img_vis = vis_gray[b]
            img_fuse = fuse_gray[b]

            # 单张图像的指标字典
            cur_metrics = {}

            # --- 仅依赖 Fused 的指标 ---
            # 假设这些函数返回的是 {'AG': val} 这种形式的字典，或者是直接返回数值
            # 这里沿用你 metric_dir 的逻辑：函数返回 dict
            try:
                cur_metrics.update(AG(img_fuse))
                cur_metrics.update(EN(img_fuse))
                cur_metrics.update(SD(img_fuse))
                cur_metrics.update(SF(img_fuse))

                # --- 依赖 IR/VIS/Fused 的指标 ---
                cur_metrics.update(Qabf(img_ir, img_vis, img_fuse))
                cur_metrics.update(SSIM(img_ir, img_vis, img_fuse))
                cur_metrics.update(VIF(img_ir, img_vis, img_fuse))
                cur_metrics.update(MI(img_ir, img_vis, img_fuse))
                cur_metrics.update(PSNR(img_ir, img_vis, img_fuse))
                cur_metrics.update(SCD(img_ir, img_vis, img_fuse))

            except Exception as e:
                # 为了防止某个指标计算出错导致训练中断，可以打印警告
                print(f"Warning: Metric calculation failed at index {b} in batch. Error: {e}")
                continue

            # 6. 累加到总字典
            for k, v in cur_metrics.items():
                if k in total_metrics:
                    # 注意：部分 Tensor 可能是 0-dim tensor，需转 float
                    total_metrics[k] += float(v)
                else:
                    # 如果指标函数返回了不在 metric_keys 里的新指标，也可以动态添加
                    total_metrics[k] = float(v)

    return total_metrics

def main(ir_dir, vis_dir, fuse_dir):
    metric_dir(ir_dir, vis_dir, fuse_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ir_dir", default=r'E:\IVIF_DataSets\IR_Low_contrast\IR_Low_contrast_slight\test\Infrared', help="IR 图像文件夹路径")
    parser.add_argument("--vis_dir",default=r'E:\IVIF_DataSets\IR_Low_contrast\IR_Low_contrast_slight\test\Visible', help="VIS 图像文件夹路径")
    parser.add_argument("--fuse_dir",default='../ir_low_constrast', help="融合图像文件夹路径")
    args = parser.parse_args()

    main(args.ir_dir, args.vis_dir, args.fuse_dir)
