import copy
import argparse
import os
import torch
import json  # 用于保存结果
import numpy as np
from tqdm import tqdm  # 建议安装 tqdm 显示进度: pip install tqdm

from complex_degradation.diffusion_dataset import get_my_train_dataloaders
from diffusion_fusion.script_util import (add_dict_to_argparser, model_and_diffusion_defaults)
from utils.yaml_utils import load_ae

def main():
    """Setup"""
    torch.manual_seed(0)
    defaults = model_and_diffusion_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--Task_type', type=str, default='VIS-IR')
    parser.add_argument('--batch_size', type=int, default=4)  # 建议根据显存适当调大 batch_size 加速
    parser.add_argument('--num_workers', type=int, default=4)

    # 解析参数
    # 注意：这里稍微处理一下 unknown args，防止因为脚本参数冲突报错
    args, unknown = parser.parse_known_args()

    opt_path = 'option/train/train_diffusion.yaml'
    from complex_degradation.diffusion_dataset import load_yaml_config
    opt = load_yaml_config(opt_path)

    # 获取 Dataloader
    # 注意：统计时不建议 shuffle，虽然不影响结果，但顺序读取通常略快
    train_loader, val_loader = get_my_train_dataloaders(opt['dataset'])

    # 定义 42 种退化列表
    degradation_list = [
        "ir_low_contrast_slight", "ir_low_contrast_moderate", "ir_low_contrast_average", "ir_low_contrast_extreme",
        "ir_noise_slight", "ir_noise_moderate", "ir_noise_average", "ir_noise_extreme",
        "ir_stripe_noise_slight", "ir_stripe_noise_moderate", "ir_stripe_noise_average", "ir_stripe_noise_extreme",
        "vi_blur_slight", "vi_blur_moderate", "vi_blur_average", "vi_blur_extreme",
        "vi_haze_slight", "vi_haze_moderate", "vi_haze_average", "vi_haze_extreme",
        "vi_haze_low",
        "vi_low_light_slight", "vi_low_light_moderate", "vi_low_light_average", "vi_low_light_extreme",
        "vi_noise_slight", "vi_noise_moderate", "vi_noise_average", "vi_noise_extreme",
        "vi_noise_low",
        "vi_over_exposure_slight", "vi_over_exposure_moderate", "vi_over_exposure_average", "vi_over_exposure_extreme",
        "vi_rain_slight", "vi_rain_moderate", "vi_rain_average", "vi_rain_extreme",
        "vi_rain_haze",
        "llsn", "oelc", "rhrn"
    ]

    # ================= 初始化统计容器 =================
    # 我们需要存储 sum(x), sum(x^2), count(n) 来计算 mean 和 var
    stats_accumulator = {}

    # 四种图像类型
    img_types = ['vis', 'vis_gt', 'ir', 'ir_gt']

    for deg in degradation_list:
        stats_accumulator[deg] = {}
        for img_type in img_types:
            stats_accumulator[deg][img_type] = {
                'sum': 0.0,
                'sum_sq': 0.0,
                'count': 0
            }

    # 加载 AutoEncoder
    print("Loading Autoencoder...")
    ae = load_ae().to(args.device)
    ae.eval()  # 必须开启 eval 模式

    print("Start Statistical Calculation...")

    # 使用 no_grad 加速并节省显存
    with torch.no_grad():
        # 使用 tqdm 包装 loader 显示进度条
        for step, input_data in enumerate(tqdm(train_loader)):
            vis, vis_gt, ir, ir_gt, type_names, text, vis_name, ir_name = input_data

            # 1. 移到 GPU
            vis = vis.to(args.device)
            vis_gt = vis_gt.to(args.device)
            ir = ir.to(args.device)
            ir_gt = ir_gt.to(args.device)

            # 2. Batch Encode (关键修改：移出内层循环，一次处理整个 Batch)
            # 假设 ae.encode 返回 (latent, hidden_state)，我们需要 latent
            cond_vis, _ = ae.encode(vis)
            cond_vis_gt, _ = ae.encode(vis_gt)
            cond_ir, _ = ae.encode(ir)
            cond_ir_gt, _ = ae.encode(ir_gt)

            # 3. 遍历 Batch 中的每一个样本进行统计
            batch_size_current = vis.shape[0]

            for b in range(batch_size_current):
                deg_type = type_names[b]  # 获取当前样本的退化类型

                # 确保这个类型在我们的列表里（防止数据加载器有一些未知类型）
                if deg_type not in stats_accumulator:
                    print(f"Warning: Unknown degradation type '{deg_type}' encountered. Skipping.")
                    continue

                # 定义一个临时字典方便循环处理
                current_latents = {
                    'vis': cond_vis[b],
                    'vis_gt': cond_vis_gt[b],
                    'ir': cond_ir[b],
                    'ir_gt': cond_ir_gt[b]
                }

                # 4. 累加统计量
                for img_type, tensor in current_latents.items():
                    # 计算当前样本的 sum, sum_sq, count
                    # .item() 将 tensor 转换为 python float，从计算图中分离，防止显存泄漏
                    s = tensor.sum().item()
                    s_sq = (tensor ** 2).sum().item()
                    c = tensor.numel()  # 元素总数 (C * H * W)

                    # 更新累加器
                    stats_accumulator[deg_type][img_type]['sum'] += s
                    stats_accumulator[deg_type][img_type]['sum_sq'] += s_sq
                    stats_accumulator[deg_type][img_type]['count'] += c

    print("Calculation finished. Computing final Mean and Std/Var...")

    # ================= 最终计算与结果保存 =================
    final_stats = {}

    for deg in degradation_list:
        final_stats[deg] = {}
        for img_type in img_types:
            accum = stats_accumulator[deg][img_type]

            total_count = accum['count']

            if total_count > 0:
                # 公式: Mean = Sum / N
                mean = accum['sum'] / total_count

                # 公式: Var = E[X^2] - (E[X])^2 = (SumSq / N) - Mean^2
                mean_sq = accum['sum_sq'] / total_count
                var = mean_sq - (mean ** 2)

                # 处理数值计算误差导致的微小负数
                if var < 0:
                    var = 0.0

                std = np.sqrt(var)

                # 记录结果 (保留 float 格式)
                final_stats[deg][img_type] = {
                    'mean': float(mean),
                    'var': float(var),
                    'std': float(std),
                    'min': float(mean - 3 * std),  # 简单的 3-sigma 估计范围，仅供参考
                    'max': float(mean + 3 * std)
                }
            else:
                print(f"Warning: No data found for {deg} - {img_type}")
                final_stats[deg][img_type] = None


    import yaml
    output_file = 'latent_mean_var.yaml'
    with open(output_file, 'w') as f:
        # default_flow_style=False 确保输出为块状格式（易读），而不是行内格式
        # sort_keys=False 保持字典原本的插入顺序（mean, var, std...）
        yaml.dump(final_stats, f, default_flow_style=False, sort_keys=False)

    print(f"Statistics saved to {output_file}")

    # 打印一个示例看看
    if len(degradation_list) > 0:
        example_key = degradation_list[0]
        print(f"\nExample stats for {example_key}:")
        # 使用 yaml.dump 将字典转为字符串打印
        print(yaml.dump(final_stats[example_key], default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()