import argparse
import copy
import os
import itertools
import sys  # <--- 新增：用于系统输入输出控制
import time # <--- 新增：用于获取时间戳

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from diffusion_fusion.nn_util import EMA
from diffusion_fusion.resample import create_named_schedule_sampler
from diffusion_fusion.util import GET_TrainDataset, to_numpy_image
from diffusion_fusion.script_util import (add_dict_to_argparser, args_to_dict,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults)

from complex_degradation.diffusion_dataset import get_my_train_dataloaders, get_my_test_dataloaders

from utils.yaml_utils import (
    yaml_to_config,
    dict_to_yaml,
    load_ae,
    common_to_latent,
    latent_to_common,
    calculate_psnr,
    calculate_ssim
)

# ### <--- 新增：Logger 类定义，用于同时写文件和终端
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 确保实时写入文件，防止程序崩溃时丢失日志

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    """Setup"""
    defaults = model_and_diffusion_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='./log_diffusion')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ema_rate', type=float, default=0.9999)
    parser.add_argument('--schedule_sampler', type=str, default="uniform")
    parser.add_argument('--max_iterations', type=int, default=1500000)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=3000)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.set_defaults(timestep_respacing="1000")
    args = parser.parse_args()

    # 1. 先创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    val_save_dir = os.path.join(args.save_dir, "val_images")
    os.makedirs(val_save_dir, exist_ok=True)

    # ### <--- 新增：配置 Logger 功能
    # 获取当前时间标签，格式：YYYYMMDD_HHMMSS
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_name = f'train_{timestamp}.log'
    log_path = os.path.join(args.save_dir, log_name)

    # 重定向 stdout (print) 和 stderr (报错信息)
    sys.stdout = Logger(log_path, sys.stdout)
    sys.stderr = Logger(log_path, sys.stderr)

    print(f"Log file created at: {log_path}")
    print(f"Arguments: {args}") # 顺便记录一下运行参数
    # ### <--- Logger 配置结束


    opt_path = 'option/train/train_vis_diffusion.yaml'
    opt = yaml_to_config(opt_path)
    train_loader, val_loader = get_my_train_dataloaders(opt['dataset'])

    ae = load_ae()

    """Model loader"""
    diffusion_stage1, diffusion_stage2, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion_stage1 = diffusion_stage1.to(args.device)
    diffusion_stage2 = diffusion_stage2.to(args.device)
    diffusion_stage1.train()
    diffusion_stage2.train()

    optimizer = AdamW(itertools.chain(diffusion_stage1.parameters(), diffusion_stage2.parameters()),
                      lr=2e-5, weight_decay=0.0)

    ema = EMA(args.ema_rate)
    diffusion_stage1_ema = copy.deepcopy(diffusion_stage1)
    diffusion_stage2_ema = copy.deepcopy(diffusion_stage2)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    """Begin Train diffusion"""
    print(f"Start training on {args.device}...")
    step = 0
    it = iter(train_loader)

    while step < args.max_iterations:
        try:
            vis, vis_gt, ir, ir_gt, type_name, text, vis_name, ir_name = next(it)
        except StopIteration:
            it = iter(train_loader)
            vis, vis_gt, ir, ir_gt, type_name, text, vis_name, ir_name = next(it)

        # batch 为干净的内容；
        cond,cond_hs=ae.encode(vis_gt.to(args.device))
        # batch1 为存在退化的内容；
        cond1,cond1_hs=ae.encode(vis.to(args.device))

        batch = cond1 #干净的内容
        batch1 = {'condition': cond} # 带有噪声的内容

        optimizer.zero_grad()

        t, weights = schedule_sampler.sample(batch.shape[0], args.device)
        losses = diffusion.training_losses(diffusion_stage1, diffusion_stage2, batch, t, model_kwargs=batch1)
        loss = (losses["loss"] * weights).mean()

        loss.backward()
        optimizer.step()

        ema.update_model_average(diffusion_stage1_ema, diffusion_stage1)
        ema.update_model_average(diffusion_stage2_ema, diffusion_stage2)

        if step % args.log_interval == 0:
            # 这个 print 现在会自动写入 log 文件
            print(f"Step {step}: Loss = {loss.detach().item():.6f}")

        # ================= Validation Loop =================
        if (step + 1) % args.val_interval == 0:
            print(f'Starting Validation at step {step + 1}...')
            diffusion_stage1.eval()
            diffusion_stage2.eval()
            ae.eval()

            total_psnr = 0.0
            total_ssim = 0.0
            total_images = 0  # <--- 修改1：改为统计图片数量，而不是 val_count (batch数)
            target_val_images = 1
            # <--- 修改2：设定目标验证图片数量

            with torch.no_grad():
                for val_idx, val_data in enumerate(val_loader):
                    # <--- 修改3：基于图片数量判断退出
                    # 如果已经验证了超过50张，就停止
                    if total_images >= target_val_images:
                        break

                    val_vis, val_vis_gt, val_ir, val_ir_gt, val_type_name, _, _, _ = val_data
                    val_vis = val_vis.to(args.device)
                    val_vis_gt = val_vis_gt.to(args.device)

                    # 获取当前这个 batch 有几张图 (防止最后一个 batch 不足的情况)
                    current_batch_size = val_vis.size(0)

                    cond1,cond1_hs=ae.encode(val_vis.to(args.device))
                    sample_shape = cond1.shape
                    model_kwargs_val = {'condition': cond1}

                    pred_norm_latent = diffusion.p_sample_loop_single(
                        diffusion_stage1,
                        diffusion_stage2,
                        sample_shape,
                        noise=None,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=model_kwargs_val,
                        device=args.device,
                        progress=False
                    )

                    pred_img = ae.decode(pred_norm_latent,cond1_hs)
                    pred_img = torch.clamp(pred_img, 0, 1)
                    val_vis_gt = torch.clamp(val_vis_gt, 0, 1)

                    # 计算当前 Batch 的平均指标
                    # 注意：您的函数返回的是 Tensor，建议用 .item() 转为 python float，节省显存
                    batch_psnr = calculate_psnr(pred_img, val_vis_gt)
                    batch_ssim = calculate_ssim(pred_img, val_vis_gt)

                    # 确保转为 float，防止 calculate_psnr 返回 tensor 导致无法加权
                    if isinstance(batch_psnr, torch.Tensor):
                        batch_psnr = batch_psnr.item()
                    if isinstance(batch_ssim, torch.Tensor):
                        batch_ssim = batch_ssim.item()

                    # <--- 修改4：加权累加
                    # 公式：总分 += 平均分 * 数量
                    total_psnr += batch_psnr * current_batch_size
                    total_ssim += batch_ssim * current_batch_size
                    total_images += current_batch_size

                    # 保存第一张图片
                    if val_idx == 0:
                        save_tensor = torch.cat([val_vis[0], pred_img[0], val_vis_gt[0]], dim=2)
                        save_path = os.path.join(val_save_dir, f"val_{step + 1}.png")
                        save_image(save_tensor, save_path)
                        print(f"Saved validation image to {save_path}")

            if total_images > 0:
                # <--- 修改5：最后除以总图片数
                avg_psnr = total_psnr / total_images
                avg_ssim = total_ssim / total_images
                print(
                    f"Validation Step {step + 1} (on {total_images} images): PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
            else:
                print("Validation skipped (No data).")

            diffusion_stage1.train()
            diffusion_stage2.train()

        # ================= Save Model =================
        if (step + 1) % args.save_interval == 0 and step != 0:
            ckpt_dir = os.path.join(args.save_dir, "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_dir_ema = os.path.join(args.save_dir, "checkpoint_ema")
            os.makedirs(ckpt_dir_ema, exist_ok=True)

            model_save_path1 = os.path.join(ckpt_dir, f'diffusion_stage1_iter_{step + 1}.pt')
            model_save_path2 = os.path.join(ckpt_dir, f'diffusion_stage2_iter_{step + 1}.pt')
            model_save_path1_ema = os.path.join(ckpt_dir_ema, f'diffusion_stage1_iter_{step + 1}.pt')
            model_save_path2_ema = os.path.join(ckpt_dir_ema, f'diffusion_stage2_iter_{step + 1}.pt')

            torch.save(diffusion_stage1.state_dict(), model_save_path1)
            torch.save(diffusion_stage2.state_dict(), model_save_path2)
            torch.save(diffusion_stage1_ema.state_dict(), model_save_path1_ema)
            torch.save(diffusion_stage2_ema.state_dict(), model_save_path2_ema)

            print(f"Saved checkpoint at step {step + 1}")

        step += 1

    print("End of training")

if __name__ == "__main__":
    main()