import copy
import argparse
import os
import torch.nn.functional as F
import random
import torch
import cv2
import random
from torch.optim import AdamW
from PIL import Image
from torch.utils.data import DataLoader

from complex_degradation.diffusion_dataset import get_my_train_dataloaders
from diffusion_fusion.unet import Get_Fusion_Control_Model
from diffusion_fusion.resample import create_named_schedule_sampler
from diffusion_fusion.util import GET_TrainDataset, to_numpy_image, GET_TestDataset
from diffusion_fusion.script_util import (add_dict_to_argparser, args_to_dict,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults)

import copy
from utils.yaml_utils import load_ae, yaml_to_config,latent_to_common,common_to_latent



def main():
    mean_var_dict=yaml_to_config('./latent_mean_var.yaml')


    """Setup"""
    torch.manual_seed(0)
    defaults = model_and_diffusion_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='./log_FCM')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.set_defaults(timestep_respacing="25")  # 对应为反向扩散的时候使用的步数情况；
    args = parser.parse_args()

    opt_path = 'option/train/train_fcm_diffusion.yaml'

    opt = yaml_to_config(opt_path)

    train_loader, val_loader = get_my_train_dataloaders(opt['dataset'])

    os.makedirs(args.save_dir, exist_ok=True)

    ae = load_ae()
    ae = ae.to(args.device)

    """Model loader"""
    diffusion_stage1_vis, diffusion_stage2_vis, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    diffusion_stage1_ir = copy.deepcopy(diffusion_stage1_vis)
    diffusion_stage2_ir = copy.deepcopy(diffusion_stage2_vis)

    Fusion_Control_Model = Get_Fusion_Control_Model()

    # 我们需要重新加载两个路径；
    # diffusion_stage1_vis.load_state_dict(torch.load(diffusion_stage1_vis_path))
    # diffusion_stage2_vis.load_state_dict(torch.load(diffusion_stage2_vis_path))
    # diffusion_stage1_ir.load_state_dict(torch.load(diffusion_stage1_ir_path))
    # diffusion_stage2_ir.load_state_dict(torch.load(diffusion_stage2_ir_path))


    diffusion_stage1_vis = diffusion_stage1_vis.to(args.device)
    diffusion_stage2_vis = diffusion_stage2_vis.to(args.device)
    diffusion_stage1_ir = diffusion_stage1_ir.to(args.device)
    diffusion_stage2_ir = diffusion_stage2_ir.to(args.device)

    Fusion_Control_Model = Fusion_Control_Model.to(args.device)
    optimizer = AdamW(Fusion_Control_Model.parameters(), lr=2e-5, weight_decay=0.0)
    diffusion_stage1_vis.eval()
    diffusion_stage2_vis.eval()  # 模型的参数仍旧是可以训练的状态的；
    diffusion_stage1_ir.eval()
    diffusion_stage2_ir.eval()

    '''
    for p in model.parameters():
        p.requires_grad = False

    或者是： 

    with torch.no_grad(): 
        out=model(x) 
    '''
    Fusion_Control_Model.train()

    steps = 0

    """Begin Train FCM"""
    for epoch in range(args.epoch):
        print("Text-DiFuse....Begin Train Fusion.....")
        for i, input in enumerate(train_loader):
            steps += 1
            # 现在并不是在单一的一个通道维度上去进行扩散的训练了；
            vis, vis_gt, ir, ir_gt, type_name, text, vis_name, ir_name = input
            cond_vis,cond_hs_vis,cond_vis_gt,cond_vis_gt_hs=common_to_latent(ae,vis,vis_gt,type_name,mean_var_dict,'vis',args.device)
            cond_ir,cond_hs_ir,cond_ir_gt,cond_ir_gt_hs=common_to_latent(ae,ir,ir_gt,type_name,mean_var_dict,'ir',args.device)

            filename = vis_name # 这里是对应了多个的name的情况了；

            output_GT1 = cond_vis_gt
            output_GT2 = cond_ir_gt
            """Begin Train FCM"""
            indices = list(range(diffusion.num_timesteps))[::-1]  # 这里对应的长度就是 100了；
            torch.manual_seed(0)
            # 从开始的噪声内容开始的操作了；
            x_F_t = torch.randn(cond.shape, device=args.device)  # 从开始的完整的噪声内容 逐步得到我们需要的融合结果信息；
            number = random.randint(0, 24)  # 其中可以包含 0-24 的内容；
            condition_vis={'condition': cond_vis}
            condition_ir={'condition': cond_ir}
            # 这里训练的时候使用的 t 是一样的形式的；；
            for timestep in indices:  # 19-0  这里是在反向减小的操作的；
                # 构建一个batch_size 的时间步长全都是  0-19 的相同的内容；
                t = torch.tensor([timestep] * x_F_t.shape[0], device=args.device)
                if timestep > number:
                    out = diffusion.p_mean_variance_fuse(
                        diffusion_stage1_vis,
                        diffusion_stage2_vis,
                        diffusion_stage1_ir,
                        diffusion_stage2_ir,
                        Fusion_Control_Model,
                        x_F_t,
                        t,
                        model_kwargs=condition_vis,
                        model_kwargs1=condition_ir,
                    )
                    x_F_t = x_F_t.detach()  # 释放这一步和之前的计算图内容 ；
                    sample = out["mean"]
                    x_F_t = sample.detach()
                elif timestep == number:
                    optimizer.zero_grad()
                    loss, out = diffusion.train_FCM_loss(
                        diffusion_stage1,
                        diffusion_stage2,
                        Fusion_Control_Model,
                        x_F_t,
                        t,
                        output_GT1,
                        output_GT2,
                        model_kwargs=condition_vis,
                        model_kwargs1=condition_ir,
                    )  # 从这里去计算开始的位置上的内容实现；随后计算对应的损失内容；
                    # 问题是训练的时候使用的就是对应的低质量作为了引导的内容
                    # 要不然后面test 的时候无法获取到的啊；
                    loss.backward()
                    optimizer.step()
                else:
                    pass

            if (steps + 1) % args.val_interval == 0:
                metric_keys = [
                    "AG", "EN", "SD", "SF",
                    "Qabf", "SSIM", "VIF", "MI",
                    "PSNR", "SCD"  # 根据需要增减
                ]
                total_metrics = {k: 0.0 for k in metric_keys}

                for i, input in enumerate(test_loader):
                    # cond = {'condition': cond.to(args.device)}  # vis
                    # cond1 = {'condition': cond1.to(args.device)}  # ir
                    # filename = os.path.basename(str(input[4]))[:-3]
                    # 现在并不是在单一的一个通道维度上去进行扩散的训练了；
                    vis, vis_gt, ir, ir_gt, text, vis_name, ir_name = input
                    # 这里我们只用获取其对应的低质量的图像内容就行了对吧；
                    with torch.no_grad():
                        cond, cond_hs = ae.encode(ir.to(args.device))
                        cond1, cond1_hs = ae.encode(vis.to(args.device))
                        cond = cond * 2.0 - 1.0
                        cond1 = cond1 * 2.0 - 1.0
                        # 数据的范围应该是从 -1 到 1 之间的状态的；

                    condition = {'condition': cond.to(args.device)}  # ir Y 通道分量
                    condition1 = {'condition': cond1.to(args.device)}  # vis Y 通道分量
                    # filename = os.path.basename(str(input[4]))[:-3]
                    """GET GT"""
                    filename = vis_name

                    output = diffusion.p_sample_loop(  # 类似这样反向执行多次的Loop操作得到的；
                        diffusion_stage1,
                        diffusion_stage2,
                        Fusion_Control_Model,
                        cond.shape,
                        couple_single=True,
                        model_kwargs=condition,
                        model_kwargs1=condition1,
                        progress=True,
                    )
                    # 对应这个是融合图像的内容了；

                    output_vis = ae.decoder((output + 1) / 2.0, cond_hs)
                    output_ir = ae.decoder((output + 1) / 2.0, cond1_hs)
                    output = (output_vis + output_ir) / 2.0  # 简单平均一下其融合的结果情况；
                    # 将其内容进行融合性能的比较和测试的操作；
                    from metrics import metric
                    current_metrics = metric.metric_batch_images(ir_gt.to(args.device), vis_gt.to(args.device), output)
                    # 这些内容应该都是在对应的cuda 设备上的状态；
                    for k in metric_keys:
                        total_metrics[k] += current_metrics[k]

                    # 需要将其内容转换为图像保存
                    # 视觉上的内容应该也是没什么区别的吧我想来；
                    # 将一个指定的latent内容合并到后面的操作上去也是很不容易了；
                    output = output.detach().cpu().clamp(0, 1)
                    output = output.mul(255).round().byte()
                    output = output.permute(1, 2, 0).numpy()
                    output_name = filename
                    out_dir = os.path.join(args.save_dir, f'{epoch}', steps)
                    os.makedirs(out_dir, exist_ok=True)
                    for name in filename:
                        Image.fromarray(output).save(os.path.join(out_dir, filename + '.png'))

                # 计算平均指标
                num_test_images = len(test_loader)
                avg_metrics = {k: v / num_test_images for k, v in total_metrics.items()}
                print(f'Validation Metrics at step {steps + 1}:')
                for k, v in avg_metrics.items():
                    print(f'{k}: {v:.4f}')

    ckpt_dir = os.path.join(args.save_dir, args.Task_type + '_checkpoint')
    os.makedirs(ckpt_dir, exist_ok=True)
    FCM_save_path = os.path.join(ckpt_dir, f'FCM.pt')
    torch.save(Fusion_Control_Model.state_dict(), FCM_save_path)


if __name__ == "__main__":
    main()
