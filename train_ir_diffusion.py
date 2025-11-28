import argparse
import copy
import os
from email.policy import strict

from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
import itertools
from diffusion_fusion.nn_util import EMA
from diffusion_fusion.resample import create_named_schedule_sampler  # uniform 从指定的次数中抽样出来一定数量的步数
from diffusion_fusion.util import GET_TrainDataset, \
    to_numpy_image  # 从BCHW 的tensor的形式转换为 NHWC 的numpy图像范围也是从 -1 1 到 0-255
from diffusion_fusion.script_util import (add_dict_to_argparser, args_to_dict,  # 将args 转换为dict
                                          create_model_and_diffusion,  # 将dict的内容添加到argparse 中
                                          model_and_diffusion_defaults)

from AE.ae_arch import TransformerUNet


def load_ae():
    model = TransformerUNet()
    pretrained_path = 'pretrained/ae.pth'
    model.load_state_dict(torch.load(pretrained_path),strict=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model



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
    parser.add_argument('--log_interval', type=int, default=50)  # iter
    parser.add_argument('--save_interval', type=int, default=3000)
    parser.set_defaults(timestep_respacing="1000")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)  # ./log_diffusion
    # """TrainData Dataloader"""
    # LR_path = "./data/train_diffusion/LQ/"
    # HR_path = "./data/train_diffusion/HQ/"
    # Train_Dataset = GET_TrainDataset(HR_path, LR_path, MAX_SIZE=512, CROP_SIZE=256)# 并不需要重写，将之前的ControlFusionDataset 内容复用
    #
    # # LR_Y,H_R_Y,CB,CR  # 用于红外图像的去噪的扩散模型的操作；条件扩散的操作；
    #
    # train_loader = DataLoader(
    #     Train_Dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=False,
    # )

    from complex_degradation.diffusion_dataset import get_my_train_dataloaders, get_my_test_dataloaders
    # 图像读取过来之后应该都是简单的 0-1 之间的浮点数内容；
    # 随后应该修改为 -1 到 1 之间的内容；
    # 其实更多的应该是val 部分的代码你内容；

    opt_path = 'option/train/train_diffusion.yaml'
    from complex_degradation.diffusion_dataset import load_yaml_config
    opt = load_yaml_config(opt_path)

    train_loader, val_loader = get_my_train_dataloaders(opt['dataset'])  # max_size 640  crop_size 480
    # 我需要将切分出来的图像块的维度设定为 512

    # encoder 之后的内容为 B 16 60 60  我们设定的embed_dim 内容就是16

    ae = load_ae()

    """Model loader"""
    diffusion_stage1, diffusion_stage2, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )  # 构建diffusion的时候就已经是确定了；
    diffusion_stage1 = diffusion_stage1.to(args.device)  # cuda:0
    diffusion_stage2 = diffusion_stage2.to(args.device)
    diffusion_stage1.train()
    diffusion_stage2.train()
    # 将多个迭代器无缝整合为同一个迭代器的操作； 合并为同一个参数迭代器的操作；
    optimizer = AdamW(itertools.chain(diffusion_stage1.parameters(), diffusion_stage2.parameters()), lr=2e-5,
                      weight_decay=0.0)
    ema = EMA(args.ema_rate)  # update_model_average  update_average 更新模型参数/更新数值 （按照一定的比例）
    diffusion_stage1_ema = copy.deepcopy(diffusion_stage1)
    diffusion_stage2_ema = copy.deepcopy(diffusion_stage2)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    """Begin Train diffusion"""
    step = 0
    it = iter(train_loader)
    while step < args.max_iterations:
        try:
            # LR_Y HR_Y
            # batch, batch1, *rest = next(it)
            # 使用list的形式保留多余的内容；
            vis, vis_gt, ir, ir_gt, text, vis_name, ir_name = next(it)
            batch, batch1 = vis, vis_gt
        except StopIteration:
            it = iter(train_loader)
            # 一个epoch 的数据被读取完成之后，出发这里的错误重新创建dataloader 迭代器，从头开始读取batch
            # batch, batch1, *rest = next(it) # LQ GT 对应的内容；
            vis, vis_gt, ir, ir_gt, text, vis_name, ir_name = next(it)
            batch, batch1 = vis, vis_gt


        with torch.no_grad():
            batch, _ = ae.encode(batch)  # batch LQ and HQ 内容；
            batch1, _ = ae.encode(batch1)
            print(f'batch shape after ae encode: {batch.shape}')
            print(f'batch1 shape after ae encode: {batch1.shape}')
            batch = batch * 2 - 1 # 转换范围到 -1 1 之间
            batch1 = batch1 * 2 - 1

        # B 16 480/8 480/8 维度下降了8倍

        # batch LQ and HQ 内容；
        # 因为按照原始的形式是 B 1 H W 的状态的；
        '''
        现在的范围是从 -1 1 并且全都分离为了 YCbCr的状态 
        需要重新映射到指定的范围中提取操作； 
        '''

        batch = batch.to(args.device)  # LQ_Y
        cond = {'condition': batch1.to(args.device)}  # GT_Y 清晰的Y分量内容作为扩散的条件内容；
        optimizer.zero_grad()  # LR HR  HR 内容是作为cond 出现在这里的 LQ HQ
        # 0-len(抽取的权重数量）
        t, weights = schedule_sampler.sample(batch.shape[0], args.device)  # 抽样生成时间步长 ts 抽样得到的是0-19 之间的数字内容；
        losses = diffusion.training_losses(diffusion_stage1, diffusion_stage2, batch, t, model_kwargs=cond)
        loss = (losses["loss"] * weights).mean()
        # 忘记了一个事情

        # 对应的维度就是batch_size 数量的内容；
        # 计算出来随机的对应的t个位置上的内容；
        loss.backward()
        optimizer.step()
        ema.update_model_average(diffusion_stage1_ema, diffusion_stage1)
        ema.update_model_average(diffusion_stage2_ema, diffusion_stage2)
        # ema 模型的参数平滑处理；

        print({"loss": loss.detach().item()})
        # if (step + 1) % args.log_interval == 0:
        #     print({"loss": loss.detach().item()})


        if (step+1)% args.val_interval==0:
            print(f'现在开始进行模型的验证的操作；')
            # from metrics.psnr_metric import batch_PSNR, batch_SSIM
            # 问题是扩散模型val的时候是需要响应的前向内容存在的；
            # 这个很不好了


        # save model
        if (step) % args.save_interval == 0:
            ckpt_dir = os.path.join(args.save_dir, "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_dir_ema = os.path.join(args.save_dir, "checkpoint_ema")
            os.makedirs(ckpt_dir_ema, exist_ok=True)
            model_save_path1 = os.path.join(ckpt_dir, f'diffusion_stage1_iter_{step}.pt')
            model_save_path2 = os.path.join(ckpt_dir, f'diffusion_stage2_iter_{step}.pt')
            model_save_path1_ema = os.path.join(ckpt_dir_ema, f'diffusion_stage1_iter_{step}.pt')
            model_save_path2_ema = os.path.join(ckpt_dir_ema, f'diffusion_stage2_iter_{step}.pt')
            torch.save(diffusion_stage1.state_dict(), model_save_path1)
            torch.save(diffusion_stage2.state_dict(), model_save_path2)
            torch.save(diffusion_stage1_ema.state_dict(), model_save_path1_ema)
            torch.save(diffusion_stage2_ema.state_dict(), model_save_path2_ema)

        step += 1

    print("End of training")
    # 验证的时候的逻辑和内容状态也还是未知的本质上；


if __name__ == "__main__":
    main()
