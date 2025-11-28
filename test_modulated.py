import copy
import argparse
import os
import torch.nn.functional as F
import random
import torch
import cv2
from torch.optim import AdamW
from PIL import Image
from torch.utils.data import DataLoader
from diffusion_fusion.unet import Get_Fusion_Control_Model
from diffusion_fusion.resample import create_named_schedule_sampler
from diffusion_fusion.util import GET_TestDataset, to_numpy_image
from diffusion_fusion.script_util import (add_dict_to_argparser, args_to_dict,
                              create_model_and_diffusion,
                              model_and_diffusion_defaults)

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from modulated.segment_anything.segment_anything import build_sam, SamPredictor
from modulated.util import load_owlvit,OWL_VIT_SAM,resize_and_align16_batch

def main():

    """Setup"""
    torch.manual_seed(0)
    defaults = model_and_diffusion_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='./result_modulated')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--text_prompt', type=str, default='people,car,bike')
    parser.set_defaults(timestep_respacing="25")  # 选择从其中抽样25个位置；
    args = parser.parse_args()

    """TestData Dataloader"""

    diffusion_stage1_path = "./pretrained/diffusion_stage1.pth"
    diffusion_stage2_path = "./pretrained/diffusion_stage2.pth"
    VIS_path = "./data/test/modulated/VIS/"
    IR_path = "./data/test/modulated/IR/"
    Test_Dataset = GET_TestDataset(VIS_path,IR_path,MAX_SIZE=640)
    FCM_path = "./pretrained/FCM-VIS-IR.pt"

    test_loader = DataLoader(
        Test_Dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    os.makedirs(args.save_dir, exist_ok = True)#./result_modulated

    """Model loader"""

    diffusion_stage1,diffusion_stage2, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    Fusion_Control_Model=Get_Fusion_Control_Model()
    Fusion_Control_Model_modulated=Get_Fusion_Control_Model()
    diffusion_stage1.load_state_dict(torch.load(diffusion_stage1_path))
    diffusion_stage2.load_state_dict(torch.load(diffusion_stage2_path))
    Fusion_Control_Model.load_state_dict(torch.load(FCM_path),strict=False)
    Fusion_Control_Model_modulated.load_state_dict(torch.load(FCM_path),strict=False)
    diffusion_stage1 = diffusion_stage1.to(args.device)
    diffusion_stage2 = diffusion_stage2.to(args.device)
    Fusion_Control_Model=Fusion_Control_Model.to(args.device)
    Fusion_Control_Model_modulated=Fusion_Control_Model_modulated.to(args.device)
    optimizer = AdamW(Fusion_Control_Model_modulated.parameters(), lr=2e-5, weight_decay=0.0)
    diffusion_stage1.eval()
    diffusion_stage2.eval()
    Fusion_Control_Model.eval()

    """OWT-VIT-SAM Model loader"""
    OWL_VIT_model, processor = load_owlvit(checkpoint_path="./modulated/checkpoint/owlvit-base-patch32/", device=args.device)
    predictor = SamPredictor(build_sam(checkpoint="./modulated/checkpoint/sam_vit_h_4b8939.pth").to(args.device))


    """Begin Test Fusion"""
    print("Text-DiFuse....Begin Test Fusion.....")
    for i, input in enumerate(test_loader):
        cond={'condition': input[0].to(args.device)}
        cond1={'condition': input[1].to(args.device)}
        cond_modulated,cond1_modulated,target_modulated,target=OWL_VIT_SAM(batch=input[0].to(args.device),batch1=input[1].to(args.device),input=input,text_prompt=args.text_prompt,processor=processor,model=OWL_VIT_model,predictor=predictor)
        # target_modulated 已经被调制过，可以用来进行扩散过程的控制；
        torch.manual_seed(0)
        x_F_t=torch.randn(input[0].shape, device=args.device)
        # 主分支噪声
        torch.manual_seed(0)
        x_F_t_m=torch.randn(input[0].shape, device=args.device)
        # 混合分支噪声
        torch.manual_seed(0)
        x_F_t_modulated=torch.randn(target_modulated.shape, device=args.device)
        # 这里是语义引导分支的噪声内容；
        filename = os.path.basename(str(input[4]))[:-3] # 对应文件的扩展名内容不需要了；
        indices = list(range(diffusion.num_timesteps))[::-1]
        for timestep in indices:
            # -----------------------------------------------------
            # 1. 当前时间步 t
            # -----------------------------------------------------
            t = torch.tensor([timestep] * x_F_t.shape[0], device=args.device)

            # =====================================================
            # 2. 主扩散分支：负责基本逆扩散，不训练
            # =====================================================
            out_main = diffusion.p_mean_variance(
                diffusion_stage1,
                diffusion_stage2,
                Fusion_Control_Model,
                x_F_t,
                t,
                model_kwargs=cond,
                model_kwargs1=cond1,
            )

            # 主分支输出 X_t-1 的 mean
            output_xt = out_main['mean']  # 不训练，只推理
            output_GT = resize_and_align16_batch(out_main['pred_xstart']).detach()
            # output_GT 作为 target，不参与梯度，因此 detach 只是更安全

            # 下一步噪声输入，切断梯度
            x_F_t = output_xt.detach()

            # =====================================================
            # 3. Modulated 语义引导分支（唯一训练分支）
            # =====================================================
            optimizer.zero_grad()

            loss, out_mod = diffusion.modulated_loss(
                diffusion_stage1,
                diffusion_stage2,
                Fusion_Control_Model_modulated,  # 这里的参数将被训练
                x_F_t_modulated,  # 当前 timestep 的噪声输入
                t,
                output_GT,  # 来自主分支 x0 的 target
                target_modulated,  # 常量 mask，不需要 requires_grad
                model_kwargs=cond_modulated,
                model_kwargs1=cond1_modulated,
            )

            loss.backward()
            optimizer.step()

            # 更新下一次 timestep 的输入
            output_xt_mod = out_mod['mean']
            x_F_t_modulated = output_xt_mod.detach()

            # =====================================================
            # 4. 混合分支（semantic mask 引导区域）
            # =====================================================
            out_mix = diffusion.p_mean_variance(
                diffusion_stage1,
                diffusion_stage2,
                Fusion_Control_Model_modulated,  # 与 modulated 分支共享参数，但此处无梯度
                x_F_t_m,
                t,
                model_kwargs=cond,
                model_kwargs1=cond1,
            )

            # 当前 timestep 的混合结果
            output_xt_m = out_mix['mean']

            # 语义区域 = modulated；非语义区域 = main
            sample_m = output_xt_m * target + output_xt * (1 - target)

            # 下一步噪声输入（切断计算图）
            x_F_t_m = sample_m.detach()
            # x_F_t 在上面主分支已经 detach
        output = to_numpy_image(torch.cat((x_F_t_m,input[2].to(args.device),input[3].to(args.device)),dim=1))
        output5= cv2.cvtColor(output[0], cv2.COLOR_YCrCb2RGB)
        output_name = filename[:-4]+'_modulated.png'
        Image.fromarray(output5).save(os.path.join(args.save_dir, output_name))
        output = to_numpy_image(torch.cat((x_F_t,input[2].to(args.device),input[3].to(args.device)),dim=1))
        output5= cv2.cvtColor(output[0], cv2.COLOR_YCrCb2RGB)
        output_name = filename
        Image.fromarray(output5).save(os.path.join(args.save_dir, output_name))



if __name__ == "__main__":
    main()
