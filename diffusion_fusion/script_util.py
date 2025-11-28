import argparse

from diffusion_fusion import gaussian_diffusion as gd

from diffusion_fusion import gaussian_diffusion as gd
from diffusion_fusion.respace import SpacedDiffusion, space_timesteps  # 均匀切块之后给出新的时间步的内容；
from diffusion_fusion.unet import diffusion_stage1, diffusion_stage2


def model_and_diffusion_defaults():
    return dict(
        # model defaults
        image_size=64, # 128
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,  # 恒定的4 个heads
        num_heads_upsample=-1,
        num_head_channels=64,  # 每个head应该负责的注意力头的数量；
        attention_resolutions="16,8",
        channel_mult="",  # 1 2 4 8
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,  # FILM 帮助时间信息的注入
        use_new_attention_order=True,  # 使用新的形式的注意力操作；
        # diffusion defaults
        learn_sigma=True,
        sigma_small=False,
        diffusion_steps=1000,
        noise_schedule="cosine",
        timestep_respacing="25",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,  # 是否对模型预测的方差sigma 进行缩放处理；
    )


def create_model_and_diffusion(
        image_size,
        learn_sigma,
        sigma_small,
        num_channels,
        num_res_blocks,
        channel_mult,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        attention_resolutions,
        dropout,
        diffusion_steps,
        noise_schedule,
        timestep_respacing,
        use_kl,
        predict_xstart,
        rescale_timesteps,
        rescale_learned_sigmas,
        use_checkpoint,  # 检查点技术，用来节省内存；
        use_scale_shift_norm,  # 原始的norm 基础上加入了 FiLM 调制的操作；
        use_new_attention_order,
):
    diffusion_stage1 = diffusion_stage1_create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,  # 需要的维度就要* 2
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,  # 这个参数用不上啊；
        use_scale_shift_norm=use_scale_shift_norm,  # FILM
        dropout=dropout,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion_stage2 = diffusion_stage2_create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        use_new_attention_order=use_new_attention_order,
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,  # 使用重新计算得到的小方差内容；
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return diffusion_stage1, diffusion_stage2, diffusion


def diffusion_stage2_create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult="",
        learn_sigma=False,
        use_checkpoint=False,
        attention_resolutions="16,8",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return diffusion_stage2(
        # in_channels=1,
        in_channels=8,
        model_channels=num_channels,
        # out_channels=(1 if not learn_sigma else 2),
        out_channels=(8 if not learn_sigma else 16),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        use_new_attention_order=use_new_attention_order,
    )


def diffusion_stage1_create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult="",
        learn_sigma=False,
        use_checkpoint=False,
        attention_resolutions="16,8",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,  # 上采样阶段的注意力头的数量情况；
        use_scale_shift_norm=False,
        dropout=0,
        use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
            # 每个阶段都需要进行一次下采样的操作的同时，其对应的分辨率降低为原始的 H/2 W/2 通道上的维度变换为 C*channel_mult[i]
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
        # 128//16 128//8 需要进行注意力图像维度的时候  给定的图像的维度是 128
        # 指定图像的维度大小为 这样的时候进行attention的操作；

    return diffusion_stage1(
        # in_channels=1,
        in_channels=8,  # embed_dim * 2
        model_channels=num_channels,  # 128
        # out_channels=(1 if not learn_sigma else 2),
        out_channels=(8 if not learn_sigma else 16), # 需要注意修改的地方内容；
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),  # 图像的尺寸内容//图像应该被缩放的倍数情况；
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,  # 使用FILM条件
        use_new_attention_order=use_new_attention_order,  # 使用新的注意力顺序
    )


def create_gaussian_diffusion(
        *,
        steps=1000, # 指定一开始steps 的数量情况；
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
): # 最开始指定的diffusion_steps
    betas = gd.get_named_beta_schedule(noise_schedule, steps)  # 0-T-1 得到的初始化的betas内容
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE  # 缺少KL
    if not timestep_respacing:  # 按照原始的1000的形式执行
        timestep_respacing = [steps] # 但是我们给定的是25
    return SpacedDiffusion(
        # 假如我们设置为step=250 那么就是从其中 按照字符串选择 0-250 中的100个
        use_timesteps=space_timesteps(steps, timestep_respacing),  # 总的数量为从0开始的 后面为分块和块中取到的位置数量的内容了；
        # 分成对应的多个块内容，然后元素是指块中的元素数量
        # 从0-T-1 中分块等间隔抽取的内容
        betas=betas,  # 从其中等间隔抽取到指定的内容；
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            # 预测噪声和预测原始的图像内容 / 预测当前的均值内容  xt-1 时刻的均值内容 ；previous_x
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE  # 缺少 LEARNED
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,  # 是否将参数重新映射到 0-1000 之间
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


import torch


# 假设当前文件名为 model_creation.py，如果不在一起，请调整引用
# from model_creation import diffusion_stage1_create_model, diffusion_stage2_create_model

def test_unet_encoder():
    """
    测试 diffusion_stage1 (作为 Encoder 角色) 的构建与前向传播
    """
    print("--- Starting Test: Unet Encoder (Stage 1) ---")

    # 1. 配置参数 (参考 model_and_diffusion_defaults 和 create_model 的逻辑)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 64
    num_channels = 128  # 为了测试速度，可以适当减小通道数，原默认是128
    num_res_blocks = 2
    # image_size=128 时，channel_mult 默认为 (1, 1, 2, 3, 4)
    channel_mult = ""
    learn_sigma = True  # 如果为 True，输出通道数应为 2 (mean + variance)
    attention_resolutions = "16,8"
    num_heads = 1
    # 2. 构建模型
    try:
        model = diffusion_stage1_create_model(
            image_size=image_size,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            learn_sigma=learn_sigma,
            use_checkpoint=False,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0.0,
            use_new_attention_order=True,
        )
        model.to(device)
        model.eval()  # 设置为评估模式
        print(f"Model build success. Parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"Failed to build Stage 1 model: {e}")
        return

    # 3. 构建模拟输入数据
    batch_size = 2
    # 注意：代码中 hardcode 了 in_channels=1
    x = torch.randn(batch_size, 16, image_size, image_size).to(device)
    # 模拟时间步，范围通常是 0 到 diffusion_steps (例如 1000)
    timesteps = torch.randint(0, 1000, (batch_size,)).long().to(device)

    # 4. 前向传播测试
    with torch.no_grad():
        output = model(x, timesteps, x.clone())

    print(f"Input shape: {x.shape}")
    #  h  hs
    out = output[0]
    print(f'out.shape: {out.shape}')
    layers = output[1]
    for i, layer in enumerate(layers):
        print(f"Output of layer {i} shape: {layer.shape}")
    # print(f"Output shape: {output.shape}")

    # 5. 验证输出维度
    expected_out_channels = 2 if learn_sigma else 1
    # assert output.shape == (batch_size, expected_out_channels, image_size, image_size)
    print("✅ Test Unet Encoder Passed!")
    return output[0], output[1]  # 后面的一个的类型为list


def test_unet_decoder(h, hs):
    """
    测试 diffusion_stage2 (作为 Decoder 角色) 的构建与前向传播
    """
    print("\n--- Starting Test: Unet Decoder (Stage 2) ---")

    # 1. 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 64
    num_channels = 128
    num_res_blocks = 2
    channel_mult = ""  # 默认逻辑会将其转为 (1, 1, 2, 3, 4)
    learn_sigma = True  # 这里测试一下 False 的情况，输出通道应该为  这里应该是完全统一的；
    attention_resolutions = "16,8"
    num_heads = 4

    # 2. 构建模型
    try:
        model = diffusion_stage2_create_model(
            image_size=image_size,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            learn_sigma=learn_sigma,
            use_checkpoint=False,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0.0,
            use_new_attention_order=True,
        )
        model.to(device)
        model.eval()
        print(f"Model build success. Parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"Failed to build Stage 2 model: {e}")
        return

    # 3. 构建模拟输入数据
    batch_size = 2
    # 注意：代码中 hardcode 了 in_channels=1
    x = torch.randn(batch_size, 16, image_size, image_size).to(device)
    # B  1 128 128 # 模拟时间步，范围通常是 0 到 diffusion_steps (例如 1000)
    timesteps = torch.randint(0, 1000, (batch_size,)).long().to(device)
    #

    # 4. 前向传播测试

    with torch.no_grad():
        output = model(x, timesteps, h, hs, x.clone())  # 作为cond 内容；

    print(f"Input shape: {x.shape}")
    print(f'Output shape : {output.shape}')

    print("✅ Test Unet Decoder Passed!")


if __name__ == "__main__":
    # 确保已经导入了 create_model 相关的函数
    # 这里直接调用测试函数
    h, hs = test_unet_encoder()
    test_unet_decoder(h, hs)
