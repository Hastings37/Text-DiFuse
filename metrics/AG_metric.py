import torch
import torch.nn.functional as F
# from basicivif.utils.registry import METRIC_REGISTRY

'''
该函数计算图像的平均梯度强度（Average Gradient），
衡量图像灰度变化的剧烈程度（即清晰度）。
值越大 → 图像越清晰、细节越丰富。
'''


# @METRIC_REGISTRY.register()
def AG(fused_image, **kwargs):
    """
    计算图像的平均梯度 (Average Gradient)
    使用差分法计算梯度（兼容所有PyTorch版本）

    参数:
        image_tensor: [H, W] 的tensor

    返回:
        AG: 平均梯度值
    """

    # 转换为float类型

    fused_image = fused_image.clamp(0, 255).byte()
    image_tensor = fused_image
    img = image_tensor.float()

    # 使用差分计算梯度
    # x方向梯度（列方向）
    gradx = torch.zeros_like(img)
    gradx[:, :-1] = img[:, 1:] - img[:, :-1]
    gradx[:, -1] = img[:, -1] - img[:, -2]

    # y方向梯度（行方向）
    grady = torch.zeros_like(img)
    grady[:-1, :] = img[1:, :] - img[:-1, :]
    grady[-1, :] = img[-1, :] - img[-2, :]

    # 计算梯度幅值
    s = torch.sqrt((gradx ** 2 + grady ** 2) / 2)

    # 计算平均梯度
    AG = torch.sum(s) / (image_tensor.shape[0] * image_tensor.shape[1])

    return {'AG': AG.detach().cpu().item()}
