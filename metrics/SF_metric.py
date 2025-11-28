# from basicivif.utils.registry import METRIC_REGISTRY
import torch

'''
衡量的是图像在空间域中的灰度变化速率
图像中内容的变换的频繁程度 
图像中的边缘和纹理越多，其对应的SF也就是越大的
'''

# @METRIC_REGISTRY.register()
def SF(image_tensor):

    RF = image_tensor[1:, :] - image_tensor[:-1, :]
    CF = image_tensor[:, 1:] - image_tensor[:, :-1]
    RF1 = torch.sqrt(torch.mean(RF ** 2))
    CF1 = torch.sqrt(torch.mean(CF ** 2))

    SF = torch.sqrt(RF1 ** 2 + CF1 ** 2)
    return {'SF': SF.detach().cpu().item()}