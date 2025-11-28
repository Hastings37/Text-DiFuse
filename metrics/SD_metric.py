# from basicivif.utils.registry import METRIC_REGISTRY
import torch

'''
图像灰度分布的离散程度 
数值越大越是说明： 
   图像的两亮暗对比明显
   图像灰度变化更加剧烈
   图像细节丰富 
   
数值越小越是说明： 
    图像灰度分布比较集中
    图像亮暗对比不明显
    图像细节较少
    内容相对平坦，对比度低
'''
# @METRIC_REGISTRY.register()
def SD(image_tensor):
    h,w=image_tensor.shape
    u=torch.mean(image_tensor) # tensor标量的形式
    SD=torch.sqrt(torch.sum((image_tensor-u)**2)/(h*w))
    return {'SD':SD.detach().cpu().item()}
# 返回的是单个的数值；