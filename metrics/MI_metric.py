

import math
import torch
'''
归一化的互信息计算结果： 0-1 范围内，靠近1 说明效果更好； 
融合图像和红外源图像共享的信息量 （热目标保留的程度） 
融合图像和可见光源图像共享的信息量 （纹理细节保留的程度）
数值越大说明融合图像和源图像共享的信息量越多，融合效果越好；
'''



# 要求是给定的图像都是灰度的情况并且，返回的是单个的+综合的情况；
def Hab(img1,img2,gray_level):
    '''
    Args:
        img1: C H W  RGB—>Gray(avg) 0-255
        img2: C H W  RGB—>Gray(avg) 0-255
        gray_level: 256
    输入的数据的范围和维度 以及通道的数量情况是什么？？？

    Returns:
    融合后的图像对比原始的红外/可见光图像的信息保留的程度
    '''
    img1=img1.long() # 确定为H W 维度的整数内容；
    img2=img2.long()
    h=torch.zeros((gray_level,gray_level),dtype=torch.float64)
    idx=torch.stack((img1.flatten(),img2.flatten()),dim=0)
    #idx.shape=(2,N*N)
    for k in range(idx.shape[1]):
        x,y=idx[:,k]
        # 使用相对应的数值构造起来的索引；
        h[x,y]+=1
    h=h/torch.sum(h)
    # h/=h.sum()
    px=h.sum(dim=1)
    py=h.sum(dim=0)
    mask_x=px>0
    mask_y=py>0
    mask_xy=h>0

    Hx=-torch.sum(px[mask_x]*torch.log2(px[mask_x]))
    Hy=-torch.sum(py[mask_y]*torch.log2(py[mask_y]))
    Hxy=-torch.sum(h[mask_xy]*torch.log2(h[mask_xy]))
    MI=Hx+Hy-Hxy
    return MI # 真是绝美的代码啊；

# from basicivif.utils.registry import METRIC_REGISTRY

# @METRIC_REGISTRY.register()
def MI(ir,vis,fuse,gray_level=256,**kwargs):
    # 数据为tensor H W 范围是 0-255 的灰度图像；
    MI_ir=Hab(ir,fuse,gray_level)
    MI_vis=Hab(vis,fuse,gray_level)
    MI_total=MI_ir+MI_vis
    return {'MI_ir':MI_ir.detach().cpu().item(),'MI_vis':MI_vis.detach().cpu().item(),'MI_total':MI_total.detach().cpu().item()}
# 全都默认返回为标量的形式
# 而且一般来说严重的val_dataloader 不进行特别的说明的时候都是在CPU上进行的计算；





