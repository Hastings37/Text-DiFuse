import torch
# from basicivif.utils.registry import METRIC_REGISTRY
'''
衡量的是融合图像和两个源图像之间的相关性差异
具有良好的相关性： (想似并且互补) SCD相对更大
SCD 越大说明融合图中红外和可见光的信息整合越是充分的； 
'''

def corr2(a,b):
    a=a-torch.mean(a)
    b=b-torch.mean(b)
    cor=torch.sum(a*b)/torch.sqrt(torch.sum(a**2)*torch.sum(b**2)+1e-10)
    return cor
# @METRIC_REGISTRY.register()
def SCD(ir,vis,fuse):
    r=corr2(fuse-vis,ir)+corr2(fuse-ir,vis)
    return {'SCD':r.detach().cpu().item()}

