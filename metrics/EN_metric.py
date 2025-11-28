

# from basicivif.utils.registry import METRIC_REGISTRY
import numpy as np
import torch

'''
数值越大对应的信息越多，细节越是丰富； 
'''

# @METRIC_REGISTRY.register()
def EN(image_tensor):
    image_tensor=image_tensor.detach().cpu().flatten()
    histogram=torch.histc(image_tensor,bins=256,min=0,max=255)
    histogram=histogram/histogram.sum()
    entropy=-torch.sum(histogram*torch.log2(histogram+1e-10))
    return {'EN':entropy.detach().cpu().item()}
