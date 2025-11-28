import importlib
import os.path as osp
from basicivif.utils import scandir  # 如果你已经把这个函数搬进 basicivif.utils，也可以改路径
from copy import deepcopy
from basicivif.utils.registry import METRIC_REGISTRY

__all__ = ['calculate_metrics']
# 获取当前 metrics 文件夹路径
metric_folder = osp.dirname(osp.abspath(__file__))

# 获取所有以 _metric.py 结尾的文件
metric_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(metric_folder) if v.endswith('_metric.py')
]

# 动态导入所有 metric 模块
_metric_modules = [
    importlib.import_module(f'basicivif.metrics.{file_name}')
    for file_name in metric_filenames
]



def calculate_metrics(data,opt):

    opt=deepcopy(opt)
    metric_type=opt.pop('type') # 后面的内容放弃了opt的形式；
    metric=METRIC_REGISTRY.get(metric_type)(**data) # 这里就只是将metric_data给到了其中；
    # 这里只是解开对应的data内容在这里的；
    return metric