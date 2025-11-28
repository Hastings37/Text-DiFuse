# from basicivif.utils.registry import METRIC_REGISTRY

'''
融合图像是否保留了红外图像和可见光图像的边缘信息

基于梯度信息的相似性求出
'''

import torch
import torch.nn.functional as F
import math


def sobel_fn(x):
    """
    计算Sobel梯度
    输入: x - [H, W] 的tensor
    输出: gv, gh - 垂直和水平梯度
    """
    vtemp = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
    htemp = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8

    # 移到相同设备
    vtemp = vtemp.to(x.device)
    htemp = htemp.to(x.device)

    a, b = htemp.shape
    x_ext = per_extn_im_fn(x, a)

    # 添加batch和channel维度 [1, 1, H, W]
    x_ext = x_ext.unsqueeze(0).unsqueeze(0)
    vtemp = vtemp.unsqueeze(0).unsqueeze(0)
    htemp = htemp.unsqueeze(0).unsqueeze(0)

    gv = F.conv2d(x_ext, vtemp, padding=0).squeeze(0).squeeze(0)
    gh = F.conv2d(x_ext, htemp, padding=0).squeeze(0).squeeze(0)

    return gv, gh


def per_extn_im_fn(x, wsize):
    """
    边界扩展
    输入: x - [H, W] 的tensor
    """
    hwsize = (wsize - 1) // 2
    p, q = x.shape
    xout_ext = torch.zeros((p + wsize - 1, q + wsize - 1), dtype=x.dtype, device=x.device)
    xout_ext[hwsize: p + hwsize, hwsize: q + hwsize] = x

    # 边界扩展
    if wsize - 1 == hwsize + 1:
        xout_ext[0: hwsize, :] = xout_ext[2, :].unsqueeze(0)
        xout_ext[p + hwsize: p + wsize - 1, :] = xout_ext[-3, :].unsqueeze(0)

    xout_ext[:, 0: hwsize] = xout_ext[:, 2].unsqueeze(1)
    xout_ext[:, q + hwsize: q + wsize - 1] = xout_ext[:, -3].unsqueeze(1)

    return xout_ext

# @METRIC_REGISTRY.register()
def Qabf(ir,vis,fuse):
    pA=ir
    pB=vis
    pF=fuse
    """
    计算Qabf图像融合质量指标

    输入:
        pA, pB, pF - [H, W] 的tensor，取值范围 0-255
    输出:
        output - Qabf分数 (标量)
    """
    # 参数设置
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Sobel算子
    h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
    h2 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32)
    h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)

    # 确保输入在正确的设备上
    device = pA.device
    h1 = h1.to(device)
    h3 = h3.to(device)

    strA = pA.float()
    strB = pB.float()
    strF = pF.float()

    def convolution(k, data):
        """使用torch的卷积"""
        # 添加batch和channel维度
        data = data.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        k = k.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        # padding=1保持尺寸不变
        img_new = F.conv2d(data, k, padding=1)
        return img_new.squeeze(0).squeeze(0)

    def getArray(img):
        """计算梯度幅值和方向"""
        SAx = convolution(h3, img)
        SAy = convolution(h1, img)
        gA = torch.sqrt(SAx * SAx + SAy * SAy)

        n, m = img.shape
        aA = torch.zeros((n, m), dtype=torch.float32, device=device)

        # 避免除零
        zero_mask = SAx == 0
        aA[~zero_mask] = torch.atan(SAy[~zero_mask] / SAx[~zero_mask])
        aA[zero_mask] = math.pi / 2

        return gA, aA

    gA, aA = getArray(strA)
    gB, aB = getArray(strB)
    gF, aF = getArray(strF)

    def getQabf(aA, gA, aF, gF):
        """计算Qabf"""
        # 梯度保持度
        mask = (gA > gF)
        GAF = torch.where(mask, gF / (gA + 1e-8),
                          torch.where(gA == gF, gF, gA / (gF + 1e-8)))

        # 方向保持度
        AAF = 1 - torch.abs(aA - aF) / (math.pi / 2)

        # 梯度强度保持度
        QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))

        # 方向保持度
        QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))

        # 总体质量
        QAF = QgAF * QaAF

        return QAF

    QAF = getQabf(aA, gA, aF, gF)
    QBF = getQabf(aB, gB, aF, gF)

    # 加权求和
    deno = torch.sum(gA + gB)
    nume = torch.sum(QAF * gA + QBF * gB)
    output = nume / (deno + 1e-8)

    return {'Qabf': output.detach().cpu().item()}


def get_Qabf_batch(pA, pB, pF):
    """
    批量版本的Qabf计算

    输入:
        pA, pB, pF - [B, H, W] 或 [B, 1, H, W] 的tensor，取值范围 0-255
    输出:
        output - [B] 的tensor，每个样本的Qabf分数
    """
    # 确保是3维 [B, H, W]
    if pA.dim() == 4:
        pA = pA.squeeze(1)
        pB = pB.squeeze(1)
        pF = pF.squeeze(1)

    batch_size = pA.shape[0]
    results = []

    for i in range(batch_size):
        score = Qabf(pA[i], pB[i], pF[i])
        results.append(score)

    return torch.tensor(results, device=pA.device)


# 示例用法
if __name__ == "__main__":
    # 创建示例数据 [H, W]
    H, W = 256, 256
    imgA = torch.randint(0, 256, (H, W), dtype=torch.float32)
    imgB = torch.randint(0, 256, (H, W), dtype=torch.float32)
    imgF = (imgA + imgB) / 2  # 简单的融合结果

    # 计算Qabf
    qabf_score = Qabf(imgA, imgB, imgF)
    print(f"Qabf Score: {qabf_score:.4f}")

    # 批量计算
    batch_imgA = torch.randint(0, 256, (4, H, W), dtype=torch.float32)
    batch_imgB = torch.randint(0, 256, (4, H, W), dtype=torch.float32)
    batch_imgF = (batch_imgA + batch_imgB) / 2

    batch_scores = get_Qabf_batch(batch_imgA, batch_imgB, batch_imgF)
    print(f"Batch Qabf Scores: {batch_scores}")