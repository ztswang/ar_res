import pyiqa
import torch
import os
from pyiqa import imread2tensor
from tqdm import tqdm
import numpy as np
from SMD2 import SMD2Calculator as SMD2

def img_trans(image_path):
    """
        image_path-->tensor
    """
    img_tensor = imread2tensor(image_path).unsqueeze(0).to(device)
    return img_tensor

# def img_ssim_calculate(img1_tensor, img2_tensor):
#     # ssim_loss_hww = pyiqa.create_metric('ssimc', device=device, as_loss=True)
#     # loss_hww = 1 - ssim_loss_hww(img1_tensor, img2_tensor)   # ssim is not lower better
#     ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr', device=device)
#     ssim_score = ssim_metric(img1_tensor, img2_tensor)
#     return ssim_score
#     # return loss_hww

# def ssim_calculate(image1_folder, image2_folder):
#     img1_list = sorted(os.listdir(image1_folder))
#     img2_list = sorted(os.listdir(image2_folder))
#     assert len(img1_list) == len(img2_list), "the num of images in two folders should be same"
#     imgs_ssim_score = torch.zeros(len(img1_list), device=device)
#     for i in tqdm(range(len(img1_list))):
#         assert img1_list[i][:3] == img2_list[i][:3], f"{img1_list[i]} and {img2_list[i]} are not a pair."
#         img1_path = os.path.join(image1_folder, img1_list[i])
#         img2_path = os.path.join(image2_folder, img2_list[i])
#         img1_tensor = img_trans(image_path=img1_path)
#         img2_tensor = img_trans(image_path=img2_path)  
#         print(img1_tensor.shape)
#         print(img2_tensor.shape)      
#         imgs_ssim_score[i] = img_ssim_calculate(img1_tensor, img2_tensor)
#     return imgs_ssim_score.mean()

# def img_psnr_calculate(img1_tensor, img2_tensor):
#     psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=device)
#     psnr_score = psnr_metric(img1_tensor, img2_tensor)
#     return psnr_score

# def psnr_calculate(image1_folder, image2_folder):
#     img1_list = sorted(os.listdir(image1_folder))
#     img2_list = sorted(os.listdir(image2_folder))
#     assert len(img1_list) == len(img2_list), "the num of images in two folders should be same"
#     imgs_psnr_score = torch.zeros(len(img1_list), device=device)
#     for i in tqdm(range(len(img1_list))):
#         assert img1_list[i][:4] == img2_list[i][:4], f"{img1_list[i]} and {img2_list[i]} are not a pair."
#         img1_path = os.path.join(image1_folder, img1_list[i])
#         img2_path = os.path.join(image2_folder, img2_list[i])
#         img1_tensor = img_trans(image_path=img1_path)
#         img2_tensor = img_trans(image_path=img2_path)
#         imgs_psnr_score[i] = img_psnr_calculate(img1_tensor, img2_tensor)
#     return imgs_psnr_score.mean()

# def img_lpips_calculate(img1_tensor, img2_tensor):
#     lpips_metric = pyiqa.create_metric('lpips', device=device)
#     lpips_score = lpips_metric(img1_tensor, img2_tensor)
#     return lpips_score

# def lpips_calculate(image1_folder, image2_folder):
#     img1_list = sorted(os.listdir(image1_folder))
#     img2_list = sorted(os.listdir(image2_folder))
#     assert len(img1_list) == len(img2_list), "the number of images in two folders should be the same"
#     imgs_lpips_score = []
#     lpips_model = lpips.LPIPS(net="alex", version='0.1').to(device)
#     for i in tqdm(range(len(img1_list))):
#         assert img1_list[i][:4] == img2_list[i][:4], f"{img1_list[i]} and {img2_list[i]} are not a pair."
#         img1_path = os.path.join(image1_folder, img1_list[i])
#         img2_path = os.path.join(image2_folder, img2_list[i])
#         img1_tensor = img_trans(img1_path).to(device)
#         img2_tensor = img_trans(img2_path).to(device)
        
#         with torch.no_grad():
#             lpips_score = lpips_model(img1_tensor, img2_tensor).item()
#         imgs_lpips_score.append(lpips_score)
        
#         # 清理变量，释放显存
#         del img1_tensor, img2_tensor, lpips_score
#         torch.cuda.empty_cache()
#         gc.collect()

def fid_calculate(image1_folderpath, image2_folderpath):
    fid_metric = pyiqa.create_metric('fid', device=device)
    fid_score = fid_metric(image1_folderpath, image2_folderpath)
    return fid_score

def round_to_sigfigs(x, sigfigs=4):
    """保留 x 的四位有效数字"""
    if x == 0 or not np.isfinite(x):
        return x
    factor = sigfigs - int(np.floor(np.log10(abs(x)))) - 1
    return round(x, factor) if factor > 0 else round(x)

def sr_metrics_calculate(gt_folder, sr_folder):
    """
    计算sr指标分数
    """
    # 创建计算
    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr', device=device)
    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=device)
    fsim_metric = pyiqa.create_metric('fsim', device=device)
    musiq_metric = pyiqa.create_metric('musiq', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    SMD2_metric = SMD2()
    clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)

    print(sr_folder)
    gt_list = sorted(os.listdir(gt_folder))
    sr_list = sorted(os.listdir(sr_folder))
    assert len(gt_list) == len(sr_list), f"the num of images in two folders should be same, gt:{len(gt_list)}, sr:{len(sr_list)}"

    # Initialize scores
    psnr_score = torch.zeros(len(gt_list))
    ssim_score = torch.zeros(len(gt_list))
    fsim_score = torch.zeros(len(gt_list))
    niqe_score = torch.zeros(len(gt_list))
    lpips_score = torch.zeros(len(gt_list))
    smd2_score = torch.zeros(len(gt_list))
    musiq_score = torch.zeros(len(gt_list))
    clipiqa_score = torch.zeros(len(gt_list))
    lpips_score = torch.zeros(len(gt_list))

    # Calculate metrics
    for i in tqdm(range(len(sr_list))):
        assert sr_list[i][:-4] == gt_list[i], f"{sr_list[i]} and {gt_list[i]} are not a pair."
        
        gt_img_path = os.path.join(gt_folder, gt_list[i])
        sr_img_path = os.path.join(sr_folder, sr_list[i])
        gt_img_tensor = img_trans(gt_img_path)
        sr_img_tensor = img_trans(sr_img_path)
        
        psnr_score[i] = psnr_metric(gt_img_tensor, sr_img_tensor)
        ssim_score[i] = ssim_metric(gt_img_tensor, sr_img_tensor)
        fsim_score[i] = fsim_metric(gt_img_tensor, sr_img_tensor)
        musiq_score[i] = musiq_metric(sr_img_tensor)
        niqe_score[i] = niqe_metric(sr_img_tensor)
        smd2_score[i] = SMD2_metric.calculate_smd2_from_path(sr_img_path)
        clipiqa_score[i] = clipiqa_metric(sr_img_tensor)
        lpips_score[i] = lpips_metric(gt_img_tensor, sr_img_tensor)
        
    print('start compute fid')
    fid_score = fid_calculate(gt_folder, sr_folder)

    # print('start compute lpips')
    # lpips_score, _ = lpips_compute(gt_folder, sr_folder)

    # Format the scores to 4 significant figures
    data = {
        "sr_folder": [sr_folder],
        "psnr_score": [round_to_sigfigs(psnr_score.mean().item(), 4)],
        "ssim_score": [round_to_sigfigs(ssim_score.mean().item(), 4)],
        "fsim_score": [round_to_sigfigs(fsim_score.mean().item(), 4)],
        "lpips_score": [round_to_sigfigs(lpips_score.mean().item(), 4)],
        "fid_score": [round_to_sigfigs(fid_score, 4)],
        "niqe_score": [round_to_sigfigs(niqe_score.mean().item(), 4)],
        
        "smd2_score": [round_to_sigfigs(smd2_score.mean().item(), 4)],
        "musiq_score": [round_to_sigfigs(musiq_score.mean().item(), 4)],
        "clipiqa_score": [round_to_sigfigs(clipiqa_score.mean().item(), 4)],
    }

    return data

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 目标路径
# 这个很奇怪 我忘记了 可能是我设置了bs以后重新复现 但是结果很高 所以选择official比较
# image1_folderpath = "/data/hww/hww-gen/AST/logs/derain/raindrop/AST_B_1102_s1/results"          # stage 1 fuxian bs-32-10-4
# image1_folderpath = "/data/hww/hww-gen/AST/logs/derain/raindrop/AST_B_1103_s2/results"        # stage 2 fuxian bs-32-10-4
# image1_folderpath = "/data/hww/hww-gen/AST/logs/derain/raindrop/AST_B_1103_s3/results"        # stage 3 fuxian bs-32-10-4

# 复现差的 但是我找不到 模型结构了
"""
{'sr_folder': ['/data/hww/hww-gen/derain-fuxian/raindrop/AST_B_1103_s3/results'], 'psnr_score': [32.33], 'ssim_score': [0.9348], 'fsim_score': [0.9716], 'lpips_score': [0.06755], 'fid_score': [29.9], 'niqe_score': [3.229], 'smd2_score': [0.1488], 'musiq_score': [69.39], 'clipiqa_score': [0.4131]}
"""

# 这个应该是另一个版本的quac，结果不好
# image1_folderpath = "/data/hww/hww-gen/AST-quac1/logs/derain/raindrop/AST_B_quac1_s1/results"
# image1_folderpath = "/data/hww/hww-gen/AST-quac1/logs/derain/raindrop/AST_B_quac1_s2/results"
# image1_folderpath = "/data/hww/hww-gen/AST-quac1/logs/derain/raindrop/AST_B_quac1_s3/results"
"""
{'sr_folder': ['/data/hww/hww-gen/AST-quac1/logs/derain/raindrop/AST_B_quac1_s3/results'], 'psnr_score': [32.28], 'ssim_score': [0.9329], 
'fsim_score': [0.9706],  'lpips_score': [0.06741], 'fid_score': [31.66], 'niqe_score': [3.171], 'smd2_score': [0.148], 'musiq_score': [67.98], 
'clipiqa_score': [0.4201]}
"""

# quac 为什么路进是raindrop1，因为raindrop我也训练过(第一次训练结果不好 我删掉了)
# {'sr_folder': ['/data/hww/hww-gen/AST-quac/logs/derain/raindrop/AST_B_quac_s3/results'], 'psnr_score': [32.28], 'ssim_score': [0.9355], 'fsim_score': [0.971], 'lpips_score': [0.06328], 'fid_score': [31.1], 'niqe_score': [3.242], 'smd2_score': [0.1452], 'musiq_score': [69.81], 'clipiqa_score': [0.4186]}
"""
{'sr_folder': ['/data/hww/hww-gen/AST-results/quac'], 'psnr_score': [32.33], 'ssim_score': [0.9364], 'fsim_score': [0.9717], 'lpips_score': [0.06137], 'fid_score': [28.7], 'niqe_score': [3.247], 'smd2_score': [0.1488], 'musiq_score': [70.13], 'clipiqa_score': [0.4209]}
/data/hww/hww-gen/AST-quac/logs/derain/raindrop1/AST_B_1103_s3/results
"""
# github
"""
{'sr_folder': ['/data/hww/hww-gen/AST-results/github'], 'psnr_score': [32.34], 'ssim_score': [0.9355], 'fsim_score': [0.9714], 'lpips_score': [0.06588], 'fid_score': [30.28], 'niqe_score': [3.272], 'smd2_score': [0.1454], 'musiq_score': [69.75], 'clipiqa_score': [0.4267]}
/data/hww/hww-gen/AST-quac/logs/github/results
"""
# meta-ACON
"""
{'sr_folder': ['/data/hww/hww-gen/AST-results/metaacon'], 'psnr_score': [32.21], 'ssim_score': [0.934], 'fsim_score': [0.9703], 'lpips_score': [0.06975], 'fid_score': [31.76], 'niqe_score': [3.287], 'smd2_score': [0.1458], 'musiq_score': [69.76], 'clipiqa_score': [0.4117]}
/data/hww/hww-gen/AST-metaacon/logs/derain/raindrop/AST_B_metaacon_s3/results
"""
# image1_folderpath = "/data/hww/hww-gen/AST-results/quac"
# image1_folderpath = "/data/hww/hww-gen/AST-results/metaacon"
# image1_folderpath = "/data/hww/hww-gen/AST-results/quac-q"
# image1_folderpath = "/data/hww/hww-gen/AST-results/new-quac"
# image1_folderpath = "/data/hww/hww-gen/AST-results/new-quac-q"
# image1_folderpath = "/data/hww/hww-gen/AST-results/AST"
# image1_folderpath = "/data/hww/hww-gen/AST/logs/derain/raindrop/AST_B_1103_s3/results"
# image1_folderpath = "/data/hww/hww-gen/AST-quac/logs/github/results"
image1_folderpath = "/data/hww/hww-gen/AST-metaacon/logs/derain/raindrop/AST_B_metaacon_s3/results"
# print("image1_folderpath", image1_folderpath)
image2_folderpath = "/data/hww/hww-gen/AST/dataset/raindrop/test_a/gt"
sr_metrics_calculate = sr_metrics_calculate(gt_folder=image2_folderpath, sr_folder=image1_folderpath)
print(sr_metrics_calculate)
"""
{'sr_folder': ['/data/hww/hww-gen/AST-results/metaacon'], 'psnr_score': [32.21], 'ssim_score': [0.934], 'fsim_score': [0.9703], 'lpips_score': [0.06975], 'fid_score': [31.76], 'niqe_score': [3.287], 'smd2_score': [0.1458], 'musiq_score': [69.76], 'clipiqa_score': [0.4117]}
{'sr_folder': ['/data/hww/hww-gen/AST-results/github'], 'psnr_score': [32.34], 'ssim_score': [0.9355], 'fsim_score': [0.9714], 'lpips_score': [0.06588], 'fid_score': [30.28], 'niqe_score': [3.272], 'smd2_score': [0.1454], 'musiq_score': [69.75], 'clipiqa_score': [0.4267]}
{'sr_folder': ['/data/hww/hww-gen/AST-results/quac'], 'psnr_score': [32.33], 'ssim_score': [0.9364], 'fsim_score': [0.9717], 'lpips_score': [0.06137], 'fid_score': [28.7], 'niqe_score': [3.247], 'smd2_score': [0.1488], 'musiq_score': [70.13], 'clipiqa_score': [0.4209]}
### 下面这个结果很奇怪 我需要iclr之后确认 github和这个的区别
{'sr_folder': ['/data/hww/hww-gen/AST/logs/derain/raindrop/AST_B_1103_s3/results'], 'psnr_score': [32.38], 'ssim_score': [0.9363], 'fsim_score': [0.9716], 'lpips_score': [0.06376], 'fid_score': [29.0], 'niqe_score': [3.291], 'smd2_score': [0.1503], 'musiq_score': [70.08], 'clipiqa_score': [0.4242]}
"""

