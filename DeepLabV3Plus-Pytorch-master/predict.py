from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
torch.hub.set_dir('D:\\torch_cache')

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import json

def get_argparser():
    parser = argparse.ArgumentParser()
    # Datset Options
    parser.add_argument("--input", type=str, required=True,help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='voc',choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,help="save segmentation results to the specified dir")
    parser.add_argument("--crop_val", action='store_true', default=False,help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)   
    parser.add_argument("--ckpt", default=None, type=str,help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',help="GPU ID")
    parser.add_argument("--json_save_path", default='datasets/segment_result/panorama_images_1_segmentation_results.json', type=str,
                    help="path to save the JSON results file")
    return parser

def get_original_class_from_train_id(train_id):
    """
    根据 train_id 查找原始分类的 id 和名称。
    
    参数:
        train_id (int): 训练时使用的类别 ID
    
    返回:
        dict: 包含原始分类的 id 和名称
    """
    for cls in Cityscapes.classes:
        if cls.train_id == train_id and cls.train_id != 255:  # 忽略无效类别
            return {
                "original_id": cls.id,
                "name": cls.name,
                "color": cls.color
            }
    return {"original_id": None, "name": "Unknown", "color": (0, 0, 0)}

def main():
    # train_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # 示例 train_id 列表
    # results = []

    # for tid in train_ids:
    #     result = get_original_class_from_train_id(tid)
    #     results.append(result)
        

    # for i, res in enumerate(results):
    #     print(f"Train ID {train_ids[i]} 对应的原始分类: ID={res['original_id']}, 名称={res['name']}, 颜色={res['color']}")
    
    
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    if opts.crop_val:
        transform = T.Compose([
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    # 创建一个字典来存储每个图片的类别统计信息
    results = {}

    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name+'.png'))

            # 计算每个类别的像素数量和占比
            unique, counts = np.unique(pred, return_counts=True)
            total_pixels = pred.size
            class_counts = dict(zip(unique, counts))
            class_percentages = {cls: (count / total_pixels) * 100 for cls, count in class_counts.items()}
            # 创建一个列表来存储当前图片的类别信息
            class_info = []
            for cls, count in class_counts.items():
                percentage = class_percentages[cls]
                original_class = get_original_class_from_train_id(cls)
                class_info.append({
                    "class_id": int(cls),
                    "pixels": int(count),
                    "percentage": float(percentage),
                    "name": original_class["name"]
                })

            # 将当前图片的类别信息添加到结果字典中
            results[img_name] = class_info

            # 输出结果
            print(f"Image: {img_name}")
            for cls, count in class_counts.items():
                percentage = class_percentages[cls]
                original_class = get_original_class_from_train_id(cls)
                print(f"Class {cls}: {count} pixels, {percentage:.2f}% - 原始分类: ID={original_class['original_id']}, 名称={original_class['name']}")

        # 将结果字典保存为 JSON 文件
        with open('datasets/segment_result/panorama_images_1_pixel.json', 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()

# python predict.py --input D:/A_test/HK/result  --dataset cityscapes --model deeplabv3plus_resnet101 --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar --save_val_results_to  D:/A_test/HK/datasets/segment_result --json_save_path D:/A_test/HK/datasets/segment_result/panorama_images_1_pixel.json --gpu_id 0
