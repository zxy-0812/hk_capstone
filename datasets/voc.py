import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import Dataset
import torchvision.transforms as transforms

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

# class RegressionDataset(Dataset):
#     def __init__(self, image_path, label_path, transform=None, transform_mode=False):
#         """
#         Args:
#             image_path (str): 图像存储的完整目录路径
#             label_path (str): 包含图像名和分数的CSV文件完整路径
#             transform (callable, optional): 图像预处理和增强转换
#             transform_mode (bool): 是否为测试模式
#         """
#         self.data_df = pd.read_csv(label_path)
#         self.image_path = image_path  # 直接使用完整图像路径
#         self.transform = transform
#         self.transform_mode = transform_mode

#         # 计算分数的统计信息（用于归一化）
#         self.scores = self.data_df['score'].values
#         self.mean_score = self.scores.mean()
#         self.std_score = self.scores.std() if len(self.scores) > 1 else 1.0

#     def __len__(self):
#         return len(self.data_df)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         # 加载图像（直接拼接目录和文件名）
#         img_name = os.path.join(self.image_path, self.data_df.iloc[idx, 0])
#         image = Image.open(img_name).convert('RGB')

#         # 获取目标分数
#         score = self.data_df.iloc[idx, 1]

#         # 归一化分数（推荐保留）
#         normalized_score = (score - self.mean_score) / self.std_score

#         # 应用转换（保留原有增强逻辑）
#         if self.transform:
#             if not self.transform_mode and 'Random' in str(self.transform):
#                 image = self.transform(image)
#             else:
#                 test_transform = transforms.Compose([
#                     t for t in self.transform.transforms if not isinstance(t, (
#                         transforms.RandomCrop, 
#                         transforms.RandomHorizontalFlip,
#                         transforms.RandomRotation,
#                         transforms.ColorJitter
#                     ))
#                 ])
#                 image = test_transform(image)

#         sample = {
#             'image': image,
#             'score': torch.tensor(normalized_score, dtype=torch.float32).squeeze()
#         }

#         return sample

class RegressionDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None, transform_mode=False):
        """
        Args:
            image_path (str): 图像存储目录路径
            label_path (str): 包含有效图片名的CSV文件路径
            transform (callable): 图像预处理流水线
            transform_mode (bool): 是否启用测试模式转换
        """
        # 读取标签文件并验证完整性
        self.data_df = pd.read_csv(label_path)
        self.image_path = image_path
        self._validate_files()  # 新增文件校验
      
        # 统计信息计算
        self.scores = self.data_df['score'].values
        self.mean_score = self.scores.mean()
        self.std_score = self.scores.std() if len(self.scores) > 1 else 1.0
      
        # 图像增强配置
        self.transform = transform
        self.transform_mode = transform_mode

    def _validate_files(self):
        """验证CSV中所有图片文件都存在"""
        valid_records = []
        missing_count = 0
      
        for idx, row in self.data_df.iterrows():
            img_name = os.path.join(self.image_path, row[0])
            if os.path.isfile(img_name):
                valid_records.append(row)
            else:
                missing_count += 1
                print(f"警告：缺少图片 {row[0]}，已自动过滤")
      
        # 更新有效数据
        self.data_df = pd.DataFrame(valid_records, columns=self.data_df.columns)
      
        if missing_count > 0:
            print(f"共过滤{missing_count}个无效记录，剩余有效样本：{len(self.data_df)}")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # 获取文件名和分数
        img_name = os.path.join(self.image_path, self.data_df.iloc[idx, 0])
        score = self.data_df.iloc[idx, 1]
      
        # 加载图像（已通过校验，确保存在）
        image = Image.open(img_name).convert('RGB')
      
        # 分数标准化
        normalized_score = (score - self.mean_score) / self.std_score
      
        # 动态生成转换流水线
        if self.transform:
            if self.transform_mode:
                image = self._get_test_transforms()(image)
            else:
                image = self.transform(image)
      
        return {
            'image': image,
            'score': torch.tensor(normalized_score, dtype=torch.float32).squeeze()
        }

    def _get_test_transforms(self):
        """生成测试模式转换流水线（移除随机增强）"""
        test_transforms = []
        for t in self.transform.transforms:
            if not isinstance(t, (
                transforms.RandomCrop, 
                transforms.RandomHorizontalFlip,
                transforms.RandomRotation,
                transforms.ColorJitter
            )):
                test_transforms.append(t)
        return transforms.Compose(test_transforms)

    

class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None):

        is_aug=False
        if year=='2012_aug':
            is_aug = True
            year = '2012'
        
        self.root = os.path.expanduser(root)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform
        
        self.image_set = image_set
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if is_aug and image_set=='train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            split_f = os.path.join( self.root, 'train_aug.txt')#'./datasets/data/train_aug.txt'
        else:
            mask_dir = os.path.join(voc_root, 'SegmentationClass')
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)