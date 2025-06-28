# import torch
# import torch.nn as nn
# from network.modeling import deeplabv3plus_resnet101
# from torch.utils.data import DataLoader, random_split
# from network.model import PredictRegressionModel
# from datasets.voc import RegressionDataset
# import torchvision.transforms as transforms
# from tqdm import tqdm
# import numpy as np
# import argparse
# import math
# import random
# from torch.optim.lr_scheduler import LambdaLR
# # from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
# from datetime import datetime
# import os

# def set_seed(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def parse_args():
#     parser = argparse.ArgumentParser(description='Regression Model Training')
#     parser.add_argument('--exp_name', type=str, default='0524_panorama_images1_ep50__bs8_lr2e-5_test',
#                        help='实验名称（用于日志记录）(默认: regression_exp)')
#     parser.add_argument('--image_path', type=str, 
#                         default='/home/zhongxinyu/capstone/HK/datasets/panorama_images_1',
#                         help='数据集根目录路径')
#     parser.add_argument('--label_path', type=str, 
#                         default='/home/zhongxinyu/capstone/HK/datasets/labels.csv',
#                         help='训练标签路径')
#     parser.add_argument('--batch_size', type=int, default=8,
#                         help='训练批次大小 (默认: 4)')
#     parser.add_argument('--lr', type=float, default=2e-5,
#                         help='学习率 (默认: 0.0001)')
#     parser.add_argument('--epochs', type=int, default=50,
#                         help='训练轮数 (默认: 20)')
#     parser.add_argument('--pretrained', type=str, 
#                         default='/home/zhongxinyu/capstone/HK/DeepLabV3Plus-Pytorch-master/checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar',
#                         help='预训练权重路径')
#     parser.add_argument('--save_path', type=str, default='/home/zhongxinyu/capstone/HK/DeepLabV3Plus-Pytorch-master/checkpoints/best_regression_model.pth',
#                         help='模型保存路径 (默认: /home/zhongxinyu/capstone/HK/DeepLabV3Plus-Pytorch-master/checkpoints/deeplabv3plus_resnet101_regression_best_model.pth)')
#     parser.add_argument('--freeze_backbone', default=False, action='store_true',
#                         help='是否冻结骨干网络')
#     parser.add_argument('--warmup_epochs', type=int, default=1,
#                        help='warmup轮数 (默认: 5)')
#     parser.add_argument('--min_lr', type=float, default=1e-7,
#                        help='最小学习率 (默认: 1e-7)')
#     return parser.parse_args()


# # 新增调度器创建函数
# def create_scheduler(optimizer, warmup_epochs, num_epochs, min_lr):
#     def lr_lambda(current_epoch):
#         if current_epoch < warmup_epochs:
#             return float(current_epoch) / float(max(1, warmup_epochs))
#         progress = float(current_epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
#         return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
  
#     return LambdaLR(optimizer, lr_lambda)

# # 2. 加载预训练权重
# def load_pretrained_weights(model, path_to_pth):
#     state_dict = torch.load(path_to_pth)['model_state']
#     # 创建新的state_dict，只包含特征提取部分的权重
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'regression_layer' not in k}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
#     return model

# # 3. 冻结特征提取层
# def freeze_feature_extractor(model):
#     for param in model.feature_extractor.parameters():
#         param.requires_grad = False
#     return model

# # 验证函数
# def validate_model(model, val_loader, criterion, device):
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for batch in val_loader:
#             inputs = batch['image'].to(device)
#             labels = batch['score'].to(device).squeeze()
#             outputs = model(inputs).squeeze()
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#     return val_loss / len(val_loader)
    
# def train_model(model, train_loader, val_loader, criterion, optimizer, 
#                num_epochs, device, scheduler=None, log_interval=10):
#     best_loss = np.inf
#     global_step = 0
#     train_losses = []  # 记录训练损失
#     val_losses = []    # 记录验证损失
#     learning_rates = []  # 记录学习率
    
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
#         # 记录当前epoch的学习率
#         current_lr = optimizer.param_groups[0]['lr']
#         learning_rates.append(current_lr)
#         progress_bar.set_postfix({'lr': f'{current_lr:.2e}'})
        
#         for batch_idx, batch in enumerate(progress_bar):
#             inputs = batch['image'].to(device)
#             labels = batch['score'].to(device).squeeze()

#             optimizer.zero_grad()
#             outputs = model(inputs).squeeze()
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             global_step += 1
            
#             if log_interval > 0 and global_step % log_interval == 0:
#                 print(f'Step {global_step} | Train Loss: {loss.item():.4f} | LR: {current_lr:.2e}')

#         # 验证阶段
#         val_loss = validate_model(model, val_loader, criterion, device)
#         avg_train_loss = train_loss / len(train_loader)
        
#         # 记录损失
#         train_losses.append(avg_train_loss)
#         val_losses.append(val_loss)
      
#         # 更新学习率调度器
#         if scheduler:
#             scheduler.step()
#             # 更新学习率显示（如果调度器在epoch结束后更新）
#             current_lr = optimizer.param_groups[0]['lr']
      
#         # 保存最佳模型
#         if val_loss < best_loss:
#             best_loss = val_loss
#             torch.save(model.state_dict(), args.save_path)
#             print(f"New best model saved with val_loss: {val_loss:.4f}")

#         print(f'Epoch {epoch+1}/{num_epochs} | '
#               f'Train Loss: {avg_train_loss:.4f} | '
#               f'Val Loss: {val_loss:.4f} | '
#               f'LR: {current_lr:.2e}')
    
#     # 绘制并保存损失图
#     plt.figure(figsize=(12, 5))
    
#     # 左侧子图：损失曲线
#     plt.subplot(1, 2, 1)
#     plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
#     plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss', marker='s')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.grid(True)
    
#     # 右侧子图：学习率曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(range(1, num_epochs + 1), learning_rates, label='Learning Rate', marker='^', color='orange')
#     plt.xlabel('Epoch')
#     plt.ylabel('Learning Rate')
#     plt.title('Learning Rate Schedule')
#     plt.legend()
#     plt.grid(True)
#     plt.yscale('log')  # 使用对数刻度更好地显示学习率变化
    
#     plt.tight_layout()  # 确保子图之间有足够的间距
    
#     # 保存图像
#     plot_dir = '/home/zhongxinyu/capstone/HK/DeepLabV3Plus-Pytorch-master/train_loss_curves/'
#     os.makedirs(plot_dir, exist_ok=True)
#     plot_path = os.path.join(plot_dir, f'{args.exp_name}_{timestamp}_loss_lr_plot.png')
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"损失和学习率曲线图已保存至: {plot_path}")
#     # 保存图像

# if __name__ == "__main__":
#     set_seed(123)
#     args = parse_args()

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  
#     # # 创建TensorBoard Writer
#     # log_dir = f"/home/zhongxinyu/capstone/HK/tenosrboard/runs/{args.exp_name}_{timestamp}"  # 新日志目录格式
#     # writer = SummaryWriter(
#     #     log_dir=log_dir,
#     # )
  
#     # 初始化模型
#     model = PredictRegressionModel(num_classes=19)
#     model = load_pretrained_weights(model, args.pretrained)
#     print(f"预训练权重加载完成: {args.pretrained}")

#     # 冻结特征层（根据参数决定）
#     if args.freeze_backbone:
#         model = freeze_feature_extractor(model)
#         print("特征提取层已冻结")

#     # 设备配置
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     # 数据转换
#     transform = transforms.Compose([
#         # transforms.Resize((513, 513)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
#     ])

#     # 创建数据集
#     full_dataset = RegressionDataset(
#         image_path=args.image_path,
#         label_path=args.label_path,
#         transform=transform,
#         transform_mode=False,
#     )

#     # 数据集划分
#     train_size = int(0.8 * len(full_dataset))
#     val_size = len(full_dataset) - train_size
#     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

#     # 创建数据加载器
#     def seed_worker(worker_id):
#         worker_seed = torch.initial_seed() % 2**32
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)

#     generator = torch.Generator()
#     generator.manual_seed(2023)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         pin_memory=True,
#         worker_init_fn=seed_worker,
#         generator=generator
#     )
  
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         pin_memory=True,
#         worker_init_fn=seed_worker,
#         generator=generator
#     )

#     # 训练配置（使用参数中的学习率和轮数）
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.AdamW(model.parameters(), 
#                                 lr=args.lr, 
#                                 weight_decay=1e-4)
  
#     # 创建带warmup的cosine调度器
#     scheduler = create_scheduler(
#         optimizer=optimizer,
#         warmup_epochs=args.warmup_epochs,
#         num_epochs=args.epochs,
#         min_lr=args.min_lr
#     )
  
#     # 开始训练
#     train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         num_epochs=args.epochs,
#         device=device,
#         scheduler=scheduler
#     )
     
#     # writer.close()
#     print(f"模型训练完成，最佳模型已保存为 {args.save_path}")

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from network.modeling import deeplabv3plus_resnet101
from torch.utils.data import DataLoader, random_split
from network.model import PredictRegressionModel
from datasets.voc import RegressionDataset
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import argparse
import math
import random
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file, level=logging.INFO, rank=0):
    """设置日志记录器，同时输出到文件和控制台"""
    if rank != 0:  # 只在主进程中记录日志
        return None
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建文件处理器
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    
    # 创建控制台处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Regression Model Training')
    parser.add_argument('--exp_name', type=str, default='0622_panorama_images_all_merge_gpu4_ep50_bs32_lr1e-4',
                       help='实验名称（用于日志记录）(默认: regression_exp)')
    parser.add_argument('--image_path', type=str, 
                        default='/home/zhongxinyu/capstone/HK/datasets/total_images',
                        # default='/home/zhongxinyu/capstone/HK/datasets/images',
                        help='数据集根目录路径')
    parser.add_argument('--label_path', type=str, 
                        # default='/home/zhongxinyu/capstone/HK/datasets/labels_test.csv',
                        default='/home/zhongxinyu/capstone/HK/datasets/all_image_scores_labels.csv',
                        help='训练标签路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='训练批次大小 (默认: 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (默认: 0.0001)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数 (默认: 20)')
    parser.add_argument('--pretrained', type=str, 
                        default='/home/zhongxinyu/capstone/HK/DeepLabV3Plus-Pytorch-master/checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar',
                        help='预训练权重路径')
    parser.add_argument('--save_path', type=str, default='/home/zhongxinyu/capstone/HK/DeepLabV3Plus-Pytorch-master/checkpoints/0622_best_regression_model.pth',
                        help='模型保存路径 (默认: /home/zhongxinyu/capstone/HK/DeepLabV3Plus-Pytorch-master/checkpoints/total_sample_deeplabv3plus_resnet101_regression_best_model.pth)')
    parser.add_argument('--freeze_backbone', default=False, action='store_true',
                        help='是否冻结骨干网络')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                       help='warmup轮数 (默认: 5)')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                       help='最小学习率 (默认: 1e-7)')
    # 多GPU参数
    parser.add_argument('--nproc_per_node', type=int, default=torch.cuda.device_count(),
                        help='每个节点使用的GPU数量，默认使用所有可用GPU')
    parser.add_argument('--dist-url', type=str, default='env://',
                        help='分布式训练的URL')
    parser.add_argument('--rank', type=int, default=0,
                        help='进程排名')
    parser.add_argument('--world-size', type=int, default=1,
                        help='总进程数')
    # 日志参数
    parser.add_argument('--log_dir', type=str, default='/home/zhongxinyu/capstone/HK/DeepLabV3Plus-Pytorch-master/logs',
                        help='日志保存目录')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='记录日志的步数间隔')
    # 新增参数
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help='主节点地址')
    parser.add_argument('--master_port', type=int, default=12345,
                        help='主节点端口')
    return parser.parse_args()

# 新增调度器创建函数
def create_scheduler(optimizer, warmup_epochs, num_epochs, min_lr):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
  
    return LambdaLR(optimizer, lr_lambda)

# 2. 加载预训练权重
def load_pretrained_weights(model, path_to_pth):
    state_dict = torch.load(path_to_pth, map_location='cpu')['model_state']
    # 创建新的state_dict，只包含特征提取部分的权重
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'regression_layer' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

# 3. 冻结特征提取层
def freeze_feature_extractor(model):
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    return model

# 验证函数
def validate_model(model, val_loader, criterion, device, logger=None):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['image'].to(device)
            labels = batch['score'].to(device).squeeze()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    if logger:
        logger.info(f'Validation Loss: {avg_val_loss:.4f}')
    return avg_val_loss

def train_model(gpu, ngpus_per_node, args):
    # 设置环境变量
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    
    rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=rank
    )
    
    set_seed(123 + gpu)  # 每个GPU使用不同的随机种子
    
    # 仅在主进程中记录日志和保存模型
    is_master = rank == 0
    
    # 设置日志
    if is_master:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.log_dir, args.exp_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'train_{timestamp}.log')
        
        logger = setup_logger('train_logger', log_file, rank=rank)
        if logger:
            # 记录所有训练参数
            logger.info('=' * 50)
            logger.info('Training Parameters:')
            for arg, value in sorted(vars(args).items()):
                logger.info(f'{arg}: {value}')
            logger.info(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            logger.info(f'PyTorch Version: {torch.__version__}')
            logger.info(f'GPU Count: {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                logger.info(f'GPU {i}: {torch.cuda.get_device_name(i)}')
            logger.info('=' * 50)
    else:
        logger = None
    
    # 初始化模型
    model = PredictRegressionModel(num_classes=19)
    model = load_pretrained_weights(model, args.pretrained)
    if logger:
        logger.info(f"预训练权重加载完成: {args.pretrained}")

    # 冻结特征层（根据参数决定）
    if args.freeze_backbone:
        model = freeze_feature_extractor(model)
        if logger:
            logger.info("特征提取层已冻结")

    # 设备配置
    torch.cuda.set_device(gpu)
    model = model.to(gpu)
    
    # 包装模型用于分布式训练
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    full_dataset = RegressionDataset(
        image_path=args.image_path,
        label_path=args.label_path,
        transform=transform,
        transform_mode=False,
    )

    # 数据集划分
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 为分布式训练创建采样器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=False
    )

    # 创建数据加载器
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(123 + gpu)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.world_size,  # 调整batch size
        shuffle=False,  # 采样器已经处理了shuffle
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        sampler=train_sampler
    )
  
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // args.world_size,  # 调整batch size
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        sampler=val_sampler
    )

    # 训练配置
    criterion = nn.MSELoss()
    # 移除线性缩放学习率（如需保留分布式训练的学习率缩放需谨慎处理）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
  
    # 记录训练指标
    if is_master:
        train_losses = []  # 记录训练损失
        val_losses = []    # 记录验证损失
        best_loss = np.inf
    
    # 开始训练
    for epoch in range(args.epochs):
        # 设置epoch以确保采样器的shuffle正确工作
        train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0.0
        
        # 记录当前epoch的学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 训练阶段
        if is_master:
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        else:
            progress_bar = train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch['image'].to(gpu)
            labels = batch['score'].to(gpu).squeeze()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # 记录每一步的损失
            if is_master and logger and args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                step_loss = loss.item()
                global_step = epoch * len(train_loader) + batch_idx + 1
                logger.info(f'Step {global_step} | Batch {batch_idx+1}/{len(train_loader)} | Train Loss: {step_loss:.4f} | LR: {current_lr:.2e}')

        # 验证阶段
        val_loss = validate_model(model, val_loader, criterion, gpu, logger)
        avg_train_loss = train_loss / len(train_loader)
        
        # 同步所有进程的验证损失
        val_loss_tensor = torch.tensor(val_loss, device=gpu)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / args.world_size
        
        # 仅在主进程中记录损失和保存模型
        if is_master:
            # 记录损失
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)

            current_lr = optimizer.param_groups[0]['lr']
          
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.module.state_dict(), args.save_path)
                if logger:
                    logger.info(f"New best model saved with val_loss: {val_loss:.4f}")

            if logger:
                logger.info(f'Epoch {epoch+1}/{args.epochs} Summary:')
                logger.info(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val Loss: {best_loss:.4f}')
                logger.info('-' * 50)
    
    # 仅在主进程中绘制并保存损失图
    if is_master and logger:
        plt.figure(figsize=(8, 5))
        
        # 仅保留损失曲线子图
        plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, args.epochs + 1), val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        
        # 保存图像
        plot_dir = '/home/zhongxinyu/capstone/HK/DeepLabV3Plus-Pytorch-master/train_loss_curves/'
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'{args.exp_name}_{timestamp}_loss_lr_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"损失和学习率曲线图已保存至: {plot_path}")
        logger.info(f"模型训练完成，最佳模型已保存为 {args.save_path}")
        logger.info('=' * 50)
    
    # 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    args.world_size = args.nproc_per_node * args.world_size
    
    # 创建保存模型和日志的目录
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 使用spawn启动多个进程
    mp.spawn(
        train_model,
        args=(args.nproc_per_node, args),
        nprocs=args.nproc_per_node,
        join=True
    )