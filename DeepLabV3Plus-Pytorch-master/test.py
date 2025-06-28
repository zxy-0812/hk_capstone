import torch
import torch.nn as nn
from network.model import PredictRegressionModel  # 确保模型定义可用
from datasets.voc import RegressionDataset  # 使用您已定义的数据集
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ##################################### 配置参数 #####################################
# 模型参数
NUM_CLASSES = 19  # 与训练时保持一致
MODEL_PATH = '/home/zhongxinyu/capstone/HK/DeepLabV3Plus-Pytorch-master/checkpoints/0622_best_regression_model.pth'  # 模型权重路径

# 数据集参数
ROOT_DIR = '/home/zhongxinyu/capstone/HK/datasets/total_images'  # 数据集根目录
CSV_FILE = '/home/zhongxinyu/capstone/HK/datasets/all_test_image_scores_labels.csv'  # 测试用CSV文件
BATCH_SIZE = 8
IMAGE_SIZE = (640, 640)  # 输入图像尺寸

# 输出保存路径
OUTPUT_DIR = '/home/zhongxinyu/capstone/HK/DeepLabV3Plus-Pytorch-master/0622_test_all_labels_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ##################################### 1. 加载模型 #####################################
def load_model(model_path, num_classes=19):
    """加载训练好的回归模型"""
    model = PredictRegressionModel(num_classes=num_classes)
    
    # 加载权重
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    # 设置为评估模式
    model.eval()
    
    # 使用GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"模型加载完成，运行在: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    return model, device

# ##################################### 2. 构建测试数据集 #####################################
def build_test_dataloader(root_dir, csv_file, image_size=(640, 640), batch_size=4):
    """构建测试数据加载器"""
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    test_dataset = RegressionDataset(
        image_path=root_dir,
        label_path=csv_file,
        transform=transform,
        transform_mode=False,
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    return test_loader, test_dataset.mean_score, test_dataset.std_score

# ##################################### 3. 模型评估函数 #####################################
def evaluate_model(model, test_loader, device, mean_score, std_score):
    """评估模型性能并收集预测结果"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # 禁用梯度计算
        for batch in test_loader:
            # 获取数据
            inputs = batch['image'].to(device)
            labels = batch['score'].cpu().numpy()
            
            # 模型预测（确保输出为 numpy 数组）
            outputs = model(inputs).cpu().numpy()
            
            # 确保输出为数组（即使只有一个样本）
            if np.isscalar(outputs):
                outputs = np.array([outputs])
            elif outputs.ndim == 0:
                outputs = outputs.reshape(1)
            
            # 确保标签为数组
            if np.isscalar(labels):
                labels = np.array([labels])
            
            # 反归一化
            all_preds.extend(outputs * std_score + mean_score)
            all_labels.extend(labels * std_score + mean_score)
    
    # 转换为 numpy 数组
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    # 计算评估指标
    mse = np.mean((preds - labels) ** 2)
    mae = np.mean(np.abs(preds - labels))
    corr = np.corrcoef(preds, labels)[0, 1]
    
    print(f"\n评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"皮尔逊相关系数: {corr:.4f}")
    
    return preds, labels, {'MSE': mse, 'MAE': mae, 'Correlation': corr}

# ##################################### 4. 可视化函数 #####################################
def visualize_predictions(preds, labels, images, output_dir):
    """可视化预测结果与真实值"""
    num_samples = min(6, images.shape[0])  # 使用 images 的实际数量（如 batch_size=4）
    plt.figure(figsize=(15, 10))
    
    for i in range(num_samples):
        # 获取图像
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])  # 反归一化
        
        # 确保数据范围 [0, 1]
        img = np.clip(img, 0, 1)  # 防止 imshow 警告
        
        # 创建子图
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(f"Pred: {preds[i]:.2f}\nTrue: {labels[i]:.2f}")
        plt.axis('off')
    
    # 保存可视化结果
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_visualization.png'))
    plt.close()
    print(f"\n预测可视化已保存至: {os.path.join(output_dir, 'prediction_visualization.png')}")

# ##################################### 5. 保存预测结果 #####################################
def save_predictions(preds, labels, images, output_dir, dataset):
    """保存预测结果和原始图像名称"""
    # 获取图像名称
    image_names = [os.path.basename(img_path) for img_path in dataset.data_df.iloc[:, 0]]
    
    # 创建DataFrame
    results_df = pd.DataFrame({
        'image_name': image_names[:len(preds)],
        'true_score': labels,
        'predicted_score': preds
    })
    
    # 保存为CSV
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print(f"\n预测结果已保存至: {os.path.join(output_dir, 'predictions.csv')}")
    
    return results_df

# ##################################### 6. 散点图可视化 #####################################
def plot_scatter(preds, labels, output_dir):
    """绘制预测值与真实值的散点图"""
    plt.figure(figsize=(8, 8))
    plt.scatter(labels, preds, alpha=0.6)
    plt.plot([min(labels), max(labels)], [min(labels), max(labels)], 'r--')
    plt.xlabel('True Score')
    plt.ylabel('Predicted Score')
    plt.title('True vs Predicted Scores')
    
    # 保存散点图
    plt.savefig(os.path.join(output_dir, 'scatter_plot.png'))
    # plt.close()
    print(f"\n散点图已保存至: {os.path.join(output_dir, 'scatter_plot.png')}")

# ##################################### 主程序 #####################################
if __name__ == "__main__":
    # 1. 加载模型
    model, device = load_model(MODEL_PATH, NUM_CLASSES)
    
    # 2. 构建测试数据集
    test_loader, mean_score, std_score = build_test_dataloader(ROOT_DIR, CSV_FILE, IMAGE_SIZE, BATCH_SIZE)
    
    # 3. 获取第一个batch用于可视化
    sample_batch = next(iter(test_loader))
    sample_images = sample_batch['image']
    sample_labels = sample_batch['score'].numpy()
    
    # 4. 模型评估
    preds, labels, metrics = evaluate_model(model, test_loader, device, mean_score, std_score)

    # 只取第一个 batch 用于可视化（4 个样本）
    batch_preds = preds[:sample_images.size(0)]   # 取前 4 个预测
    batch_labels = labels[:sample_labels.shape[0]] # 取前 4 个标签
    
    
    # 5. 结果可视化
    visualize_predictions(preds, labels, sample_images, OUTPUT_DIR)
    plot_scatter(preds, labels, OUTPUT_DIR)
    
    # # 6. 保存预测结果
    results_df = save_predictions(preds, labels, sample_images, OUTPUT_DIR, test_loader.dataset)
    
    # # 7. 打印部分预测结果
    print("\n前10个预测结果:")
    print(results_df.head(10))