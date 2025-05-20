import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# 设置随机种子
torch.manual_seed(42)  # 这里使用42作为示例，请替换为你的学号后两位

# 第一部分：创建对角线边缘图像
def create_diagonal_edge_image(size=8):
    X = torch.zeros((size, size))
    for i in range(size):
        X[i, i] = 1
    return X

# 第二部分：自定义CNN模型
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # 输入图像大小: 32x32x3
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输出: 32x32x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),  # 1x1 卷积层，输出: 32x32x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 16x16x64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输出: 16x16x128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 8x8x128
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 输出: 8x8x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 4x4x256
        )
        
        # 计算全连接层的输入维度
        # 最后一层卷积输出: 4x4x256
        # 展平后的维度: 4 * 4 * 256 = 4096
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),  # 4096 -> 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 打印进度
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100 * correct/total:.2f}%')
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        losses.append(epoch_loss)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return losses

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    return accuracy, all_predictions, all_labels

def plot_loss_curve(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')
    plt.close()

def visualize_predictions(model, test_loader, device, class_names, num_samples=5):
    model.eval()
    plt.figure(figsize=(15, 3))
    
    with torch.no_grad():
        # 获取一批测试数据
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        
        # 获取预测结果
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # 显示前num_samples个样本
        for i in range(min(num_samples, len(images))):
            plt.subplot(1, num_samples, i + 1)
            
            # 反归一化图像
            img = images[i].cpu().numpy().transpose((1, 2, 0))
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            plt.imshow(img)
            color = 'green' if predicted[i] == labels[i] else 'red'
            plt.title(f'True: {class_names[labels[i]]}\nPred: {class_names[predicted[i]]}', 
                     color=color)
            plt.axis('off')
    
    plt.tight_layout()
    # plt.savefig('prediction_samples.png')
    plt.show()

def main():
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 第一部分：对角线边缘图像实验
    X = create_diagonal_edge_image()
    K = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    
    # 进行卷积操作
    X_reshaped = X.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
    K_reshaped = K.unsqueeze(0).unsqueeze(0)
    
    # 原始卷积
    conv_result = nn.functional.conv2d(X_reshaped, K_reshaped, padding=1)
    print("原始卷积结果：")
    print(conv_result.squeeze())
    
    # 转置X后的卷积
    X_transposed = X.t()
    X_transposed_reshaped = X_transposed.unsqueeze(0).unsqueeze(0)
    conv_result_transposed_X = nn.functional.conv2d(X_transposed_reshaped, K_reshaped, padding=1)
    print("\n转置X后的卷积结果：")
    print(conv_result_transposed_X.squeeze())
    
    # 转置K后的卷积
    K_transposed = K.t()
    K_transposed_reshaped = K_transposed.unsqueeze(0).unsqueeze(0)
    conv_result_transposed_K = nn.functional.conv2d(X_reshaped, K_transposed_reshaped, padding=1)
    print("\n转置K后的卷积结果：")
    print(conv_result_transposed_K.squeeze())

    # 第二部分：农业害虫分类任务
    # 加载数据集
    print("加载数据集...")
    dataset = torchvision.datasets.ImageFolder(root='./agriculture_pests', transform=transform)
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数量: {len(dataset.classes)}")
    print(f"类别名称: {dataset.classes}")
    
    # 划分训练集和测试集 (70% 训练, 30% 测试)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    num_classes = len(dataset.classes)
    model = CustomCNN(num_classes).to(device)
    print("模型结构:")
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("\n开始训练...")
    losses = train_model(model, train_loader, criterion, optimizer, device)
    
    # 绘制损失曲线
    plot_loss_curve(losses)
    print("损失曲线已保存为 'loss_curve.png'")
    
    # 评估模型
    print("\n评估模型...")
    accuracy, predictions, labels = evaluate_model(model, test_loader, device)
    
    # 保存模型参数
    torch.save(model.state_dict(), 'pest_classifier.pth')
    print("模型参数已保存为 'pest_classifier.pth'")
    
    # 加载模型参数
    loaded_model = CustomCNN(num_classes).to(device)
    loaded_model.load_state_dict(torch.load('pest_classifier.pth'))
    print("模型参数已加载")
    
    # 可视化测试结果
    print("\n生成测试结果可视化...")
    visualize_predictions(loaded_model, test_loader, device, dataset.classes)
    print("测试结果可视化已保存为 'prediction_samples.png'")

if __name__ == '__main__':
    main() 