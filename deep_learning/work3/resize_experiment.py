import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

# 自定义数据集类，用于调整图像大小
class ResizedFashionMNIST:
    def __init__(self, root, train, download, transform, target_transform=None):
        self.dataset = FashionMNIST(root=root, train=train, download=download, 
                                    transform=transform, target_transform=target_transform)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

# 加载调整大小后的数据
def load_resized_data(batch_size, image_size=64):
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 调整图像大小为64×64
        transforms.ToTensor()
    ])
    
    # 加载训练集
    train_dataset = ResizedFashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 加载测试集
    test_dataset = ResizedFashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_iter, test_iter

def train_model(batch_size, num_epochs, learning_rate, image_size=64):
    # 加载调整大小后的数据
    train_iter, test_iter = load_resized_data(batch_size, image_size)
    
    # 定义模型 - 注意输入维度变化
    input_dim = image_size * image_size  # 64×64 = 4096
    net = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, 10))
    
    # 初始化权重
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    
    net.apply(init_weights)
    
    # 定义损失函数和优化器
    loss_func = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    
    # 训练
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练
        train_loss_sum = 0
        train_batch_count = 0
        train_correct = 0
        train_total = 0
        
        for x, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(x)
            loss = loss_func(y_hat, y)
            loss.mean().backward()
            optimizer.step()
            train_loss_sum += loss.mean().detach().numpy()
            train_batch_count += 1
            
            # 计算训练准确率
            _, predicted = torch.max(y_hat.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        avg_train_loss = train_loss_sum / train_batch_count
        train_losses.append(avg_train_loss)
        train_accuracy = 100 * train_correct / train_total
        train_accuracies.append(train_accuracy)
        
        # 验证
        test_loss_sum = 0
        test_batch_count = 0
        test_correct = 0
        test_total = 0
        
        for x_val, y_val in test_iter:
            y_pre = net(x_val)
            loss_val = loss_func(y_pre, y_val)
            test_loss_sum += loss_val.mean().detach().numpy()
            test_batch_count += 1
            
            # 计算测试准确率
            _, predicted = torch.max(y_pre.data, 1)
            test_total += y_val.size(0)
            test_correct += (predicted == y_val).sum().item()
        
        avg_test_loss = test_loss_sum / test_batch_count
        test_losses.append(avg_test_loss)
        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
        print(f'  验证损失: {avg_test_loss:.4f}, 验证准确率: {test_accuracy:.2f}%')
    
    return train_losses, test_losses, train_accuracies, test_accuracies

# 实验1：使用原始大小 (28×28)
print("\n实验1：使用原始大小 (28×28)")
train_losses1, test_losses1, train_acc1, test_acc1 = train_model(
    batch_size=256, 
    num_epochs=10, 
    learning_rate=0.1,
    image_size=28
)

# 实验2：使用调整后的大小 (64×64)
print("\n实验2：使用调整后的大小 (64×64)")
train_losses2, test_losses2, train_acc2, test_acc2 = train_model(
    batch_size=256, 
    num_epochs=10, 
    learning_rate=0.1,
    image_size=64
)

# 绘制结果
plt.figure(figsize=(15, 10))

# 绘制损失
plt.subplot(2, 2, 1)
plt.plot(train_losses1, label='训练损失')
plt.plot(test_losses1, label='验证损失')
plt.xlabel('迭代周期')
plt.ylabel('损失')
plt.title('实验1：原始大小 (28×28)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(train_losses2, label='训练损失')
plt.plot(test_losses2, label='验证损失')
plt.xlabel('迭代周期')
plt.ylabel('损失')
plt.title('实验2：调整后大小 (64×64)')
plt.legend()
plt.grid(True)

# 绘制准确率
plt.subplot(2, 2, 3)
plt.plot(train_acc1, label='训练准确率')
plt.plot(test_acc1, label='验证准确率')
plt.xlabel('迭代周期')
plt.ylabel('准确率 (%)')
plt.title('实验1：原始大小 (28×28)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(train_acc2, label='训练准确率')
plt.plot(test_acc2, label='验证准确率')
plt.xlabel('迭代周期')
plt.ylabel('准确率 (%)')
plt.title('实验2：调整后大小 (64×64)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('resize_experiments.png')
plt.show()

# 比较结果
print("\n结果比较：")
print(f"原始大小 (28×28) - 最终训练准确率: {train_acc1[-1]:.2f}%, 最终验证准确率: {test_acc1[-1]:.2f}%")
print(f"调整后大小 (64×64) - 最终训练准确率: {train_acc2[-1]:.2f}%, 最终验证准确率: {test_acc2[-1]:.2f}%") 