import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np

def train_model(batch_size, num_epochs, learning_rate, weight_decay=0):
    # 加载数据
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    
    # 定义模型
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    
    # 初始化权重
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    
    net.apply(init_weights)
    
    # 定义损失函数和优化器
    loss_func = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
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

# 实验1：增加迭代周期数，观察过拟合
print("\n实验1：增加迭代周期数，观察过拟合")
train_losses1, test_losses1, train_acc1, test_acc1 = train_model(
    batch_size=256, 
    num_epochs=30,  # 增加迭代周期
    learning_rate=0.1
)

# 实验2：使用权重衰减（L2正则化）来减少过拟合
print("\n实验2：使用权重衰减（L2正则化）来减少过拟合")
train_losses2, test_losses2, train_acc2, test_acc2 = train_model(
    batch_size=256, 
    num_epochs=30, 
    learning_rate=0.1,
    weight_decay=0.01  # 添加权重衰减
)

# 实验3：使用早停（Early Stopping）来防止过拟合
print("\n实验3：使用早停（Early Stopping）来防止过拟合")
train_losses3, test_losses3, train_acc3, test_acc3 = train_model(
    batch_size=256, 
    num_epochs=30, 
    learning_rate=0.1
)

# 绘制结果
plt.figure(figsize=(15, 10))

# 绘制损失
plt.subplot(2, 2, 1)
plt.plot(train_losses1, label='训练损失')
plt.plot(test_losses1, label='验证损失')
plt.xlabel('迭代周期')
plt.ylabel('损失')
plt.title('实验1：增加迭代周期数')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(train_losses2, label='训练损失')
plt.plot(test_losses2, label='验证损失')
plt.xlabel('迭代周期')
plt.ylabel('损失')
plt.title('实验2：使用权重衰减')
plt.legend()
plt.grid(True)

# 绘制准确率
plt.subplot(2, 2, 3)
plt.plot(train_acc1, label='训练准确率')
plt.plot(test_acc1, label='验证准确率')
plt.xlabel('迭代周期')
plt.ylabel('准确率 (%)')
plt.title('实验1：增加迭代周期数')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(train_acc2, label='训练准确率')
plt.plot(test_acc2, label='验证准确率')
plt.xlabel('迭代周期')
plt.ylabel('准确率 (%)')
plt.title('实验2：使用权重衰减')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('overfitting_experiments.png')
plt.show()

# 分析早停点
best_epoch = np.argmin(test_losses3)
print(f"\n早停分析：")
print(f"最佳验证损失出现在第 {best_epoch+1} 个迭代周期")
print(f"最佳验证准确率: {test_acc3[best_epoch]:.2f}%")
print(f"对应的训练准确率: {train_acc3[best_epoch]:.2f}%") 