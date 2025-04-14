import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from d2l import torch as d2l
import numpy as np

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 生成数据
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 定义L1正则化惩罚项
def l1_penalty(w):
    return torch.sum(torch.abs(w))

# 定义L2正则化惩罚项
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 训练函数
def train(lambd, penalty_type='l2'):
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    
    # 选择正则化类型
    penalty_fn = l1_penalty if penalty_type == 'l1' else l2_penalty
    
    # 记录训练和测试损失
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 计算损失，包括正则化项
            l = loss(net(X), y) + lambd * penalty_fn(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        
        # 记录每个epoch的训练和测试损失
        train_loss = d2l.evaluate_loss(net, train_iter, loss)
        test_loss = d2l.evaluate_loss(net, test_iter, loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    # 计算权重的稀疏度（非零元素的比例）
    sparsity = (w == 0).float().mean().item()
    print(f'{penalty_type.upper()} Regularization:')
    print(f'Weight L2 norm: {torch.norm(w).item():.4f}')
    print(f'Weight sparsity: {sparsity:.4f}')
    
    return train_losses, test_losses, w

# 运行实验
def run_experiments():
    # 测试不同的正则化强度
    lambdas = [0, 1, 3, 10]
    results = {}
    
    for lambd in lambdas:
        print(f"\nTesting lambda = {lambd}")
        
        # L1正则化
        print("\nL1 Regularization:")
        l1_train_losses, l1_test_losses, l1_w = train(lambd, 'l1')
        
        # L2正则化
        print("\nL2 Regularization:")
        l2_train_losses, l2_test_losses, l2_w = train(lambd, 'l2')
        
        results[lambd] = {
            'l1': {'train': l1_train_losses, 'test': l1_test_losses, 'w': l1_w},
            'l2': {'train': l2_train_losses, 'test': l2_test_losses, 'w': l2_w}
        }
    
    return results

# 绘制结果
def plot_results(results):
    plt.figure(figsize=(15, 10))
    
    # 绘制训练和测试损失
    for i, lambd in enumerate(results.keys()):
        plt.subplot(2, 2, i+1)
        plt.plot(results[lambd]['l1']['train'], label='L1 Train')
        plt.plot(results[lambd]['l1']['test'], label='L1 Test')
        plt.plot(results[lambd]['l2']['train'], label='L2 Train')
        plt.plot(results[lambd]['l2']['test'], label='L2 Test')
        plt.title(f'Lambda = {lambd}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('l1_vs_l2_comparison.png')
    plt.close()

if __name__ == "__main__":
    print("Running L1 vs L2 regularization experiments...")
    results = run_experiments()
    plot_results(results)
    print("\nExperiments completed. Results have been saved to 'l1_vs_l2_comparison.png'") 