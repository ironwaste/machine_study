import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from d2l import torch as d2l
import numpy as np
import time

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 加载Fashion-MNIST数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义dropout层函数
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

# 定义准确率计算函数
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 定义评估准确率函数
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 定义基础神经网络模型
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 dropout1, dropout2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
        self.dropout1 = dropout1
        self.dropout2 = dropout2

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training:
            H1 = dropout_layer(H1, self.dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout2)
        out = self.lin3(H2)
        return out

# 训练函数
def train_model(net, train_iter, test_iter, num_epochs, lr=0.5):
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        # 训练
        net.train()
        train_metric = d2l.Accumulator(3)
        for X, y in train_iter:
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.mean().backward()
            trainer.step()
            train_metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        
        train_loss = train_metric[0] / train_metric[2]
        train_acc = train_metric[1] / train_metric[2]
        
        # 测试
        test_acc = evaluate_accuracy(net, test_iter)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    return train_losses, train_accs, test_accs

# 实验1：比较不同的dropout概率配置
def experiment_dropout_configs():
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    num_epochs = 10
    
    # 测试不同的dropout配置
    configs = [
        {'name': 'No Dropout', 'dropout1': 0.0, 'dropout2': 0.0},
        {'name': 'Standard (0.2, 0.5)', 'dropout1': 0.2, 'dropout2': 0.5},
        {'name': 'Swapped (0.5, 0.2)', 'dropout1': 0.5, 'dropout2': 0.2},
        {'name': 'High (0.5, 0.5)', 'dropout1': 0.5, 'dropout2': 0.5},
        {'name': 'Low (0.1, 0.1)', 'dropout1': 0.1, 'dropout2': 0.1}
    ]
    
    results = {}
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 config['dropout1'], config['dropout2'])
        train_losses, train_accs, test_accs = train_model(
            net, train_iter, test_iter, num_epochs)
        
        results[config['name']] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_accs': test_accs
        }
    
    return results

# 实验2：比较不同训练轮数的效果
def experiment_training_epochs():
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    
    # 测试不同的训练轮数
    epochs = [10, 20, 30]
    results = {}
    
    for num_epochs in epochs:
        print(f"\nTesting with {num_epochs} epochs")
        
        # 不使用dropout
        print("Without Dropout:")
        net_no_dropout = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                            0.0, 0.0)
        train_losses_no, train_accs_no, test_accs_no = train_model(
            net_no_dropout, train_iter, test_iter, num_epochs)
        
        # 使用dropout
        print("\nWith Dropout (0.2, 0.5):")
        net_with_dropout = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                             0.2, 0.5)
        train_losses_with, train_accs_with, test_accs_with = train_model(
            net_with_dropout, train_iter, test_iter, num_epochs)
        
        results[num_epochs] = {
            'no_dropout': {
                'train_losses': train_losses_no,
                'train_accs': train_accs_no,
                'test_accs': test_accs_no
            },
            'with_dropout': {
                'train_losses': train_losses_with,
                'train_accs': train_accs_with,
                'test_accs': test_accs_with
            }
        }
    
    return results

# 绘制结果
def plot_results(results, experiment_type):
    if experiment_type == 'dropout_configs':
        plt.figure(figsize=(15, 5))
        
        # 绘制训练准确率
        plt.subplot(1, 2, 1)
        for name, data in results.items():
            plt.plot(data['train_accs'], label=name)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # 绘制测试准确率
        plt.subplot(1, 2, 2)
        for name, data in results.items():
            plt.plot(data['test_accs'], label=name)
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('dropout_configs_comparison.png')
        plt.close()
        
    elif experiment_type == 'training_epochs':
        plt.figure(figsize=(15, 10))
        
        for i, (epochs, data) in enumerate(results.items()):
            plt.subplot(2, 2, i+1)
            plt.plot(data['no_dropout']['test_accs'], label='No Dropout')
            plt.plot(data['with_dropout']['test_accs'], label='With Dropout')
            plt.title(f'{epochs} Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Test Accuracy')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_epochs_comparison.png')
        plt.close()

if __name__ == "__main__":
    print("Running Dropout Experiments...")
    
    print("\nExperiment 1: Comparing different dropout configurations")
    dropout_results = experiment_dropout_configs()
    plot_results(dropout_results, 'dropout_configs')
    
    print("\nExperiment 2: Comparing different training epochs")
    epochs_results = experiment_training_epochs()
    plot_results(epochs_results, 'training_epochs')
    
    print("\nExperiments completed. Results have been saved as PNG files.") 