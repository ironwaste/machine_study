import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from d2l import torch as d2l
import numpy as np
import time

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 加载数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义MLP模型类
class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens, activation_fn=nn.ReLU(), init_method='normal'):
        super(MLP, self).__init__()
        self.num_hiddens = num_hiddens
        self.activation_fn = activation_fn
        
        # 构建网络层
        layers = []
        # 添加展平层
        layers.append(nn.Flatten())
        
        # 构建隐藏层
        prev_dim = num_inputs
        for num_hidden in num_hiddens:
            layers.append(nn.Linear(prev_dim, num_hidden))
            layers.append(activation_fn)
            prev_dim = num_hidden
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_outputs))
        self.net = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights(init_method)
    
    def _init_weights(self, init_method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_method == 'normal':
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif init_method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_method == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_method == 'zeros':
                    nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)

# 训练函数
def train_mlp(num_epochs, learning_rate, model, train_iter, test_iter):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, y in train_iter:
            optimizer.zero_grad()
            output = model(X)
            l = loss(output, y)
            l.backward()
            optimizer.step()
            total_loss += l.item()
        
        avg_loss = total_loss / len(train_iter)
        train_losses.append(avg_loss)
        
        # 评估测试集准确率
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for X, y in test_iter:
                output = model(X)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = correct / total
        test_accuracies.append(accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    
    return train_losses, test_accuracies

# 实验1：测试不同的隐藏层配置
def experiment_hidden_layers():
    num_epochs = 10
    learning_rates = [0.01, 0.05, 0.1]
    hidden_configs = [
        [256],
        [256, 128],
        [256, 128, 64],
        [256, 128, 64, 32]
    ]
    
    results = {}
    for lr in learning_rates:
        for hidden in hidden_configs:
            print(f"\nTesting hidden layers {hidden} with learning rate {lr}")
            model = MLP(784, 10, hidden)
            train_losses, test_accuracies = train_mlp(num_epochs, lr, model, train_iter, test_iter)
            results[f"hidden_{len(hidden)}_lr_{lr}"] = {
                "train_losses": train_losses,
                "test_accuracies": test_accuracies
            }
    
    return results

# 实验2：测试不同的激活函数
def experiment_activation_functions():
    num_epochs = 10
    learning_rate = 0.1
    hidden = [256, 128]
    activation_fns = {
        'ReLU': nn.ReLU(),
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh(),
        'LeakyReLU': nn.LeakyReLU(0.1),
        'ELU': nn.ELU()
    }
    
    results = {}
    for name, activation_fn in activation_fns.items():
        print(f"\nTesting {name} activation function")
        model = MLP(784, 10, hidden, activation_fn=activation_fn)
        train_losses, test_accuracies = train_mlp(num_epochs, learning_rate, model, train_iter, test_iter)
        results[name] = {
            "train_losses": train_losses,
            "test_accuracies": test_accuracies
        }
    
    return results

# 实验3：测试不同的权重初始化方法
def experiment_weight_initialization():
    num_epochs = 10
    learning_rate = 0.1
    hidden = [256, 128]
    init_methods = ['normal', 'xavier', 'kaiming', 'zeros']
    
    results = {}
    for init_method in init_methods:
        print(f"\nTesting {init_method} weight initialization")
        model = MLP(784, 10, hidden, init_method=init_method)
        train_losses, test_accuracies = train_mlp(num_epochs, learning_rate, model, train_iter, test_iter)
        results[init_method] = {
            "train_losses": train_losses,
            "test_accuracies": test_accuracies
        }
    
    return results

# 绘制结果
def plot_results(results, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data["test_accuracies"], label=name)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

# 运行所有实验
if __name__ == "__main__":
    print("Running experiment 1: Hidden Layers")
    hidden_results = experiment_hidden_layers()
    plot_results(hidden_results, "Hidden Layers Comparison", "Epoch", "Test Accuracy")
    
    print("\nRunning experiment 2: Activation Functions")
    activation_results = experiment_activation_functions()
    plot_results(activation_results, "Activation Functions Comparison", "Epoch", "Test Accuracy")
    
    print("\nRunning experiment 3: Weight Initialization")
    init_results = experiment_weight_initialization()
    plot_results(init_results, "Weight Initialization Comparison", "Epoch", "Test Accuracy")
    
    print("\nAll experiments completed. Results have been saved as PNG files.") 