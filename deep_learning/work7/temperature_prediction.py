import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 数据加载和预处理
def load_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 将日期列转换为datetime类型
    df['date'] = pd.to_datetime(df['date'])
    
    # 筛选2021年7月1日到12月30日的数据
    mask = (df['date'] >= '2021-07-01') & (df['date'] <= '2021-12-30')
    train_data = df.loc[mask]
    
    # 筛选2021年12月31日的数据作为测试集
    test_mask = (df['date'] == '2021-12-31')
    test_data = df.loc[test_mask]
    
    return train_data, test_data

# 创建时间序列数据集
class TemperatureDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

# 定义GRU模型
class TemperatureGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TemperatureGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU前向传播
        out, _ = self.gru(x, h0)
        
        # 只使用最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

def predict_temperature(model, test_data, scaler, seq_length, device):
    model.eval()
    predictions = []
    
    # 准备输入序列
    input_seq = test_data[:seq_length].values.reshape(1, seq_length, 1)
    input_seq = torch.FloatTensor(input_seq).to(device)
    
    # 预测未来100个点
    with torch.no_grad():
        for _ in range(100):
            output = model(input_seq)
            predictions.append(output.item())
            
            # 更新输入序列
            input_seq = torch.cat([input_seq[:, 1:, :], output.unsqueeze(0).unsqueeze(0)], dim=1)
    
    # 反归一化预测结果
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()

def main():
    # 加载数据
    train_data, test_data = load_data('weather.csv')
    
    # 数据标准化
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[['temperature']].values)
    test_scaled = scaler.transform(test_data[['temperature']].values)
    
    # 设置参数
    seq_length = 24  # 使用24个时间步作为输入
    batch_size = 32
    hidden_size = 64
    num_layers = 2
    num_epochs = 100
    learning_rate = 0.01
    
    # 创建数据集和数据加载器
    train_dataset = TemperatureDataset(train_scaled, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = TemperatureGRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("开始训练模型...")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    
    # 预测温度
    print("开始预测温度...")
    predictions = predict_temperature(model, test_scaled, scaler, seq_length, device)
    
    # 计算RMSE
    actual = test_data['temperature'].values[:100]  # 取前100个实际值
    rmse = math.sqrt(mean_squared_error(actual, predictions))
    print(f'RMSE: {rmse:.2f}')
    
    # 绘制预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='实际温度')
    plt.plot(predictions, label='预测温度')
    plt.title('温度预测结果')
    plt.xlabel('时间点')
    plt.ylabel('温度')
    plt.legend()
    plt.savefig('temperature_prediction.png')
    plt.close()

if __name__ == '__main__':
    main() 