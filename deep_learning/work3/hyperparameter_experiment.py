import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

def train_model(batch_size, num_epochs, learning_rate):
    # Load data
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    
    # Define model
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    
    # Initialize weights
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    
    net.apply(init_weights)
    
    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    
    # Training
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # Training
        train_loss_sum = 0
        train_batch_count = 0
        
        for x, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(x)
            loss = loss_func(y_hat, y)
            loss.mean().backward()
            optimizer.step()
            train_loss_sum += loss.mean().detach().numpy()
            train_batch_count += 1
        
        avg_train_loss = train_loss_sum / train_batch_count
        train_losses.append(avg_train_loss)
        
        # Validation
        test_loss_sum = 0
        test_batch_count = 0
        
        for x_val, y_val in test_iter:
            y_pre = net(x_val)
            loss_val = loss_func(y_pre, y_val)
            test_loss_sum += loss_val.mean().detach().numpy()
            test_batch_count += 1
        
        avg_test_loss = test_loss_sum / test_batch_count
        test_losses.append(avg_test_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Training loss: {avg_train_loss:.4f}')
        print(f'  Validation loss: {avg_test_loss:.4f}')
    
    return train_losses, test_losses

# Experiment with different hyperparameters
experiments = [
    {'batch_size': 128, 'num_epochs': 10, 'learning_rate': 0.1, 'label': 'Default'},
    {'batch_size': 64, 'num_epochs': 10, 'learning_rate': 0.1, 'label': 'Smaller batch'},
    {'batch_size': 256, 'num_epochs': 10, 'learning_rate': 0.1, 'label': 'Larger batch'},
    {'batch_size': 256, 'num_epochs': 10, 'learning_rate': 0.01, 'label': 'Lower learning rate'},
    {'batch_size': 256, 'num_epochs': 10, 'learning_rate': 0.5, 'label': 'Higher learning rate'}
]

# Run experiments and plot results
plt.figure(figsize=(12, 8))

for exp in experiments:
    print(f"\nRunning experiment: {exp['label']}")
    train_losses, test_losses = train_model(
        exp['batch_size'], 
        exp['num_epochs'], 
        exp['learning_rate']
    )
    
    plt.plot(train_losses, label=f"{exp['label']} - Train")
    plt.plot(test_losses, label=f"{exp['label']} - Test", linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Different Hyperparameters')
plt.legend()
plt.grid(True)
plt.savefig('hyperparameter_experiments.png')
plt.show() 