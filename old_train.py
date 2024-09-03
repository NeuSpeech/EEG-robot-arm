import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('data/blink and girt_EPOCX_233756_2024.08.19T17.29.05+08.00.md.pm.bp.csv', header=1)  # 假设第二行是列标题

# 将DataFrame转换为NumPy数组
data_np = df.to_numpy()
batch_size=4
seed=6
data_np = data_np[:300*128,4:18]
torch.manual_seed(seed)
np.random.seed(seed)
class_names = ['Blink', 'Grit']
# 假设 data_np 是已经加载好的 NumPy 数组
# 假设 data_np 的形状是 [n_samples, 14]（14个通道）

# 每个事件（眨眼或咬牙）的样本数量
samples_per_event = 640

# 初始化眨眼和咬牙的数据列表
blinks = []
grits = []

# 分割数据
for i in range(0, data_np.shape[0], 4 * samples_per_event):
    blinks.append(data_np[i + samples_per_event:i + 2 * samples_per_event])
    grits.append(data_np[i + 3 * samples_per_event:i + 4 * samples_per_event])

# 将列表转换为 NumPy 数组
blinks = np.stack(blinks)
grits = np.stack(grits)
print(blinks.shape, grits.shape)
# 标签：眨眼为1，咬牙为0
labels_blinks = np.ones(blinks.shape[0])
labels_grits = np.zeros(grits.shape[0])

# 合并眨眼和咬牙的数据和标签
data = np.concatenate((blinks, grits))
labels = np.concatenate((labels_blinks, labels_grits))

# 将数据转换为 PyTorch 张量
shuffle_indices = np.arange(data.shape[0])
np.random.shuffle(shuffle_indices)
data = data[shuffle_indices]
labels = labels[shuffle_indices]
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)
data_tensor=torch.permute(data_tensor,(0,2,1))

print(data_tensor.shape, labels_tensor.shape)
exit()

# 创建 TensorDataset
dataset = TensorDataset(data_tensor, labels_tensor)

# 分割数据集为训练集、验证集和测试集
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
print(train_size, val_size, test_size)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义一个简单的神经网络模型
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        # 第一层卷积，使用一维卷积
        self.conv1 = nn.Conv1d(in_channels=14, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=16)

        # 深度卷积层
        self.depth_conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, groups=16)
        self.batchnorm2 = nn.BatchNorm1d(num_features=32)

        # 分离卷积层
        self.separable_conv = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.batchnorm3 = nn.BatchNorm1d(num_features=32)

        # 平均池化层
        self.avgpool = nn.AvgPool1d(kernel_size=4)

        # 全连接层
        self.fc = nn.Linear(32 * 160, 2)  # 640 / 4 = 160

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)

        x = F.relu(self.depth_conv(x))
        x = self.batchnorm2(x)

        x = F.relu(self.separable_conv(x))
        x = self.batchnorm3(x)

        x = self.avgpool(x)

        # 展平特征图以输入到全连接层
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


# 实例化模型
model = EEGNet()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loss_history = []
val_loss_history = []
train_accuracy_history = []
val_accuracy_history = []

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct=0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == target).sum().item()
        train_accuracy_history.append(train_correct / len(train_loader.dataset))
        train_loss_history.append(running_loss / len(train_loader.dataset))

        # 计算验证集损失
        model.eval()

        # 计算验证集损失
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                val_correct += (predicted == target).sum().item()
        val_loss_history.append(val_loss / len(val_loader.dataset))
        val_accuracy = val_correct / total_val
        val_accuracy_history.append(val_accuracy)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_history[-1]:.4f}, Val Loss: {val_loss_history[-1]:.4f}, Val Accuracy: {val_accuracy:.4f}')



# 训练模型
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)

# 测试模型
from sklearn.metrics import confusion_matrix
import seaborn as sns


# 绘制混淆矩阵的函数
def plot_confusion_matrix(true_labels, predicted_labels, class_names, save_path):
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)

    # 使用Seaborn绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    # 保存图像
    plt.savefig(save_path)

    # 显示图像
    plt.show()

    # 关闭图像以节省内存
    plt.close()


# 在测试模型函数中使用plot_confusion_matrix
def test_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_predicted.extend(predicted.numpy())
            all_targets.extend(target.numpy())
    test_loss /= len(test_loader.dataset)
    accuracy = correct / total
    f1 = f1_score(all_targets, all_predicted, average='macro')

    # 绘制混淆矩阵
    save_path = 'fig/confusion_matrix.png'
    plot_confusion_matrix(all_targets, all_predicted, class_names, save_path)

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    return accuracy, f1


# 在测试集上评估模型
test_accuracy, test_f1 = test_model(model, test_loader)

os.makedirs('fig',exist_ok=True)
# 绘制训练和验证损失
plt.plot(train_loss_history, label='Train loss')
plt.plot(val_loss_history, label='Validation loss')
plt.legend()
plt.savefig('fig/loss.png')
plt.show()
plt.close()

plt.plot(train_accuracy_history, label='Train accuracy')
plt.plot(val_accuracy_history, label='Validation accuracy')
plt.legend()
plt.savefig('fig/acc.png')
plt.show()
plt.close()


