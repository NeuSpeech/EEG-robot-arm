import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc=0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
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

        # save model
        if val_accuracy >best_acc:
            best_acc=val_accuracy
            model_path=f'model/{exp_name}.pth'
            os.makedirs(os.path.dirname(model_path),exist_ok=True)
            torch.save(model,f=model_path)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_history[-1]:.4f}, Val Loss: {val_loss_history[-1]:.4f}, Val Accuracy: {val_accuracy:.4f}')


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
    # plt.show()

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
    save_path = f'fig/{exp_name}/confusion_matrix.png'
    plot_confusion_matrix(all_targets, all_predicted, class_names, save_path)

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    return accuracy, f1


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
        self.fc = nn.Linear(32 * 32 * duration, len(class_names))  # 640 / 4 = 160

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


def load_data(data_path):
    if type(data_path) is str:
        data_path = [data_path]
    num = len(data_path)
    all_data = []
    all_labels = []
    for i in range(num):
        datas, labels = load_one_data(data_path[i])
        all_data.append(datas)
        all_labels.append(labels)
    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)
    return all_data, all_labels


def load_one_data(data_path):
    df = pd.read_csv(data_path, header=1)  # 假设第二行是列标题
    data_np = df.to_numpy()
    print(f'data_np:{data_np.shape}')
    data_np = data_np[:total_sec * 128, 4:18]
    # 初始化多个标签的数据列表
    datas = [[] for _ in range(len(sample_intention))]

    # 分割数据
    count = 0
    global segment_marks
    if segment_marks is None:
        segment_marks = [samples_per_event] * len(segment_marks)
    # print(f'segment_marks:{segment_marks}')
    segment_indexes = [0, *segment_marks]
    # print(f'segment_indexes:{segment_indexes}')
    segment_indexes = np.cumsum(segment_indexes) * sr
    loop_time = segment_indexes[-1]
    # print(f'segment_indexes x sr:{segment_indexes}')
    for i in range(0, data_np.shape[0], loop_time):
        # i is the index of start sample
        for j in range(len(sample_intention)):
            start = i + segment_indexes[sample_intention[j]]
            end = i + segment_indexes[sample_intention[j] + 1]
            # print(f'i:{i} j:{j} start:{start} end:{end} sec:{end-start} '
            #       f'si:{sample_intention[j]} ')
            new_data=data_np[start:end]
            datas[j].append(new_data)
        count += 1
    # print([e.shape for d in datas for e in d])
    datas = [np.stack(datas[i]) for i in range(len(datas))]
    labels = [np.zeros(datas[i].shape[0]) + i for i in range(len(datas))]
    # 合并眨眼和咬牙的数据和标签
    datas = np.concatenate(datas)
    labels = np.concatenate(labels).astype(np.int32)
    # print(f'datas:{datas.shape}')
    # print(datas[:5],labels[:5])
    # print(datas[-5:],labels[-5:])
    # exit()
    return datas, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--config_path', type=str,
                        default='',
                        help='Path to the CSV data file')
    parser.add_argument('--seed', type=int,
                        default=None,
                        help='Path to the CSV data file')
    args = parser.parse_args()
    yaml_file_path = args.config_path

    # 读取 YAML 文件
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # 现在你可以通过键来访问配置
    data_path = config['data_path']
    exp_name = config.get('exp_name',os.path.basename(args.config_path)[:-5])
    batch_size = config['batch_size']
    if args.seed is None:
        seed = config['seed']
    else:
        seed = args.seed
    exp_name=exp_name+'/seed'+str(seed)
    num_epochs = config['num_epochs']
    train_split = config['train_split']
    val_split = config['val_split']
    total_sec = config['total_sec']
    sample_intention = config['sample_intention']
    class_names = config['class_names']
    sr = config['sr']
    duration = config['duration']
    # segment_marks = config['segment_marks']
    segment_marks = config.get('segment_marks',None)
    print(f'initial segment_marks:{segment_marks}')
    # print(type(sample_intention),sample_intention)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 假设 data_np 是已经加载好的 NumPy 数组
    # 假设 data_np 的形状是 [n_samples, 14]（14个通道）

    # 每个事件（眨眼或咬牙）的样本数量
    samples_per_event = duration

    data, labels = load_data(data_path=data_path)

    # 将数据转换为 PyTorch 张量
    shuffle_indices = np.arange(data.shape[0])
    np.random.shuffle(shuffle_indices)
    data = data[shuffle_indices]
    labels = labels[shuffle_indices]
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    data_tensor = torch.permute(data_tensor, (0, 2, 1))

    # print(data_tensor.shape,labels_tensor.shape)
    # exit()
    # print(data_tensor[:5],labels_tensor[:5])
    # print(data_tensor[-5:],labels_tensor[-5:])
    # 创建 TensorDataset
    dataset = TensorDataset(data_tensor, labels_tensor)

    # 分割数据集为训练集、验证集和测试集
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    print(train_size, val_size, test_size)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义一个简单的神经网络模型

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
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

    os.makedirs(f'fig/{exp_name}', exist_ok=True)
    # 在测试集上评估模型
    # load best model after training
    model = torch.load(f'model/{exp_name}.pth')
    test_accuracy, test_f1 = test_model(model, test_loader)
    # 绘制训练和验证损失
    plt.plot(train_loss_history, label='Train loss')
    plt.plot(val_loss_history, label='Validation loss')
    plt.legend()
    plt.savefig(f'fig/{exp_name}/loss.png')
    # plt.show()
    plt.close()

    plt.plot(train_accuracy_history, label='Train accuracy')
    plt.plot(val_accuracy_history, label='Validation accuracy')
    plt.legend()
    plt.savefig(f'fig/{exp_name}/acc.png')
    # plt.show()
    plt.close()

    save_json_path = f'fig/{exp_name}/results.json'
    results = {
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'train_accuracy_history': train_accuracy_history,
        'val_accuracy_history': val_accuracy_history,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
    }

    with open(save_json_path, 'w') as f:
        f.write(json.dumps(results, indent=4))
        f.close()
