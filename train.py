#### 版权声明：本代码仅供学习交流，不得用于商业用途。
#### 版权所有：@2024 Yiqian Yang, github: https://github.com/NeuSpeech


import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, ElasticNet
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid

# 添加模型注册表
_MODEL_REGISTRY = {}
ML_MODELS = ['svm', 'random_forest', 'knn', 'decision_tree', 'xgboost', 'lightgbm', 'catboost', 'adaboost',
             'gaussian_nb', 'multinomial_nb', 'bernoulli_nb', 'logistic', 'sgd', 'linear_svc',
             'lda', 'qda', 'nearest_centroid', 'lasso', 'ridge', 'elastic_net']  # 所有传统机器学习模型的名称列表


def register_model(name):
    """模型注册装饰器"""

    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name):
    """通过名称获取模型类"""
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"未知的模型名称: {name}")
    return _MODEL_REGISTRY[name]


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == target.to(device)).sum().item()
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
                output = model(data.to(device))
                loss = criterion(output, target.to(device))
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                val_correct += (predicted == target.to(device)).sum().item()
        val_loss_history.append(val_loss / len(val_loader.dataset))
        val_accuracy = val_correct / total_val
        val_accuracy_history.append(val_accuracy)

        # save model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            model_path = f'model/{exp_name}.pth'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model, f=model_path)

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

    if model_name in ML_MODELS:  # 使用模型名称列表来判断
        # 传统机器学习模型的测试过程
        for data, target in test_loader:
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            data = data.reshape(data.shape[0], -1)
            predicted = model(data)
            print("predicted shape", predicted.shape)
            print("predicted", predicted)
            print("target shape", target.shape)
            print("target", target)
            # 如果预测的形状不一样，而且只是多了个维度，且该维度是1，就变成一样的。
            if predicted.shape != target.shape:
                if predicted.shape[-1] == 1:
                    predicted = predicted.squeeze(-1)
            assert predicted.shape == target.shape, 'must be same shape, or the accuracy will be wrong'
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    else:
        # 深度学习模型的测试过程
        with torch.no_grad():
            model.to(device)
            for data, target in test_loader:
                output = model(data.to(device))
                loss = criterion(output, target.to(device))
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target.to(device)).sum().item()
                all_predicted.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        test_loss /= len(test_loader.dataset)

    accuracy = correct / total
    f1 = f1_score(all_targets, all_predicted, average='macro')

    # 绘制混淆矩阵
    save_path = f'fig/{exp_name}/confusion_matrix.png'
    plot_confusion_matrix(all_targets, all_predicted, class_names, save_path)

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    return accuracy, f1




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
    classes_num = len(sample_intention)
    datas = [[] for _ in range(classes_num)]

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
            new_data = data_np[start:end]
            datas[j].append(new_data)
        count += 1
    # print([e.shape for d in datas for e in d])
    datas = [np.stack(datas[i]) for i in range(len(datas))]
    labels = [np.zeros(datas[i].shape[0]) + i for i in range(len(datas))]
    # 合并眨眼和咬牙的数据和标签
    datas = np.concatenate(datas)
    labels = np.concatenate(labels).astype(np.int32)
    # datas shape: [classes, data_num, sample_length, channel_num]
    # labels shape:[classes, data_num]
    # 把数据按照时间的顺序组装
    datas = rearrange(datas, '(a b) c d-> (b a) c d', a=classes_num)
    labels = rearrange(labels, '(a b) -> (b a)', a=classes_num)
    # 将每个样本分割成多个piece
    datas = rearrange(datas, 'n (t p) d -> (n t) p d', p=piece_length)
    # 对应地重复标签
    labels = repeat(labels, 'n -> (n t)', t=(duration*sr//piece_length))
    
    # print(f'datas:{datas.shape}')
    # print(datas[:5],labels[:5])
    # print(datas[-5:],labels[-5:])
    # exit()
    return datas, labels


@register_model('svm')
class SVMWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        # 转换输入数据格式 [batch, channels, time] -> [batch]
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        # 返回类别预测
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            # 收集所有训练数据
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            # 训练模型
            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('random_forest')
class RandomForestWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        # 转换输入数据格式 [batch, channels, time] -> [batch, features]
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        # 返回类别预测
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            # 收集所有训练数据
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            # 训练模型
            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('knn')
class KNNWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('decision_tree')
class DecisionTreeWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier(random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('xgboost')
class XGBoostWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('lightgbm')
class LightGBMWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            random_state=42,
            verbose=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('catboost')
class CatBoostWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CatBoostClassifier(
            verbose=False,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('adaboost')
class AdaBoostWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('gaussian_nb')
class GaussianNBWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GaussianNB()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('multinomial_nb')
class MultinomialNBWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MultinomialNB()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        # MultinomialNB需要非负数据
        x = self.scaler.transform(x)
        x = x - x.min()  # 确保数据非负
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            X = X - X.min()  # 确保数据非负
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('bernoulli_nb')
class BernoulliNBWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BernoulliNB()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        # 将数据二值化（大于0的设为1，小于等于0的设为0）
        x = (x > 0).astype(np.float64)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            # 将数据二值化
            X = (X > 0).astype(np.float64)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('logistic')
class LogisticRegressionWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('sgd')
class SGDClassifierWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SGDClassifier(
            loss='log_loss',  # 使用logistic loss，相当于逻辑回归的随机梯度下降版本
            max_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('linear_svc')
class LinearSVCWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LinearSVC(
            random_state=42,
            max_iter=2000
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('lda')
class LDAWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LinearDiscriminantAnalysis(
            solver='svd'  # 使用SVD求解器，更稳定
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('qda')
class QDAWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = QuadraticDiscriminantAnalysis(
            store_covariance=True  # 存储协方差矩阵，便于后续分析
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('nearest_centroid')
class NearestCentroidWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = NearestCentroid(
            metric='euclidean',  # 使用欧氏距离
            shrink_threshold=None  # 不使用收缩阈���
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('lasso')
class LassoWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(
            penalty='l1',  # LASSO 正则化
            solver='liblinear',  # 'liblinear' 支持 L1 正则化
            multi_class='ovr',  # one-vs-rest
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('ridge')
class RidgeClassifierWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RidgeClassifier(
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('elastic_net')
class ElasticNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ElasticNet(
            alpha=1.0,  # 正则化强度
            l1_ratio=0.5,  # L1和L2正则化的混合比例
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.reshape(x.shape[0], -1)

        if not self.is_fitted:
            return torch.zeros((x.shape[0]))

        x = self.scaler.transform(x)
        predicted = self.model.predict(x)
        return torch.tensor(predicted, dtype=torch.long)

    def train(self, mode=True):
        if mode:
            all_data = []
            all_labels = []
            for data, target in train_loader:
                data = data.numpy().reshape(data.shape[0], -1)
                all_data.append(data)
                all_labels.append(target.numpy())

            X = np.concatenate(all_data)
            y = np.concatenate(all_labels)

            self.scaler.fit(X)
            X = self.scaler.transform(X)
            self.model.fit(X, y)
            self.is_fitted = True
        return self


@register_model('simple_cnn')
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=14, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 更新 fc1_input_size 为 8192
        self.fc1_input_size = 64 * sr*piece_duration//4  # 64 channels * 128 time steps
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, len(class_names))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@register_model('simple_rnn')
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=14, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, len(class_names))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, time, channels]
        _, h_n = self.rnn(x)
        x = self.fc(h_n[-1])
        return x


@register_model('simple_lstm')
class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=14, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, len(class_names))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, time, channels]
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


@register_model('resnet_like')
class ResNetLike(nn.Module):
    def __init__(self):
        super(ResNetLike, self).__init__()
        self.layer1 = BasicBlock(14, 64)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 256, stride=2)
        self.layer4 = BasicBlock(256, 256, stride=2)
        self.layer5 = BasicBlock(256, 256, stride=2)
        self.layer6 = BasicBlock(256, 256, stride=2)
        self.layer7 = BasicBlock(256, 256, stride=2)
        self.fc = nn.Linear(256 * (piece_duration*sr // 64), len(class_names))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@register_model('transformer_like')
class TransformerLike(nn.Module):
    def __init__(self):
        super(TransformerLike, self).__init__()
        self.embedding = nn.Linear(14, 64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4),
            num_layers=2
        )
        self.fc = nn.Linear(64, len(class_names))

    def forward(self, x):
        x = x.permute(2, 0, 1)  # [time, batch, channels]
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=0)  # 平均池化
        x = self.fc(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


@register_model('densenet_like')
class DenseNetLike(nn.Module):
    def __init__(self):
        super(DenseNetLike, self).__init__()
        self.conv1 = nn.Conv1d(14, 16, kernel_size=3, padding=1)  # 减少输出通道数
        self.dense_block1 = DenseBlock(16, 8, 4)  # 减少增长率
        self.transition1 = nn.Sequential(
            nn.Conv1d(48, 24, kernel_size=1),  # 减少输出通道数
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        self.dense_block2 = DenseBlock(24, 8, 4)  # 减少增长率
        self.transition2 = nn.Sequential(
            nn.Conv1d(56, 28, kernel_size=1),  # 添加额外的过渡层
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        # 更新 fc 的输入大小为 piece_duration*7*128
        self.fc = nn.Linear(piece_duration*7*128, len(class_names))  # 调整输入大小

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense_block1(x)
        x = self.transition1(x)
        x = self.dense_block2(x)
        x = self.transition2(x)  # 添加额外的过渡层
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@register_model('lstm_attention')
class LSTMAttention(nn.Module):
    def __init__(self):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_size=14, hidden_size=64, num_layers=1, batch_first=True)
        self.attention = nn.Linear(64, 1)
        self.fc = nn.Linear(64, len(class_names))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, time, channels]
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        x = self.fc(context)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv1d(in_channels, 16, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=1),
            nn.Conv1d(16, 24, kernel_size=3, padding=1)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=1),
            nn.Conv1d(16, 24, kernel_size=5, padding=2)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, 24, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


@register_model('inception_like')
class InceptionLike(nn.Module):
    def __init__(self):
        super(InceptionLike, self).__init__()
        self.conv1 = nn.Conv1d(14, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 添加池化层以减少特征图大小
        self.inception1 = InceptionModule(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 添加池化层以减少特征图大小
        self.inception2 = InceptionModule(88)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 添加池化层以减少特征图大小
        self.inception3 = InceptionModule(88)  # 新增的Inception模块
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # 新增的池化层
        self.inception4 = InceptionModule(88)  # 新增的Inception模块
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)  # 新增的池化层
        
        self.fc_input_size = 88 * piece_duration*sr // 32  # 根据池化后的特征图大小调整
        self.fc = nn.Linear(self.fc_input_size, len(class_names))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)  # 应用池化层
        x = self.inception1(x)
        x = self.pool2(x)  # 应用池化层
        x = self.inception2(x)
        x = self.pool3(x)  # 应用池化层
        x = self.inception3(x)  # 新增的Inception模块
        x = self.pool4(x)  # 新增的池化层
        x = self.inception4(x)  # 新增的Inception模块
        x = self.pool5(x)  # 新增的池化层
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@register_model('inception_like_v2') 
class InceptionLikeV2(nn.Module):
    def __init__(self):
        super(InceptionLikeV2, self).__init__()
        # 增加初始卷积层的通道数
        self.conv1 = nn.Conv1d(14, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 增加Inception模块的复杂度
        self.inception1 = InceptionModule(64)  # 输入通道增加到64
        self.bn2 = nn.BatchNorm1d(88)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.inception2 = InceptionModule(88)
        self.bn3 = nn.BatchNorm1d(88) 
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.inception3 = InceptionModule(88)
        self.bn4 = nn.BatchNorm1d(88)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.inception4 = InceptionModule(88)
        self.bn5 = nn.BatchNorm1d(88)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 添加SE注意力模块
        self.se = SELayer(88)
        
        self.fc_input_size = 88 * piece_duration*sr // 32
        # 添加dropout和更多全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, len(class_names))
        )

    def forward(self, x):
        # 添加残差连接
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        identity = x
        x = self.inception1(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = self.inception2(x)
        x = self.bn3(x)
        x = self.pool3(x)
        
        x = self.inception3(x)
        x = self.bn4(x)
        x = self.pool4(x)
        
        x = self.inception4(x)
        x = self.bn5(x)
        x = self.pool5(x)
        
        # 应用SE注意力
        x = self.se(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# SE注意力模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


@register_model('inception_like_v3')
class InceptionLikeV3(nn.Module):
    def __init__(self):
        super(InceptionLikeV3, self).__init__()
        # 增加初始卷积层的通道数
        self.conv1 = nn.Conv1d(14, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 增加Inception模块的复杂度
        self.inception1 = InceptionModule(128)
        self.bn2 = nn.BatchNorm1d(88)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.inception2 = InceptionModule(88)
        self.bn3 = nn.BatchNorm1d(88) 
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.inception3 = InceptionModule(88)
        self.bn4 = nn.BatchNorm1d(88)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.inception4 = InceptionModule(88)
        self.bn5 = nn.BatchNorm1d(88)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 添加SE注意力模块
        self.se = SELayer(88)
        
        self.fc_input_size = 88 * piece_duration * sr // 32
        # 添加dropout和更多全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, len(class_names))
        )

    def forward(self, x):
        # 添加残差连接
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.inception1(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = self.inception2(x)
        x = self.bn3(x)
        x = self.pool3(x)
        
        x = self.inception3(x)
        x = self.bn4(x)
        x = self.pool4(x)
        
        x = self.inception4(x)
        x = self.bn5(x)
        x = self.pool5(x)
        
        # 应用SE注意力
        x = self.se(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width)
        out = self.gamma * out + x
        return out

@register_model('inception_like_v4')
class InceptionLikeV4(nn.Module):
    def __init__(self):
        super(InceptionLikeV4, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Conv1d(14, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Inception模块
        self.inception1 = InceptionModule(64)
        self.bn2 = nn.BatchNorm1d(88)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.inception2 = InceptionModule(88)
        self.bn3 = nn.BatchNorm1d(88) 
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.inception3 = InceptionModule(88)
        self.bn4 = nn.BatchNorm1d(88)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.inception4 = InceptionModule(88)
        self.bn5 = nn.BatchNorm1d(88)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 自注意力模块
        self.self_attention = SelfAttention(88)
        
        self.fc_input_size = 88 * piece_duration * sr // 32
        # 添加dropout和更多全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, len(class_names))
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.inception1(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = self.inception2(x)
        x = self.bn3(x)
        x = self.pool3(x)
        
        x = self.inception3(x)
        x = self.bn4(x)
        x = self.pool4(x)
        
        x = self.inception4(x)
        x = self.bn5(x)
        x = self.pool5(x)
        
        # 应用自注意力
        x = self.self_attention(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class ImprovedInceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, pool_proj):
        super(ImprovedInceptionModule, self).__init__()
        
        # 1x1 卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out1x1, kernel_size=1),
            nn.BatchNorm1d(out1x1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 -> 3x3 卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, red3x3, kernel_size=1),
            nn.BatchNorm1d(red3x3),
            nn.ReLU(inplace=True),
            nn.Conv1d(red3x3, out3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(out3x3),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 -> 5x5 卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, red5x5, kernel_size=1),
            nn.BatchNorm1d(red5x5),
            nn.ReLU(inplace=True),
            nn.Conv1d(red5x5, out5x5, kernel_size=5, padding=2),
            nn.BatchNorm1d(out5x5),
            nn.ReLU(inplace=True)
        )
        
        # maxpool -> 1x1 卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm1d(pool_proj),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)

@register_model('inception_like_v5')
class InceptionLikeV5(nn.Module):
    def __init__(self):
        super(InceptionLikeV5, self).__init__()
        
        # 初始特征提取
        self.pre_layers = nn.Sequential(
            nn.Conv1d(14, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception模块
        self.inception3a = ImprovedInceptionModule(192, 64, 96, 128, 16, 32, 32)  # 256
        self.inception3b = ImprovedInceptionModule(256, 128, 128, 192, 32, 96, 64)  # 480
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = ImprovedInceptionModule(480, 192, 96, 208, 16, 48, 64)  # 512
        self.inception4b = ImprovedInceptionModule(512, 160, 112, 224, 24, 64, 64)  # 512
        self.inception4c = ImprovedInceptionModule(512, 128, 128, 256, 24, 64, 64)  # 512
        self.inception4d = ImprovedInceptionModule(512, 112, 144, 288, 32, 64, 64)  # 528
        self.inception4e = ImprovedInceptionModule(528, 256, 160, 320, 32, 128, 128)  # 832
        self.maxpool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = ImprovedInceptionModule(832, 256, 160, 320, 32, 128, 128)  # 832
        self.inception5b = ImprovedInceptionModule(832, 384, 192, 384, 48, 128, 128)  # 1024
        
        # SE注意力
        self.se = SELayer(1024, reduction=16)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, len(class_names))
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 检查是否有bias再初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 预处理层
        x = self.pre_layers(x)
        
        # Inception模块
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # SE注意力
        x = self.se(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


@register_model('unet_like')
class UNetLike(nn.Module):
    def __init__(self):
        super(UNetLike, self).__init__()
        # 编码器部分
        self.encoder1 = UNetBlock(14, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 解码器部分
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = UNetBlock(128, 64)  # 拼接后通道数为128
        self.up2 = nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2)
        self.decoder2 = UNetBlock(128, 8)  # 拼接后通道数为128

        # 计算池化后的特征图大小
        self.fc_input_size = 32 * (piece_duration * sr // 4)  # 根据池化后的特征图大小调整
        self.fc = nn.Linear(self.fc_input_size, len(class_names))

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        # 解码器
        dec1 = self.up1(enc2)
        # 确保拼接前调整大小
        if dec1.size(2) != enc1.size(2):
            diff = enc1.size(2) - dec1.size(2)
            dec1 = F.pad(dec1, (0, diff))  # 在最后一个维度上填充
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        dec2 = self.up2(dec1)
        # 确保拼接前调整大小
        if dec2.size(2) != x.size(2):
            diff = x.size(2) - dec2.size(2)
            dec2 = F.pad(dec2, (0, diff))  # 在最后一个维度上填充
        dec2 = torch.cat((dec2, enc1), dim=1)  # 拼接解码器输出与编码器的第一层输出
        dec2 = self.decoder2(dec2)

        # 展平特征图以输入到全连接层
        dec2 = dec2.view(dec2.size(0), -1)
        return self.fc(dec2)
    

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@register_model('mobilenet_like')
class MobileNetLike(nn.Module):
    def __init__(self):
        super(MobileNetLike, self).__init__()
        self.conv1 = nn.Conv1d(14, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256,stride=2),
            DepthwiseSeparableConv(256, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=2),
        )
        self.fc = nn.Linear(256 * piece_duration*sr // 64, len(class_names))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv1d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv1d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3x3 = nn.Conv1d(squeeze_channels, expand_channels, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.expand_activation(torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1))


@register_model('squeezenet_like')
class SqueezeNetLike(nn.Module):
    def __init__(self):
        super(SqueezeNetLike, self).__init__()
        self.conv1 = nn.Conv1d(14, 96, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(96)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.fire2 = FireModule(96, 16, 64)
        self.fire3 = FireModule(128, 16, 64)
        self.fire4 = FireModule(128, 32, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.fire5 = FireModule(256, 32, 128)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 第三个池化层
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 第四个池化层
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 第五个池化层
        self.fc = nn.Linear(256 * piece_duration*sr // 64, len(class_names))  # 修改输入维度（除以2048 = 4*8*8*8*8）

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.fire2(x)
        x = self.pool2(x)
        x = self.fire3(x)
        x = self.pool3(x)
        x = self.fire4(x)
        x = self.pool4(x)
        x = self.fire5(x)
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

@register_model('eegnet')
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
        self.fc = nn.Linear(32 * sr//4 * piece_duration, len(class_names))  # 640 / 4 = 160

    def forward(self, x):
        # x shape [B,C,T]
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
        # [B,class_num]
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加参数
    parser.add_argument('--config_path', type=str,
                        default='',
                        help='Path to the CSV data file')
    parser.add_argument('--seed', type=int,
                        default=None,
                        help='Path to the CSV data file')
    parser.add_argument('--model_name', type=str,
                        default=None,
                        help='choose model')
    args = parser.parse_args()
    print(f'args:{args}')

    yaml_file_path = args.config_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取 YAML 文件
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # 现在你可以通过键来访问配置
    data_path = config['data_path']
    data_shuffle = config.get('data_shuffle', True)  # 如果是False，就会按时间进行数据切分
    exp_name = config.get('exp_name', os.path.basename(args.config_path)[:-5])
    batch_size = config['batch_size']
    if args.seed is None:
        seed = config['seed']
    else:
        seed = args.seed
    exp_name = exp_name + '/seed' + str(seed)
    # 如果用户指定了模型，则使用用户指定的模型，否则使用配置文件中的模型
    if args.model_name is not None:
        model_name = args.model_name
    else:
        model_name = config.get('model_name', 'eegnet')
    print(f'model_name:{model_name}')
    # 在exp_name中添加模型名称
    exp_name = exp_name + '/' + model_name
    num_epochs = config['num_epochs']
    train_split = config['train_split']
    val_split = config['val_split']
    total_sec = config['total_sec']
    sample_intention = config['sample_intention']
    class_names = config['class_names']
    sr = config['sr']
    duration = config['duration']
    piece_duration = config.get('piece_duration', duration)
    piece_length = piece_duration * sr
    # segment_marks = config['segment_marks']
    segment_marks = config.get('segment_marks', None)
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
    # shuffle_indices = np.arange(data.shape[0])
    # np.random.shuffle(shuffle_indices)
    # data = data[shuffle_indices]
    # labels = labels[shuffle_indices]
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    data_tensor = torch.permute(data_tensor, (0, 2, 1))
    print(f'data_tensor.shape:{data_tensor.shape},labels_tensor.shape:{labels_tensor.shape}')
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
    if data_shuffle:
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    else:
        train_dataset = Subset(dataset, range(train_size))
        val_dataset = Subset(dataset, range(train_size, train_size + val_size))
        test_dataset = Subset(dataset, range(train_size + val_size, len(dataset)))
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 实例化模型
    model = get_model(model_name)()

    os.makedirs(f'fig/{exp_name}', exist_ok=True)
    if model_name in ML_MODELS:  # 使用模型名称列表来判断
        # 传统机器学习模型的处理流程
        model.train()  # 这会直接完成训练
        # print("fit_status_",model.model.fit_status_)
        # 直接进行测试评估
        test_accuracy, test_f1 = test_model(model, test_loader)

        # 保存结果
        save_json_path = f'fig/{exp_name}/results.json'
        results = {
            'test_accuracy': test_accuracy,
            'test_f1': test_f1
        }
        print(results)
        print(save_json_path)
        with open(save_json_path, 'w') as f:
            json.dump(results, f, indent=4)

    else:
        # 深度学习模型走原来的流程
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_loss_history = []
        val_loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []

        # 训练模型
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

        # 绘制训练曲线并保存结果

        # load best model after training
        model = torch.load(f'model/{exp_name}.pth', weights_only=False)
        test_accuracy, test_f1 = test_model(model, test_loader)

        # 绘制训练和验证损失
        plt.plot(train_loss_history, label='Train loss')
        plt.plot(val_loss_history, label='Validation loss')
        plt.legend()
        plt.savefig(f'fig/{exp_name}/loss.png')
        plt.close()

        plt.plot(train_accuracy_history, label='Train accuracy')
        plt.plot(val_accuracy_history, label='Validation accuracy')
        plt.legend()
        plt.savefig(f'fig/{exp_name}/acc.png')
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
            json.dump(results, f, indent=4)
