我们提供中英双语指导

We provide Chinese/English guidance

[中文](readme.md)|[English](readme_en.md)

# 先点个star，O(∩_∩)O哈哈~。

这是我们用脑电控制机械臂喝水的演示视频。
![image](demo/demo-eeg-robot-arm-drink-water.gif)

代码有什么特征？
我们将几乎所有机器学习的模型都加入了，并且加入了很多经典的深度学习模型。
模型和注册名称列表如下：

- SVM: svm
- Random Forest: random_forest
- KNN: knn
- Decision Tree: decision_tree
- XGBoost: xgboost
- LightGBM: lightgbm
- CatBoost: catboost
- AdaBoost: adaboost
- Gaussian Naive Bayes: gaussian_nb
- Multinomial Naive Bayes: multinomial_nb
- Bernoulli Naive Bayes: bernoulli_nb
- Logistic Regression: logistic
- SGD Classifier: sgd
- Linear SVC: linear_svc
- LDA: lda
- QDA: qda
- Nearest Centroid: nearest_centroid
- Lasso: lasso
- Ridge Classifier: ridge
- Elastic Net: elastic_net
- EEGNet: eegnet
- Simple CNN: simple_cnn
- Simple RNN: simple_rnn
- Simple LSTM: simple_lstm
- ResNet-like: resnet_like
- Transformer-like: transformer_like
- DenseNet-like: densenet_like
- LSTM with Attention: lstm_attention
- Inception-like: inception_like
- UNet-like: unet_like
- MobileNet-like: mobilenet_like
- SqueezeNet-like: squeezenet_like

如何使用这份代码？
你只需要复制一份yaml文件并编辑，然后在命令行中运行一下即可训练并评估
```shell
python train.py --config_path="your_config_path"
```

但是我们更推荐使用train_multiple_times.py 来避免小数据集下因为随机划分数据导致的过大的表现差异。

```shell
python train_multiple_times.py --config_path="your_config_path"
```

如果想训练多个模型，可以运行train_multiple_times.bat，这样会将所有模型训练10次，保存结果，并且生成一个excel表格。

* data_path 可以使用一个或多个数据
* seed 可以用于控制效果复现
* train_split,val_split是设定数据集比例的，剩余的是test split的
* total_sec是数据集总共的时间长度，以此截断数据
* sample_intention 是每个数据循环中，将要使用的数据的次序，比如循环中有4个事件，1就是第二个事件
* class_names 是sample_intention里面使用的数据的分类
* segment_marks 是每个数据循环中所有事件各自的时长

# 贡献者

Shenghao GAO: data collection and robot arm demo.

Yiqian YANG: data modeling and analysis.

