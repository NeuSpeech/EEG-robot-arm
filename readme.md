我们提供中英双语指导

We provide Chinese/English guidance

[中文](readme.md)|[English](readme_en.md)

# 先点个star，O(∩_∩)O哈哈~。

这是我们用脑电控制机械臂喝水的演示视频。
![image](demo/demo-eeg-robot-arm-drink-water.gif)

我们这里展示的是眨眼和咬牙的分类，在眨眼，咬牙和休息的三分类任务上取得了90%的正确率，结果在[这里](fig/blink_girt_3s_still_10min_3classifications_0903)。

我们还做了非常前沿的探索，比如[舌头放置位置的分类](tongue/img.png)。


如何使用这份代码？
你只需要复制一份yaml文件并编辑，然后在命令行中运行一下即可训练并评估
```shell
python train.py --config_path="your_config_path"
```

但是我们更推荐使用train_multiple_times.py 来避免小数据集下因为随机划分数据导致的过大的表现差异。

```shell
python train_multiple_times.py --config_path="your_config_path"
```


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


加个微信吧，一起研究神经信号，之后还会有更精彩的项目分享出来，保持关注哦

进群改下马甲哦，学校专业姓名

![image](contact/915ceda980134e4ff679c8c6bea5fe1.jpg)