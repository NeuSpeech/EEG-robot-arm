如何使用这份代码？
你只需要复制一份yaml文件并编辑，然后在命令行中运行一下即可训练并评估
```shell
python train.py --config_path="your_config_path"
```

* data_path 可以使用一个或多个数据
* seed 可以用于控制效果复现
* train_split,val_split是设定数据集比例的，剩余的是test split的
* total_sec是数据集总共的时间长度，以此截断数据
* sample_intention 是每个数据循环中，将要使用的数据的次序，比如循环中有4个事件，1就是第二个事件
* class_names 是sample_intention里面使用的数据的分类
* segment_marks 是每个数据循环中所有事件各自的时长
* train_multiple_times与train.py的区别是会取0到10的10个seed进行训练，不会画图。