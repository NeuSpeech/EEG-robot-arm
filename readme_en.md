我们提供中英双语指导

We provide Chinese/English guidance

[中文](readme.md)|[English](readme_en.md)

# Please star us O(∩_∩)O

This is our demo.
![image](demo/demo-eeg-robot-arm-drink-water.gif)

This project is using EMOTIV as EEG device to do some classification tasks.

We have done [blink/grit classification](fig/blink_girt_3s_still_10min_3classifications_0903), [tongue direction classification](tongue/img.png),
and we use the classification result to control robot arm.

Don't worry if you don't have data, which are all provided in this repo as well.


How to use this code?

You only need to modify yaml configuration and run. 

```shell
python train.py --config_path="your_config_path"
```

We recommend to use train_multiple_times.py to reduce the effect of random data splitting on small dataset.
This will train 10 times using different seeds and average the results to get a fair evaluation.
```shell
python train_multiple_times.py --config_path="your_config_path"
```

1. **data_path** can use one or more datasets.
2. **seed** can be used to control effect reproduction.
3. **train_split**, **val_split** are for setting the dataset ratio, the remainder is for the test split.
4. **total_sec** is the total time length of the dataset, used to truncate the data.
5. **sample_intention** is the order of the data to be used in each data cycle, for example, if there are 4 events in the cycle, 1 represents the second event.
6. **class_names** are the categories of the data used in sample_intention.
7. **segment_marks** are the durations of all events in each data cycle.

# Contributors

Shenghao GAO: data collection and robot arm demo

Yiqian YANG: data modeling and analysis


Scan my WeChat, more interesting projects will be released, stay tuned!

![image](contact/915ceda980134e4ff679c8c6bea5fe1.jpg)