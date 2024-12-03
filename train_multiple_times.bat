@echo off
REM 声明配置文件路径
set config_path=configs/blink_girt_tongue_4s_still_10min_4classifications_1202.yaml

REM 声明所有注册过的模型名称
set model_names=svm random_forest knn decision_tree xgboost lightgbm catboost adaboost gaussian_nb multinomial_nb bernoulli_nb logistic sgd linear_svc lda qda nearest_centroid lasso ridge elastic_net simple_cnn simple_rnn simple_lstm resnet_like transformer_like densenet_like lstm_attention inception_like unet_like mobilenet_like eegnet squeezenet_like

REM 运行命令
python train_multiple_times.py --config_path="%config_path%" --run_training=True --model_names %model_names% 
