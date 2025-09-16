```bash
├── data	# 把20250903.csv文件放到这里(csv格式好处理)
├── log # 记录了一些实验结果
├── utils   # 一些工具函数
├── data_processor.py   # 数据处理脚本，选取特定的列并进行预处理
├── ml_model_design.py   # 基于随机森林的导流模型实验
├── ml_model_design_onehot.py   # 基于随机森林的导流模型实验（使用独热编码）
└── ml_model_design_mlp.py   # 基于随机森林的导流模型实验（使用MLP对特征进行降维）
```