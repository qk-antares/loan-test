# 贷款数据预处理系统使用指南

## 概述

本系统提供了一套完整的贷款数据预处理解决方案，包括：
1. 从JSON报文中提取特征
2. 按合作方整理数据并生成CSV文件
3. 数据分析和统计

## 文件结构

```
loan/
├── data/
│   └── 20250903.csv              # 原始数据文件
├── processed_data/               # 处理后的数据目录（自动生成）
│   ├── AWJ_CODE.csv
│   ├── LXJ_CODE.csv
│   ├── RONG_CODE.csv
│   └── ...
├── req_param.py                  # 贷款请求类定义
├── data_processor.py             # 核心数据处理器
├── process_loan_data.py          # 主处理脚本
├── analyze_data.py               # 数据分析工具
└── README.md                     # 本文件
```

## 使用方法

### 1. 基本使用

运行默认特征集合的数据处理：

```bash
python process_loan_data.py
```

### 2. 演示模式

运行简单的演示，使用4个基本特征：

```bash
python process_loan_data.py demo
```

### 3. 交互式模式

允许用户自定义选择特征：

```bash
python process_loan_data.py interactive
```

### 4. 数据分析

分析处理后的数据：

```bash
python analyze_data.py
```

## 自定义特征列表

在 `process_loan_data.py` 中，您可以修改 `feature_list` 来自定义要提取的特征：

```python
feature_list = [
    'amount',                    # 贷款金额
    'bankCardInfo.bankCode',     # 银行代码
    'bankCardInfo.cardType',     # 银行卡类型
    'companyInfo.industry'       # 所属行业
]
```

### 支持的特征路径格式

- 简单字段：`amount`, `degree`, `income`
- 嵌套对象：`bankCardInfo.bankCode`, `idInfo.gender`
- 列表元素：`linkmanList.0.relationship` (第一个联系人的关系)

### 常用特征列表

#### 基本信息
- `amount` - 贷款金额
- `term` - 贷款期数
- `degree` - 学历
- `maritalStatus` - 婚姻状况
- `income` - 月收入
- `purpose` - 借款用途
- `customerSource` - 客户来源

#### 地理信息
- `province` - 省份
- `city` - 城市
- `area` - 区域

#### 身份信息
- `idInfo.gender` - 性别
- `idInfo.nation` - 民族

#### 银行卡信息
- `bankCardInfo.bankCode` - 银行代码
- `bankCardInfo.cardType` - 银行卡类型

#### 公司信息
- `companyInfo.industry` - 所属行业
- `companyInfo.occupation` - 职业

#### 设备信息
- `deviceInfo.osType` - 操作系统类型
- `deviceInfo.phoneType` - 手机型号
- `deviceInfo.phoneMaker` - 手机厂商

#### 联系人信息
- `linkmanList.0.relationship` - 第一个联系人关系
- `linkmanList.1.relationship` - 第二个联系人关系

## 数据分析功能

### 1. 合作方概览
- 各合作方的数据量统计
- 成功率和失败率分析

### 2. 特征分析
- 每个特征的分布情况
- 缺失值统计
- 唯一值数量

### 3. 成功模式分析
- 比较成功和失败案例的特征差异
- 识别可能的成功因素

### 4. 合作方比较
- 同一特征在不同合作方的分布对比

## 编码表参考

### 学历编码
- `DOCTOR` - 博士
- `MASTER` - 硕士
- `BACHELOR` - 本科
- `COLLEGE` - 大专
- `SENIOR` - 高中及中专
- `JUNIOR` - 初中及以下

### 婚姻状态
- `1` - 已婚
- `2` - 未婚
- `3` - 离异

### 收入编码
- `A` - 0-5000
- `B` - 5001-10000
- `C` - 10001-15000
- `D` - 15000以上

### 借款用途
- `CONSUME` - 购物消费
- `PAYRENT` - 支付房租
- `TRAINING` - 培训学习
- `TRAVEL` - 旅游度假
- `CREDIT` - 代还信用卡

### 行业编码
- `A` - 农、林、牧、渔业
- `C` - 制造业
- `E` - 建筑业
- `G` - 信息传输、计算机服务和软件业
- `H` - 批发和零售业
- `Z` - 未知

## 输出格式

生成的CSV文件格式：
```csv
feature1,feature2,feature3,...,label
value1,value2,value3,...,成功/失败
```

每个合作方生成一个独立的CSV文件，文件名为`{合作方编号}.csv`。

## 性能优化

- 数据处理支持大文件（>50MB）
- 使用进度显示跟踪处理状态
- 特征分析支持采样以提高性能
- 错误处理确保数据完整性

## 扩展功能

### 自定义DataProcessor

```python
from data_processor import DataProcessor

# 创建自定义处理器
processor = DataProcessor('data/20250903.csv')

# 自定义特征列表
my_features = ['amount', 'income', 'degree']

# 处理数据
partner_counts = processor.process_data_by_partner(my_features)

# 分析特征
analysis = processor.analyze_features(my_features)
```

### 批量处理

系统支持处理多个数据文件，只需修改DataProcessor的初始化参数即可。

## 故障排除

### 常见问题

1. **JSON解析错误**
   - 检查原始数据文件格式
   - 确认报文列包含有效的JSON字符串

2. **特征提取失败**
   - 检查特征路径是否正确
   - 确认嵌套结构存在

3. **内存不足**
   - 减少采样大小
   - 分批处理数据

4. **文件权限错误**
   - 确保有读写权限
   - 检查目录是否存在

### 调试模式

在代码中添加调试信息：

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 更新日志

- v1.0: 初始版本，支持基本的数据提取和分析
- 支持自定义特征列表
- 支持多种分析模式
- 完整的错误处理和进度跟踪
