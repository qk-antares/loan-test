# Copilot Instructions for `loan-test`

## 项目结构与主要组件
- `data_processor.py`：数据预处理脚本，负责从原始 CSV 文件中选取特定列并进行清洗、转换。
- `ml_model_design.py`：基于随机森林的主导流模型实验脚本。
- `ml_model_design_onehot.py`：与主模型类似，但特征工程采用独热编码。
- `ml_model_design_mlp.py`：特征降维采用 MLP（多层感知机），后续仍用随机森林。
- `utils/`：包含如 `date_util.py` 等通用工具函数。
- `log/`：存放实验日志，便于追踪不同实验的结果。

## 数据流与开发流程
- 数据文件（如 `20250903.csv`）需放入 `data/` 目录。
- 运行 `data_processor.py` 进行数据预处理，输出中间数据。
- 选择不同的模型脚本（如 `ml_model_design.py`、`ml_model_design_onehot.py`、`ml_model_design_mlp.py`）进行实验。
- 日志自动写入 `log/`，文件名反映实验类型（如“独热编码.log”、“降维.log”）。

## 约定与模式
- 所有实验脚本均以独立文件形式存在，便于对比不同特征工程方法。
- 工具函数集中于 `utils/`，如日期处理等，避免重复代码。
- 日志文件命名与实验脚本保持一致，便于溯源。
- 数据处理、特征工程、建模流程分离，便于扩展和维护。

## 关键文件示例
- `data_processor.py`：
  - 读取 `data/` 下的 CSV，输出预处理结果。
- `ml_model_design_onehot.py`：
  - 读取预处理数据，进行独热编码，训练随机森林，输出日志。
- `utils/date_util.py`：
  - 提供日期相关的辅助函数。

## 其他说明
- 目前无自动化测试脚本，实验结果以日志为主。
- 若需扩展新特征工程或模型，建议仿照现有脚本新建文件。
- 依赖包未在 README 明确列出，建议根据脚本头部 import 补全环境。

---
如需进一步了解各脚本具体实现，请直接查阅对应 `.py` 文件。