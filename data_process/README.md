`daily_partner_processor.py`的作用是按照天和合作方对数据进行组织和预处理，处理后的文件放在`processed`目录下，形如：

```bash
processed/
    2025-09-20/
        partnerA.csv
        partnerB.csv
    2025-09-21/
        partnerA.csv
        partnerC.csv
```

关键函数的解释：

- `__init__(self, output_dir: str = "processed")`：指定输出目录，以及可以指定抽取原始数据中的哪些特征。

- `process_single_record(self, line: str) -> Optional[Dict[str, Any]]`：解析每一行数据（字符串），返回一个数据字典（未清洗异常值）

- `process_all_data()`：处理`data`目录下的所有数据文件的示例使用方法

- `process_single_file(file_date: str)`：处理指定日期的单个数据文件
