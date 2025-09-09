import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import re

class DataProcessor:
    """
    数据预处理器，用于处理贷款数据的解析、整理和分析
    """
    
    def __init__(self, data_file: str):
        """
        初始化数据处理器
        
        Args:
            data_file: 原始数据文件路径
        """
        self.data_file = data_file
        self.output_dir = "processed_data"
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def parse_json_message(self, json_str: str) -> Dict[str, Any]:
        """
        解析JSON格式的报文
        
        Args:
            json_str: JSON字符串
            
        Returns:
            解析后的字典
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return {}
    
    def extract_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """
        从嵌套字典中提取值，支持点号分隔的路径
        
        Args:
            data: 数据字典
            field_path: 字段路径，如 "bankCardInfo.bankCode"
            
        Returns:
            提取到的值，如果不存在则返回None
        """
        if not field_path:
            return None
            
        try:
            keys = field_path.split('.')
            value = data
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                elif isinstance(value, list) and key.isdigit():
                    # 处理列表索引
                    index = int(key)
                    if 0 <= index < len(value):
                        value = value[index]
                    else:
                        return None
                else:
                    return None
            
            return value
        except (KeyError, IndexError, ValueError, TypeError):
            return None
    
    def extract_features(self, json_data: Dict[str, Any], feature_list: List[str]) -> Dict[str, Any]:
        """
        从JSON数据中提取指定的特征
        
        Args:
            json_data: 解析后的JSON数据
            feature_list: 要提取的特征列表
            
        Returns:
            提取后的特征字典
        """
        extracted = {}
        
        for feature in feature_list:
            value = self.extract_nested_value(json_data, feature)
            
            # 处理特殊情况：如果是列表，取第一个元素或转换为字符串
            if isinstance(value, list):
                if len(value) > 0:
                    if feature.startswith('linkmanList'):
                        # 对于联系人列表，我们可能需要特殊处理
                        # 这里简单地取第一个联系人的信息
                        if '.' in feature:
                            sub_field = feature.split('.', 1)[1]
                            value = self.extract_nested_value(value[0], sub_field)
                        else:
                            value = str(value)
                    elif feature.startswith('pictureInfo'):
                        # 对于图片信息，也取第一个
                        if '.' in feature:
                            sub_field = feature.split('.', 1)[1]
                            value = self.extract_nested_value(value[0], sub_field)
                        else:
                            value = str(value)
                    else:
                        value = str(value)
                else:
                    value = None
            
            extracted[feature] = value
        
        return extracted
    
    def process_data_by_partner(self, feature_list: List[str]) -> Dict[str, int]:
        """
        按合作方处理数据并生成CSV文件
        
        Args:
            feature_list: 要提取的特征列表
            
        Returns:
            各合作方处理的数据条数统计
        """
        print("开始读取数据文件...")
        df = pd.read_csv(self.data_file)
        
        # 按合作方分组
        partner_data = defaultdict(list)
        error_count = 0
        
        print("开始处理数据...")
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"处理进度: {idx}/{total_rows}")
            
            partner_code = row['合作方编号']
            message = row['报文']
            result = row['结果']
            
            # 解析JSON报文
            json_data = self.parse_json_message(message)
            if not json_data:
                error_count += 1
                continue
            
            # 提取特征
            extracted_features = self.extract_features(json_data, feature_list)
            
            # 添加结果标签（成功=1，失败=0）
            extracted_features['label'] = 1 if result == '成功' else 0
            
            # 添加到对应合作方的数据中
            partner_data[partner_code].append(extracted_features)
        
        # 生成CSV文件
        print("生成CSV文件...")
        partner_counts = {}
        
        for partner_code, data_list in partner_data.items():
            if not data_list:
                continue
                
            # 创建DataFrame
            partner_df = pd.DataFrame(data_list)
            
            # 保存为CSV
            output_file = os.path.join(self.output_dir, f"{partner_code}.csv")
            partner_df.to_csv(output_file, index=False, encoding='utf-8')
            
            partner_counts[partner_code] = len(data_list)
            print(f"已生成 {output_file}，包含 {len(data_list)} 条记录")
        
        if error_count > 0:
            print(f"警告: 有 {error_count} 条记录因JSON解析错误被跳过")
        
        return partner_counts
    
    def analyze_features(self, feature_list: List[str], sample_size: int = 10000) -> Dict[str, Any]:
        """
        分析特征的分布情况
        
        Args:
            feature_list: 要分析的特征列表
            sample_size: 用于分析的样本大小
            
        Returns:
            特征分析结果
        """
        print("开始特征分析...")
        df = pd.read_csv(self.data_file)
        
        # 如果数据量太大，进行采样
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"使用 {sample_size} 个样本进行分析")
        
        analysis_results = {}
        
        # 分析每个特征
        for feature in feature_list:
            print(f"分析特征: {feature}")
            
            feature_values = []
            for _, row in df.iterrows():
                message = row['报文']
                json_data = self.parse_json_message(message)
                if json_data:
                    value = self.extract_nested_value(json_data, feature)
                    if value is not None:
                        feature_values.append(str(value))
            
            if not feature_values:
                analysis_results[feature] = {
                    'type': 'empty',
                    'count': 0,
                    'unique_count': 0,
                    'values': {}
                }
                continue
            
            # 统计分析
            value_counts = Counter(feature_values)
            total_count = len(feature_values)
            unique_count = len(value_counts)
            
            # 判断数据类型
            numeric_values = []
            for v in feature_values:
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    break
            
            is_numeric = len(numeric_values) == len(feature_values)
            
            if is_numeric and unique_count > 10:
                # 数值型特征
                analysis_results[feature] = {
                    'type': 'numeric',
                    'count': total_count,
                    'unique_count': unique_count,
                    'min': min(numeric_values),
                    'max': max(numeric_values),
                    'mean': sum(numeric_values) / len(numeric_values),
                    'top_values': dict(value_counts.most_common(10))
                }
            else:
                # 分类型特征
                analysis_results[feature] = {
                    'type': 'categorical',
                    'count': total_count,
                    'unique_count': unique_count,
                    'values': dict(value_counts.most_common(20))  # 只显示前20个最常见的值
                }
        
        # 分析结果标签
        print("分析结果标签...")
        result_counts = Counter()
        for _, row in df.iterrows():
            result = row['结果']
            label = 1 if result == '成功' else 0
            result_counts[label] += 1
        
        analysis_results['label'] = {
            'type': 'categorical',
            'count': len(df),
            'unique_count': len(result_counts),
            'values': dict(result_counts)
        }
        
        return analysis_results
    
    def print_analysis_report(self, analysis_results: Dict[str, Any]):
        """
        打印分析报告
        
        Args:
            analysis_results: 分析结果
        """
        print("\n" + "="*60)
        print("特征分析报告")
        print("="*60)
        
        for feature, stats in analysis_results.items():
            print(f"\n特征: {feature}")
            print("-" * 40)
            print(f"数据类型: {stats['type']}")
            print(f"总数: {stats['count']}")
            print(f"唯一值数量: {stats['unique_count']}")
            
            if stats['type'] == 'numeric':
                print(f"最小值: {stats['min']}")
                print(f"最大值: {stats['max']}")
                print(f"平均值: {stats['mean']:.2f}")
                print("最常见的值:")
                for value, count in stats['top_values'].items():
                    print(f"  {value}: {count}")
            elif stats['type'] == 'categorical':
                print("值分布:")
                for value, count in stats['values'].items():
                    percentage = (count / stats['count']) * 100
                    print(f"  {value}: {count} ({percentage:.1f}%)")
    
    def generate_sample_features(self) -> List[str]:
        """
        生成一个示例特征列表
        
        Returns:
            示例特征列表
        """
        return [
            'amount',
            'area',
            'bankCardInfo.bankCardNo',
            'bankCardInfo.bankCode',
            'bankCardInfo.cardType',
            'city',
            'companyInfo.industry',
            'companyInfo.occupation',
            'degree',
            'deviceInfo.osType',
            'deviceInfo.phoneType',
            'income',
            'maritalStatus',
            'province',
            'purpose',
            'term',
            'idInfo.gender',
            'linkmanList.0.relationship',  # 第一个联系人的关系
            'customerSource'
        ]


def main():
    """主函数，演示数据处理流程"""
    # 初始化数据处理器
    processor = DataProcessor('data/20250903.csv')
    
    # 使用示例特征列表
    feature_list = [
        'amount',
        'bankCardInfo.bankCode',
        'bankCardInfo.cardType',
        'companyInfo.industry'
    ]
    
    print("使用的特征列表:")
    for i, feature in enumerate(feature_list, 1):
        print(f"{i}. {feature}")
    
    # 处理数据并按合作方生成CSV文件
    print("\n开始数据处理...")
    partner_counts = processor.process_data_by_partner(feature_list)
    
    # 打印处理结果统计
    print("\n处理结果统计:")
    print("-" * 40)
    for partner, count in sorted(partner_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{partner}: {count} 条记录")
    
    # 进行特征分析
    print("\n开始特征分析...")
    analysis_results = processor.analyze_features(feature_list)
    
    # 打印分析报告
    processor.print_analysis_report(analysis_results)


if __name__ == "__main__":
    main()
