import pandas as pd
import json
import os
from typing import List, Dict, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta
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
    
    def extract_request_date_from_id(self, request_id: str) -> datetime:
        """
        从请求ID中提取请求日期
        
        Args:
            request_id: 请求ID，格式如 "20250901000009716202"
            
        Returns:
            请求日期的datetime对象
        """
        try:
            # 提取前8位作为日期部分
            date_str = request_id[:8]
            return datetime.strptime(date_str, '%Y%m%d')
        except (ValueError, IndexError):
            # 如果解析失败，返回一个默认日期
            return datetime(2025, 9, 3)
    
    def parse_birth_date(self, birth_date_str: str) -> datetime:
        """
        解析出生日期字符串
        
        Args:
            birth_date_str: 出生日期字符串，格式如 "19930125"
            
        Returns:
            出生日期的datetime对象
        """
        try:
            return datetime.strptime(birth_date_str, '%Y%m%d')
        except ValueError:
            # 如果解析失败，返回None
            return None
    
    def parse_validity_date(self, validity_date_str: str) -> tuple:
        """
        解析证件有效期字符串
        
        Args:
            validity_date_str: 有效期字符串，格式如 "2024.10.28-2044.10.28" 或 "2018.01.20-长期"
            
        Returns:
            (开始日期, 结束日期) 的tuple，如果是长期则结束日期为None
        """
        try:
            if '-长期' in validity_date_str:
                # 处理长期有效的情况
                start_date_str = validity_date_str.replace('-长期', '')
                start_date = datetime.strptime(start_date_str, '%Y.%m.%d')
                return start_date, None
            else:
                # 处理有明确结束日期的情况
                date_parts = validity_date_str.split('-')
                if len(date_parts) == 2:
                    start_date = datetime.strptime(date_parts[0], '%Y.%m.%d')
                    end_date = datetime.strptime(date_parts[1], '%Y.%m.%d')
                    return start_date, end_date
                else:
                    return None, None
        except ValueError:
            return None, None
    
    def calculate_age_years(self, birth_date: datetime, request_date: datetime) -> float:
        """
        计算从出生日期到请求日期的年龄（保留1位小数）
        
        Args:
            birth_date: 出生日期
            request_date: 请求日期
            
        Returns:
            年龄（年，保留1位小数）
        """
        if birth_date and request_date:
            delta = request_date - birth_date
            age_years = delta.days / 365.25  # 使用365.25考虑闰年
            return round(age_years, 1)
        return None
    
    def calculate_validity_days(self, validity_end_date: datetime, request_date: datetime) -> int:
        """
        计算从请求日期到证件过期日期的天数
        
        Args:
            validity_end_date: 证件过期日期
            request_date: 请求日期
            
        Returns:
            剩余有效天数（负数表示已过期）
        """
        if validity_end_date and request_date:
            delta = validity_end_date - request_date
            return delta.days
        elif validity_end_date is None:
            # 长期有效，返回一个较大的数值
            return 99999
        return None
    
    def get_valid_values_map(self) -> Dict[str, set]:
        """
        定义各字段的合法值集合
        
        Returns:
            字段名到合法值集合的映射
        """
        return {
            'degree': {'DOCTOR', 'MASTER', 'BACHELOR', 'COLLEGE', 'SENIOR', 'JUNIOR'},
            'maritalStatus': {'1', '2', '3'},
            'income': {'A', 'B', 'C', 'D'},
            'purpose': {'CONSUME', 'PAYRENT', 'TRAINING', 'TRAVEL', 'CREDIT'},
            'linkmanList.0.relationship': {'FATHER', 'MOTHER', 'MATE', 'CHILDREN', 'SIBLING', 
                                          'FRIENDS', 'COLLEAGUE', 'RELATIVES', 'OTHER', 'PARENTS'},
            'linkmanList.1.relationship': {'FATHER', 'MOTHER', 'MATE', 'CHILDREN', 'SIBLING', 
                                          'FRIENDS', 'COLLEAGUE', 'RELATIVES', 'OTHER', 'PARENTS'},
            'companyInfo.industry': {'A', 'C', 'E', 'R', 'D', 'F', 'K', 'L', 'O', 'N', 'H', 
                                    'I', 'G', 'J', 'P', 'M', 'Q', 'S', 'T', 'Z'},
            'companyInfo.occupation': {'11', '13', '17', '21', '24', '27', '31', '37', '51', 
                                      '54', '70', '80', '90', '91', '99'},
            'customerSource': {'APP', 'XCX'},
            'idInfo.nation': {'汉', '壮', '满', '回', '苗', '维吾尔', '彝', '土家', '藏', '蒙古', 
                             '侗', '布依', '瑶', '白', '朝鲜', '哈尼', '黎', '哈萨克', '傣', '畲', 
                             '傈僳', '东乡', '仡佬', '拉祜', '佤', '水', '纳西', '羌', '土', '仫佬', 
                             '锡伯', '柯尔克孜', '景颇', '达斡尔', '撒拉', '布朗', '毛南', '塔吉克', 
                             '普米', '阿昌', '怒', '鄂温克', '京', '基诺', '德昂', '保安', '俄罗斯', 
                             '裕固', '乌孜别克', '门巴', '鄂伦春', '独龙', '赫哲', '高山', '珞巴', '塔塔尔'}
        }
    
    def clean_data(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        数据清洗：将异常值直接设置为None
        
        Args:
            data_list: 原始数据列表
            
        Returns:
            清洗后的数据列表
        """
        if not data_list:
            return data_list
        
        valid_values_map = self.get_valid_values_map()
        cleaned_count = 0
        field_clean_count = defaultdict(int)
        
        for record in data_list:
            for field, valid_values in valid_values_map.items():
                if field in record:
                    value = record[field]
                    if value is not None and str(value) not in valid_values:
                        # 发现异常值，直接设置为None
                        record[field] = None
                        cleaned_count += 1
                        field_clean_count[field] += 1
        
        if cleaned_count > 0:
            print(f"  清理了 {cleaned_count} 个异常值 (设置为None)")
            for field, count in field_clean_count.items():
                print(f"    {field}: {count} 个")
        else:
            print(f"  未发现异常值")
        
        return data_list
    
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
    
    def extract_features(self, json_data: Dict[str, Any], feature_list: List[str], request_id: str = None) -> Dict[str, Any]:
        """
        从JSON数据中提取指定的特征
        
        Args:
            json_data: 解析后的JSON数据
            feature_list: 要提取的特征列表
            request_id: 请求ID，用于计算日期偏移
            
        Returns:
            提取后的特征字典
        """
        extracted = {}
        
        # 如果有请求ID，提取请求日期用于计算日期偏移
        request_date = None
        if request_id:
            request_date = self.extract_request_date_from_id(request_id)
        
        for feature in feature_list:
            value = self.extract_nested_value(json_data, feature)
            
            # 特殊处理日期字段
            if feature == 'idInfo.birthDate' and value and request_date:
                # 计算年龄（年，保留1位小数）
                birth_date = self.parse_birth_date(value)
                if birth_date:
                    value = self.calculate_age_years(birth_date, request_date)
                    
            elif feature == 'idInfo.validityDate' and value and request_date:
                # 计算证件剩余有效天数
                start_date, end_date = self.parse_validity_date(value)
                if end_date is not None:
                    value = self.calculate_validity_days(end_date, request_date)
                elif end_date is None and start_date:
                    # 长期有效
                    value = 99999
                else:
                    value = None
                    
            elif feature == 'idInfo.nation' and value:
                # 处理民族字段，去掉结尾多余的"族"
                value_str = str(value).strip()
                if value_str.endswith('族') and value_str != '族':
                    # 去掉结尾的"族"字，但保留本身就是"族"的情况
                    value = value_str[:-1]
                else:
                    value = value_str
                    
            # 处理特殊情况：如果是列表，抛出异常
            elif isinstance(value, list):
                raise ValueError(f"无法处理列表类型的特征: {feature}")
            
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
            
            request_id = row['id']
            partner_code = row['合作方编号']
            message = row['报文']
            result = row['结果']
            
            # 解析JSON报文
            json_data = self.parse_json_message(message)
            if not json_data:
                error_count += 1
                continue
            
            # 提取特征
            extracted_features = self.extract_features(json_data, feature_list, request_id)
            
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
            
            # 数据清洗：异常值设置为None
            print(f"处理 {partner_code} 数据...")
            cleaned_data_list = self.clean_data(data_list)
                
            # 创建DataFrame
            partner_df = pd.DataFrame(cleaned_data_list)
            
            # 保存为CSV
            output_file = os.path.join(self.output_dir, f"{partner_code}.csv")
            partner_df.to_csv(output_file, index=False, encoding='utf-8')
            
            partner_counts[partner_code] = len(cleaned_data_list)
            print(f"已生成 {output_file}，包含 {len(cleaned_data_list)} 条记录")
        
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
        df = pd.read_csv(self.data_file)
        
        analysis_results = {}
        
        # 分析每个特征
        for feature in feature_list:
            print(f"分析特征: {feature}")
            
            feature_values = []
            for _, row in df.iterrows():
                request_id = row['id']
                message = row['报文']
                json_data = self.parse_json_message(message)
                if json_data:
                    # 使用extract_features方法来正确处理日期字段
                    extracted = self.extract_features(json_data, [feature], request_id)
                    value = extracted.get(feature)
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

def main():
    """主函数，演示数据处理流程"""
    # 初始化数据处理器
    processor = DataProcessor('data/20250903.csv')
    
    # 使用示例特征列表
    feature_list = [
        'amount',
        'bankCardInfo.bankCode',
        'bankCardInfo.cardType',
        'city',
        'companyInfo.companyName',
        'companyInfo.industry',
        'companyInfo.occupation',
        'customerSource',
        'degree',
        'idInfo.birthDate',
        'idInfo.gender',
        'idInfo.identityType',
        'idInfo.nation',
        'idInfo.validityDate',
        'income',
        'jobFunctions',
        'linkmanList.0.relationship',
        'linkmanList.1.relationship',
        'maritalStatus',
        'province',
        'purpose',
        'resideFunctions',
        'term',
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
