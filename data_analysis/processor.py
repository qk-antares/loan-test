# -*- coding: utf-8 -*-
"""
数据预处理脚本
用于处理以||为分隔符的贷款数据，输出CSV格式文件供机器学习使用
基于参考代码的完整数据验证和清洗逻辑
"""

import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import re


class DateUtils:
    """日期工具类，参考原代码的日期处理逻辑"""
    
    @staticmethod
    def extract_request_date_from_id(request_id: str) -> datetime:
        """从请求ID中提取申请日期"""
        try:
            date_str = request_id[:8]
            return datetime.strptime(date_str, "%Y%m%d")
        except (ValueError, IndexError):
            return datetime.now()
    
    @staticmethod
    def parse_birth_date(birth_date_str: str) -> Optional[datetime]:
        """解析出生日期"""
        if not birth_date_str:
            return None
        try:
            return datetime.strptime(birth_date_str, "%Y%m%d")
        except ValueError:
            return None
    
    @staticmethod
    def parse_validity_date(validity_str: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """解析身份证有效期"""
        if not validity_str:
            return None, None
        
        try:
            if '-' in validity_str:
                start_str, end_str = validity_str.split('-')
                start_date = datetime.strptime(start_str.strip(), "%Y.%m.%d")
                
                # 处理长期有效
                if '长期' in end_str or end_str.strip() == '':
                    return start_date, None
                
                end_date = datetime.strptime(end_str.strip(), "%Y.%m.%d")
                return start_date, end_date
            else:
                return None, None
        except ValueError:
            return None, None
    
    @staticmethod
    def calculate_age_years(birth_date: datetime, request_date: datetime) -> float:
        """计算年龄（年，保留1位小数）"""
        age_days = (request_date - birth_date).days
        age_years = age_days / 365.25
        return round(age_years, 1)
    
    @staticmethod
    def calculate_validity_days(end_date: datetime, request_date: datetime) -> int:
        """计算剩余有效天数"""
        remaining_days = (end_date - request_date).days
        return max(0, remaining_days)
    
    @staticmethod
    def process_nation_field(nation_str: str) -> str:
        """处理民族字段，去掉结尾多余的"族"，参考原代码逻辑"""
        if not nation_str:
            return nation_str

        nation = nation_str.strip()

        # 对于结尾有"族"的，去掉"族"字
        if nation.endswith('族') and len(nation) > 1:
            nation = nation[:-1]

        return nation

    @staticmethod
    def encode_degree(degree_str: str) -> Optional[int]:
        """学历编码：从低到高编码为1-5"""
        if not degree_str:
            return None

        degree_map = {
            'JUNIOR': 1,    # 初中
            'SENIOR': 2,    # 高中
            'COLLEGE': 3,   # 大专
            'BACHELOR': 4,  # 本科
            'MASTER': 5,    # 硕士
            'DOCTOR': 5     # 博士（与硕士同级）
        }
        return degree_map.get(degree_str.upper(), None)

    @staticmethod
    def encode_income(income_str: str) -> Optional[int]:
        """收入等级编码：按收入等级编码为1-4"""
        if not income_str:
            return None

        income_map = {
            'A': 1,  # 最低收入等级
            'B': 2,
            'C': 3,
            'D': 4   # 最高收入等级
        }
        return income_map.get(income_str.upper(), None)

    @staticmethod
    def process_bank_name(bank_name_str: str) -> str:
        """处理银行名称异常：去掉多余的"中国"前缀，但保留"中国银行"完整名称"""
        if not bank_name_str:
            return bank_name_str

        bank_name = bank_name_str.strip()

        # 特殊处理：如果是"中国银行"，保持不变
        if bank_name == "中国银行":
            return bank_name

        # 去掉多余的"中国"前缀，例如"中国建设银行"改为"建设银行"
        if bank_name.startswith("中国") and len(bank_name) > 2:
            # 检查去掉"中国"后的名称是否合理（至少包含"银行"）
            remaining = bank_name[2:]
            if "银行" in remaining:
                return remaining

        return bank_name

    @staticmethod
    def check_for_masked_data(value: Any, field_name: str) -> None:
        """检查是否包含脱敏数据，如果发现则抛出错误"""
        if value is None:
            return

        str_value = str(value).strip()

        # 常见的脱敏标识
        masked_patterns = ['*', '***', '****', 'null', 'NULL', '脱敏', '已脱敏']

        for pattern in masked_patterns:
            if pattern in str_value:
                raise ValueError(f"检测到脱敏数据在字段 '{field_name}': {str_value}")


class LoanDataProcessor:
    """贷款数据预处理器，精简版本"""
    
    def __init__(self, output_dir: str = ".", mode: str = "train"):
        """
        初始化数据处理器

        Args:
            output_dir: 输出目录
            mode: 处理模式 - "train" 或 "test"
        """
        self.output_dir = output_dir
        self.mode = mode

        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 目前暂定使用以下特征
        self.feature_columns = [
            'amount',
            'bankCardInfo.bankCode',
            'city',
            'companyInfo.companyName',
            'companyInfo.industry',
            'companyInfo.occupation',
            'customerSource',
            'degree',
            'idInfo.birthDate',
            'idInfo.gender',
            'idInfo.nation',
            'idInfo.validityDate',
            'income',
            'jobFunctions',
            'linkmanList.0.relationship',
            'linkmanList.1.relationship',
            'maritalStatus',
            'pictureInfo.0.faceScore',
            'province',
            'purpose',
            'resideFunctions',
            'term',
            'deviceInfo.osType',
            'deviceInfo.isCrossDomain',
            'deviceInfo.applyPos',
            'label'
        ]
        
        # 处理统计
        self.stats = {
            'total_processed': 0,
            'json_parse_errors': 0,
            'line_parse_errors': 0,
            'feature_extract_errors': 0,
            'partner_counts': {},
            'cleaned_values_count': 0,
            'field_clean_stats': defaultdict(int)
        }

    def parse_line(self, line: str) -> Optional[Dict[str, str]]:
        """
        解析单行数据
        
        Args:
            line: 原始数据行
            
        Returns:
            解析后的字典，包含id, error_code, partner_code, json_data, result_status
        """
        line = line.strip()
        if not line:
            return None
        
        # 按||分割
        parts = line.split('||')
        if len(parts) != 5:
            print(f"警告: 数据格式不正确，应为5个字段，实际为{len(parts)}个字段")
            self.stats['line_parse_errors'] += 1
            return None
        
        return {
            'request_id': parts[0].strip(),
            'error_code': parts[1].strip(),
            'partner_code': parts[1].strip(),
            'json_data': parts[3].strip(),
            'result_status': parts[4].strip()
        }
    
    def parse_json_message(self, json_str: str) -> Dict[str, Any]:
        """
        解析JSON格式的报文，参考原代码
        
        Args:
            json_str: JSON字符串
            
        Returns:
            解析后的字典
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            self.stats['json_parse_errors'] += 1
            return {}
    
    def extract_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """
        从嵌套字典中提取值，支持点号分隔的路径，参考原代码
        
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
            
            if value == '':
                return None
            return value
        except (KeyError, IndexError, ValueError, TypeError):
            return None
    
    def extract_features(self, json_data: Dict[str, Any], feature_list: List[str], request_id: str = None) -> Dict[str, Any]:
        """
        从JSON数据中提取指定的特征，参考原代码逻辑
        
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
            request_date = DateUtils.extract_request_date_from_id(request_id)
        
        for feature in feature_list:
            if feature == 'label':
                # 跳过label字段，在外层处理
                continue
                
            value = self.extract_nested_value(json_data, feature)

            # 检查脱敏数据
            try:
                DateUtils.check_for_masked_data(value, feature)
            except ValueError as e:
                print(f"警告: {e}")
                self.stats['feature_extract_errors'] += 1
                continue

            # 特殊处理日期字段
            if feature == 'idInfo.birthDate' and value and request_date:
                # 计算年龄（年，保留1位小数）
                birth_date = DateUtils.parse_birth_date(value)
                if birth_date:
                    value = DateUtils.calculate_age_years(birth_date, request_date)
                    
            elif feature == 'idInfo.validityDate' and value and request_date:
                # 计算证件剩余有效天数
                start_date, end_date = DateUtils.parse_validity_date(value)
                if end_date is not None:
                    value = DateUtils.calculate_validity_days(end_date, request_date)
                elif end_date is None and start_date:
                    # 长期有效
                    value = 99999
                else:
                    value = None
                    
            elif feature == 'idInfo.nation' and value:
                # 处理民族字段，去掉结尾多余的"族"
                value = DateUtils.process_nation_field(value)

            elif feature == 'degree' and value:
                # 学历编码：从低到高编码为1-5
                value = DateUtils.encode_degree(value)

            elif feature == 'income' and value:
                # 收入等级编码：按收入等级编码为1-4
                value = DateUtils.encode_income(value)

            elif feature == 'bankCardInfo.bankName' and value:
                # 处理银行名称异常：去掉多余的"中国"前缀
                value = DateUtils.process_bank_name(value)

            # 处理特殊情况：如果是列表，抛出异常
            elif isinstance(value, list):
                raise ValueError(f"无法处理列表类型的特征: {feature}")
            
            extracted[feature] = value
        
        return extracted
    
    def clean_data(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        数据清洗：将异常值直接设置为None，参考原代码逻辑
        
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
                        self.stats['field_clean_stats'][field] += 1
        
        self.stats['cleaned_values_count'] += cleaned_count
        
        if cleaned_count > 0:
            print(f"  清理了 {cleaned_count} 个异常值 (设置为None)")
            for field, count in field_clean_count.items():
                print(f"    {field}: {count} 个")
        else:
            print(f"  未发现异常值")
        
        return data_list

    def get_valid_values_map(self) -> Dict[str, set]:
        """
        定义各字段的合法值集合，参考原代码
        
        Returns:
            字段名到合法值集合的映射
        """
        return {
            'degree': {'1', '2', '3', '4', '5'},  # 编码后的学历等级
            'maritalStatus': {'1', '2', '3'},
            'income': {'1', '2', '3', '4'},  # 编码后的收入等级
            'purpose': {'CONSUME', 'PAYRENT', 'TRAINING', 'TRAVEL', 'CREDIT'},
            'linkmanList.0.relationship': {'FATHER', 'MOTHER', 'MATE', 'CHILDREN', 'SIBLING', 
                                          'FRIENDS', 'COLLEAGUE', 'RELATIVES', 'OTHER', 'PARENTS'},
            'linkmanList.1.relationship': {'FATHER', 'MOTHER', 'MATE', 'CHILDREN', 'SIBLING', 
                                          'FRIENDS', 'COLLEAGUE', 'RELATIVES', 'OTHER', 'PARENTS'},
            'companyInfo.industry': {'A', 'C', 'E', 'R', 'D', 'F', 'K', 'L', 'O', 'N', 'H', 
                                    'I', 'G', 'J', 'P', 'M', 'Q', 'S', 'T', 'Z'},
            'companyInfo.occupation': {'11', '13', '17', '21', '24', '27', '31', '37', '51', 
                                      '54', '70', '80', '90', '91', '99'},
            'idInfo.nation': {'汉', '壮', '满', '回', '苗', '维吾尔', '彝', '土家', '藏', '蒙古', 
                             '侗', '布依', '瑶', '白', '朝鲜', '哈尼', '黎', '哈萨克', '傣', '畲', 
                             '傈僳', '东乡', '仡佬', '拉祜', '佤', '水', '纳西', '羌', '土', '仫佬', 
                             '锡伯', '柯尔克孜', '景颇', '达斡尔', '撒拉', '布朗', '毛南', '塔吉克', 
                             '普米', '阿昌', '怒', '鄂温克', '京', '基诺', '德昂', '保安', '俄罗斯', 
                             '裕固', '乌孜别克', '门巴', '鄂伦春', '独龙', '赫哲', '高山', '珞巴', '塔塔尔'},
            'idInfo.gender': {'M', 'F'},
            'customerSource': {'APP', 'XCX'},
            'jobFunctions': {'01', '02', '03'},
            'resideFunctions': {'01', '02', '03', '04'},
            'deviceInfo.osType': {'ANDROID', 'IOS'}
        }
    
    def process_single_record(self, line: str) -> Optional[Dict[str, Any]]:
        """
        处理单条记录
        
        Args:
            line: 单行数据
            
        Returns:
            处理后的特征字典，包含partner_code
        """
        parsed_data = self.parse_line(line)
        if not parsed_data:
            return None
        
        json_data = self.parse_json_message(parsed_data['json_data'])
        if not json_data:
            return None
        
        try:
            # 提取特征
            features = self.extract_features(json_data, self.feature_columns, parsed_data['request_id'])
            
            # 添加结果标签（成功=1，失败=0）
            features['label'] = 1 if parsed_data['result_status'] == '成功' else 0
            
            # 添加合作方信息
            features['partner_code'] = parsed_data['partner_code']
            
            self.stats['total_processed'] += 1
            partner = parsed_data['partner_code']
            self.stats['partner_counts'][partner] = self.stats['partner_counts'].get(partner, 0) + 1
            
            return features
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            self.stats['feature_extract_errors'] += 1
            return None
    
    def append_to_csv(self, filepath: str, data: List[Dict[str, Any]], include_partner_code: bool = True):
        """
        追加数据到CSV文件
        
        Args:
            filepath: CSV文件路径
            data: 要追加的数据列表
            include_partner_code: 是否包含partner_code列（总表需要，分表不需要）
        """
        if not data:
            return
        
        # 准备列顺序
        columns = self.feature_columns.copy()
        if include_partner_code:
            columns.insert(0, 'partner_code')
        
        # 准备数据
        df_data = []
        for record in data:
            row = {}
            for col in columns:
                if col in record:
                    row[col] = record[col]
                else:
                    row[col] = None
            df_data.append(row)
        
        df = pd.DataFrame(df_data, columns=columns)
        
        # 检查文件是否存在
        file_exists = os.path.exists(filepath)
        
        # 写入文件
        df.to_csv(filepath, mode='a', header=not file_exists, index=False, encoding='utf-8')
        
        print(f"已追加 {len(data)} 条记录到 {filepath}")
    
    def process_file(self, input_file: str):
        """
        处理单个输入文件

        Args:
            input_file: 输入文件路径
        """
        print(f"开始处理文件: {input_file}")

        if not os.path.exists(input_file):
            print(f"错误: 文件不存在 {input_file}")
            return

        # 按合作方分组的数据
        partner_data = defaultdict(list)
        all_data = []
        total_records_processed = 0
        
        # 逐行处理
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 1000 == 0:
                    print(f"已处理 {line_num} 行")
                
                record = self.process_single_record(line)
                if record:
                    partner = record['partner_code']

                    # 添加到总数据
                    all_data.append(record)
                    total_records_processed += 1

                    # 按合作方分组
                    partner_data[partner].append(record)

                    # 批量写入（每1000条）
                    if len(all_data) >= 1000:
                        self._write_batch_data(all_data, partner_data)
                        all_data = []
                        partner_data = defaultdict(list)

        # 写入剩余数据
        if all_data:
            self._write_batch_data(all_data, partner_data)

        print(f"文件 {input_file} 处理完成，共处理 {total_records_processed} 条记录")
    
    def _write_batch_data(self, all_data: List[Dict[str, Any]], partner_data: Dict[str, List[Dict[str, Any]]]):
        """
        批量写入数据，包含数据清洗

        Args:
            all_data: 所有数据
            partner_data: 按合作方分组的数据
        """
        # 数据清洗
        print("开始数据清洗...")
        cleaned_all_data = self.clean_data(all_data.copy())

        # 写入总表
        all_csv_path = os.path.join(self.output_dir, 'all_data.csv')
        self.append_to_csv(all_csv_path, cleaned_all_data, include_partner_code=True)

        # # 写入分表
        # for partner, records in partner_data.items():
        #     print(f"处理 {partner} 数据...")
        #     cleaned_records = self.clean_data(records.copy())

        #     # 清理合作方名称作为文件名
        #     safe_partner_name = re.sub(r'[<>:"/\\|?*]', '_', partner)
        #     partner_csv_path = os.path.join(self.output_dir, f'{safe_partner_name}.csv')
        #     self.append_to_csv(partner_csv_path, cleaned_records, include_partner_code=False)
    
    def process_data_directory(self, data_dir: str = "data"):
        """
        处理data目录下的txt文件，根据模式选择不同的文件
        """
        mode = self.mode.lower()
        if mode not in ["train", "test"]:
            print(f"错误: 模式参数必须是 'train' 或 'test'")
            return
        
        # 获取数据目录路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_path = os.path.join(parent_dir, data_dir)
        
        if not os.path.exists(data_path):
            print(f"错误: 数据目录不存在 {data_path}")
            return
        
        # 查找所有txt文件
        txt_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
        
        if not txt_files:
            print(f"在 {data_path} 目录下未找到任何txt文件")
            return
        
        # 按文件名排序
        txt_files.sort(key=lambda x: os.path.basename(x))
        
        # 根据模式选择文件
        if mode == "train":
            files_to_process = txt_files[:-1] if len(txt_files) > 1 else []
            mode_desc = "训练模式（排除最新文件）"
        else:  # test模式
            files_to_process = [txt_files[-1]] if txt_files else []
            mode_desc = "测试模式（只处理最新文件）"
        
        if not files_to_process:
            print(f"{mode_desc}：没有需要处理的文件")
            return
        
        print(f"{mode_desc}：处理 {len(files_to_process)} 个文件")
        
        # 直接处理所有选择的文件
        for input_file in files_to_process:
            self.process_file(input_file)
        
        self.print_statistics()
        print("文件处理完成！")
    
    def print_statistics(self):
        """打印处理统计信息"""
        print("\n" + "="*60)
        print("数据处理统计报告")
        print("="*60)
        print(f"总处理记录数: {self.stats['total_processed']}")
        print(f"JSON解析错误数: {self.stats['json_parse_errors']}")
        print(f"行格式错误数: {self.stats['line_parse_errors']}")
        print(f"特征提取错误数: {self.stats['feature_extract_errors']}")
        print(f"总清洗异常值数: {self.stats['cleaned_values_count']}")
        
        if self.stats['field_clean_stats']:
            print("\n字段清洗统计:")
            print("-" * 40)
            for field, count in sorted(self.stats['field_clean_stats'].items(), key=lambda x: x[1], reverse=True):
                print(f"{field}: {count} 个异常值")
        
        print("\n各合作方数据统计:")
        print("-" * 40)
        for partner, count in sorted(self.stats['partner_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"{partner}: {count} 条记录")


def main():

    # 创建处理器
    processor = LoanDataProcessor(output_dir='data', mode='train')

    
    # 处理data目录下的所有txt文件
    processor.process_data_directory('data')
   


if __name__ == "__main__":
    main()     
