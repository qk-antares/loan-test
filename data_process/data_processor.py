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
from collections import defaultdict, Counter
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
    """贷款数据预处理器，完整版本"""
    
    def __init__(self, output_dir: str = ".", manual_split_mode: bool = True):
        """
        初始化数据处理器

        Args:
            output_dir: 输出目录
            manual_split_mode: 是否启用手动分割模式，直接输出到train/test目录
        """
        self.output_dir = output_dir
        self.manual_split_mode = manual_split_mode

        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 如果是手动分割模式，创建train和test目录（在loan-test目录下）
        if self.manual_split_mode:
            # 获取loan-test目录路径
            script_dir = os.path.dirname(os.path.abspath(__file__))  # data_process目录
            loan_test_dir = os.path.dirname(script_dir)  # loan-test目录
            self.train_dir = os.path.join(loan_test_dir, "train")
            self.test_dir = os.path.join(loan_test_dir, "test")
            for directory in [self.train_dir, self.test_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)

        # 处理记录文件路径（保存在data_process目录下）
        script_dir = os.path.dirname(os.path.abspath(__file__))  # data_process目录
        self.processed_files_record = os.path.join(script_dir, 'processed_files.json')
        
        # 目前暂定使用以下特征（注释的不使用）
        self.feature_columns = [
            'amount',
            # 'bankCardInfo.bankName', # 直接使用下面的bankCode即可
            'bankCardInfo.bankCode',
            # 'bankCardInfo.cardType', # 所有样本都一样，无意义
            'city',
            'companyInfo.companyName',
            'companyInfo.industry',
            'companyInfo.occupation',
            'customerSource',
            'degree',
            'idInfo.birthDate',
            'idInfo.gender',
            # 'idInfo.identityType', # 所有样本都一样，无意义
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
            'deviceInfo.gpsLatitude',
            'deviceInfo.gpsLongitude',
            'deviceInfo.osType',
            'deviceInfo.isCrossDomain',
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

    def load_processed_files_record(self) -> Dict[str, Dict[str, Any]]:
        """
        加载已处理文件记录

        Returns:
            已处理文件记录字典 {file_path: {size: int, mtime: float, records_count: int}}
        """
        if os.path.exists(self.processed_files_record):
            try:
                with open(self.processed_files_record, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"警告: 无法读取处理记录文件: {e}")
                return {}
        return {}

    def save_processed_files_record(self, processed_files: Dict[str, Dict[str, Any]]):
        """
        保存已处理文件记录

        Args:
            processed_files: 处理记录字典
        """
        try:
            with open(self.processed_files_record, 'w', encoding='utf-8') as f:
                json.dump(processed_files, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"警告: 无法保存处理记录文件: {e}")

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件信息

        Args:
            file_path: 文件路径

        Returns:
            文件信息字典 {size: int, mtime: float}
        """
        try:
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'mtime': stat.st_mtime
            }
        except OSError:
            return {'size': 0, 'mtime': 0}

    def is_file_processed(self, file_path: str, processed_files: Dict[str, Dict[str, Any]]) -> bool:
        """
        检查文件是否已经处理过

        Args:
            file_path: 文件路径
            processed_files: 已处理文件记录

        Returns:
            是否已处理
        """
        if file_path not in processed_files:
            return False

        current_info = self.get_file_info(file_path)
        recorded_info = processed_files[file_path]

        # 比较文件大小和修改时间
        return (current_info['size'] == recorded_info.get('size', 0) and
                abs(current_info['mtime'] - recorded_info.get('mtime', 0)) < 1)

    def get_unprocessed_files(self, txt_files: List[str]) -> List[str]:
        """
        筛选出未处理的文件

        Args:
            txt_files: 所有txt文件列表

        Returns:
            未处理的文件列表
        """
        processed_files = self.load_processed_files_record()
        unprocessed_files = []

        for file_path in txt_files:
            if not self.is_file_processed(file_path, processed_files):
                unprocessed_files.append(file_path)
            else:
                print(f"跳过已处理文件: {os.path.basename(file_path)}")

        return unprocessed_files

    def ask_user_file_destination(self, file_path: str) -> str:
        """
        询问用户文件应该放到训练集还是测试集

        Args:
            file_path: 文件路径

        Returns:
            'train' 或 'test'
        """
        filename = os.path.basename(file_path)
        while True:
            choice = input(f"\n文件 '{filename}' 应该放到哪里？\n[1] 训练集 (train)\n[2] 测试集 (test)\n请选择 (1/2): ").strip()
            if choice == '1':
                return 'train'
            elif choice == '2':
                return 'test'
            else:
                print("无效选择，请输入 1 或 2")

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
            'channel_code': parts[1].strip(),  # 使用channelCode而不是channelName
            'channel_name': parts[2].strip(),  # channelName作为备用
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
            
            # 添加合作方信息（使用channelCode而不是channelName）
            features['partner_code'] = parsed_data['channel_code']
            
            self.stats['total_processed'] += 1
            partner = parsed_data['channel_code']
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
    
    def process_file(self, input_file: str, destination: str = None):
        """
        处理单个输入文件

        Args:
            input_file: 输入文件路径
            destination: 目标位置 ('train', 'test', 或 None 使用默认行为)
        """
        print(f"开始处理文件: {input_file}")

        if not os.path.exists(input_file):
            print(f"错误: 文件不存在 {input_file}")
            return

        # 在手动分割模式下，询问用户文件去向
        if self.manual_split_mode and destination is None:
            destination = self.ask_user_file_destination(input_file)

        # 按合作方分组的数据
        partner_data = defaultdict(list)
        all_data = []
        total_records_processed = 0  # 记录总处理数量
        
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
                        self._write_batch_data(all_data, partner_data, destination)
                        all_data = []
                        partner_data = defaultdict(list)

        # 写入剩余数据
        if all_data:
            self._write_batch_data(all_data, partner_data, destination)

        # 记录文件已处理
        self._record_processed_file(input_file, total_records_processed)

        print(f"文件 {input_file} 处理完成")

    def _record_processed_file(self, file_path: str, records_count: int):
        """
        记录文件已处理

        Args:
            file_path: 文件路径
            records_count: 处理的记录数
        """
        processed_files = self.load_processed_files_record()
        file_info = self.get_file_info(file_path)
        file_info['records_count'] = records_count
        file_info['processed_time'] = datetime.now().isoformat()

        processed_files[file_path] = file_info
        self.save_processed_files_record(processed_files)
    
    def _write_batch_data(self, all_data: List[Dict[str, Any]], partner_data: Dict[str, List[Dict[str, Any]]], destination: str = None):
        """
        批量写入数据，包含数据清洗

        Args:
            all_data: 所有数据
            partner_data: 按合作方分组的数据
            destination: 目标位置 ('train', 'test', 或 None 使用默认目录)
        """
        # 数据清洗
        print("开始数据清洗...")
        cleaned_all_data = self.clean_data(all_data.copy())

        # 确定输出目录
        if self.manual_split_mode and destination:
            if destination == 'train':
                output_dir = self.train_dir
                print(f"写入到训练集目录: {output_dir}")
            elif destination == 'test':
                output_dir = self.test_dir
                print(f"写入到测试集目录: {output_dir}")
            else:
                output_dir = self.output_dir
        else:
            output_dir = self.output_dir

        # 写入总表
        all_csv_path = os.path.join(output_dir, 'all_data.csv')
        self.append_to_csv(all_csv_path, cleaned_all_data, include_partner_code=True)

        # 写入分表
        for partner, records in partner_data.items():
            print(f"处理 {partner} 数据...")
            cleaned_records = self.clean_data(records.copy())

            # 清理合作方名称作为文件名
            safe_partner_name = re.sub(r'[<>:"/\\|?*]', '_', partner)
            partner_csv_path = os.path.join(output_dir, f'{safe_partner_name}.csv')
            self.append_to_csv(partner_csv_path, cleaned_records, include_partner_code=False)
    
    def process_multiple_files(self, input_files: List[str]):
        """
        处理多个输入文件
        
        Args:
            input_files: 输入文件路径列表
        """
        for input_file in input_files:
            self.process_file(input_file)
        
        self.print_statistics()
    
    def process_data_directory(self, data_dir: str = "data"):
        """
        处理data目录下的所有txt文件
        
        Args:
            data_dir: 数据目录
        """
        # 获取脚本所在目录的父目录（backup的上一级），然后找data目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_path = os.path.join(parent_dir, data_dir)
        
        print(f"脚本所在目录: {script_dir}")
        print(f"查找数据目录: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"错误: 数据目录不存在 {data_path}")
            print("请确保data目录与脚本在同一目录下")
            return
        
        # 查找所有txt文件
        txt_files = []
        for file in os.listdir(data_path):
            if file.endswith('.txt'):
                txt_files.append(os.path.join(data_path, file))
        
        if not txt_files:
            print(f"警告: 在 {data_path} 目录下未找到任何txt文件")
            print(f"目录内容: {os.listdir(data_path)}")
            return
        
        print(f"找到 {len(txt_files)} 个txt文件:")
        for f in txt_files:
            print(f"  - {f}")

        # 筛选出未处理的文件（增量处理）
        unprocessed_files = self.get_unprocessed_files(txt_files)

        if not unprocessed_files:
            print("所有文件都已处理完成，无需重复处理")
            return

        print(f"需要处理 {len(unprocessed_files)} 个新文件:")
        for f in unprocessed_files:
            print(f"  - {os.path.basename(f)}")

        # 处理未处理的文件
        self.process_multiple_files(unprocessed_files)
    
    def analyze_processed_data(self) -> Dict[str, Any]:
        """
        分析处理后的数据特征分布，参考原代码逻辑
        
        Returns:
            特征分析结果
        """
        all_csv_path = os.path.join(self.output_dir, 'all_data.csv')
        if not os.path.exists(all_csv_path):
            print("错误: 总表文件不存在，无法进行特征分析")
            return {}
        
        print("开始特征分析...")
        df = pd.read_csv(all_csv_path)
        
        analysis_results = {}
        
        # 分析每个特征
        for feature in self.feature_columns:
            if feature not in df.columns:
                continue
                
            print(f"分析特征: {feature}")
            
            # 获取非空值
            feature_series = df[feature].dropna()
            
            if len(feature_series) == 0:
                analysis_results[feature] = {
                    'type': 'empty',
                    'count': 0,
                    'unique_count': 0,
                    'values': {}
                }
                continue
            
            # 转换为字符串进行统计
            feature_values = feature_series.astype(str).tolist()
            
            # 统计分析
            value_counts = Counter(feature_values)
            total_count = len(feature_values)
            unique_count = len(value_counts)
            null_count = df[feature].isnull().sum()
            
            # 判断数据类型
            numeric_values = []
            for v in feature_values:
                if v != 'nan':
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        break
            
            is_numeric = len(numeric_values) == len([v for v in feature_values if v != 'nan'])
            
            if is_numeric and unique_count > 10 and len(numeric_values) > 0:
                # 数值型特征
                analysis_results[feature] = {
                    'type': 'numeric',
                    'count': total_count,
                    'null_count': null_count,
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
                    'null_count': null_count,
                    'unique_count': unique_count,
                    'values': dict(value_counts.most_common(20))
                }
        
        # 分析结果标签
        if 'label' in df.columns:
            label_series = df['label'].dropna()
            result_counts = Counter(label_series.astype(int))
            
            analysis_results['label'] = {
                'type': 'categorical',
                'count': len(label_series),
                'null_count': df['label'].isnull().sum(),
                'unique_count': len(result_counts),
                'values': dict(result_counts)
            }
        
        return analysis_results
    
    def print_analysis_report(self, analysis_results: Dict[str, Any]):
        """
        打印分析报告，参考原代码
        
        Args:
            analysis_results: 分析结果
        """
        print("\n" + "="*60)
        print("特征分析报告 (基于清理后的数据)")
        print("="*60)
        
        for feature, stats in analysis_results.items():
            print(f"\n特征: {feature}")
            print("-" * 40)
            print(f"数据类型: {stats['type']}")
            print(f"有效数据数: {stats['count']}")
            if 'null_count' in stats:
                print(f"空值数量: {stats['null_count']}")
            print(f"唯一值数量: {stats['unique_count']}")
            
            if stats['type'] == 'numeric':
                print(f"最小值: {stats['min']}")
                print(f"最大值: {stats['max']}")
                print(f"平均值: {stats['mean']:.2f}")
                if 'top_values' in stats:
                    print("最常见的值:")
                    for value, count in stats['top_values'].items():
                        print(f"  {value}: {count}")
            elif stats['type'] == 'categorical':
                if 'values' in stats:
                    print("值分布:")
                    for value, count in stats['values'].items():
                        percentage = (count / stats['count']) * 100
                        print(f"  {value}: {count} ({percentage:.1f}%)")
            elif stats['type'] in ['empty', 'missing']:
                print("注意: 该特征没有有效数据")
    
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
        
        # 进行特征分析
        analysis_results = self.analyze_processed_data()
        if analysis_results:
            self.print_analysis_report(analysis_results)
            
            # 保存特征分析结果（保存在data_process目录下）
            script_dir = os.path.dirname(os.path.abspath(__file__))  # data_process目录
            analysis_path = os.path.join(script_dir, 'feature_analysis.json')
            with open(analysis_path, 'w', encoding='utf-8') as f:
                # 转换为可JSON序列化的格式
                serializable_results = {}
                for feature, stats in analysis_results.items():
                    serializable_results[feature] = {}
                    for key, value in stats.items():
                        if isinstance(value, (int, float, str, list, dict)):
                            serializable_results[feature][key] = value
                        else:
                            serializable_results[feature][key] = str(value)
                
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            print(f"\n特征分析结果已保存到: {analysis_path}")
        
        # 保存统计报告（保存在data_process目录下）
        script_dir = os.path.dirname(os.path.abspath(__file__))  # data_process目录
        report_path = os.path.join(script_dir, 'processing_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("数据处理统计报告\n")
            f.write("="*60 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总处理记录数: {self.stats['total_processed']}\n")
            f.write(f"JSON解析错误数: {self.stats['json_parse_errors']}\n")
            f.write(f"行格式错误数: {self.stats['line_parse_errors']}\n")
            f.write(f"特征提取错误数: {self.stats['feature_extract_errors']}\n")
            f.write(f"总清洗异常值数: {self.stats['cleaned_values_count']}\n\n")
            
            if self.stats['field_clean_stats']:
                f.write("字段清洗统计:\n")
                f.write("-" * 40 + "\n")
                for field, count in sorted(self.stats['field_clean_stats'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{field}: {count} 个异常值\n")
                f.write("\n")
            
            f.write("各合作方数据统计:\n")
            f.write("-" * 40 + "\n")
            for partner, count in sorted(self.stats['partner_counts'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"{partner}: {count} 条记录\n")
        
        print(f"\n统计报告已保存到: {report_path}")


    def add_force_reprocess_option(self):
        """
        添加强制重新处理选项（删除处理记录）
        """
        if os.path.exists(self.processed_files_record):
            os.remove(self.processed_files_record)
            print("已删除处理记录文件，将重新处理所有文件")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='贷款数据预处理脚本（支持增量处理和手动分割）')
    parser.add_argument('--force', action='store_true',
                       help='强制重新处理所有文件，忽略处理记录')
    parser.add_argument('--data_dir', default='data',
                       help='原始数据目录 (默认: data)')
    parser.add_argument('--auto-split', action='store_true',
                       help='关闭手动分割模式，使用默认输出目录（默认为手动分割）')

    args = parser.parse_args()

    # 创建处理器 (默认启用手动分割模式，除非指定了auto_split)
    processor = LoanDataProcessor(manual_split_mode=not args.auto_split)

    print("="*60)
    print("贷款数据预处理脚本")
    print("基于参考代码的完整数据验证和清洗逻辑")
    print("支持增量处理，避免重复处理已处理的文件")
    if not args.auto_split:
        print("手动分割模式：将为每个txt文件询问放到训练集还是测试集")
    else:
        print("自动模式：使用默认输出目录")
    print("="*60)

    if args.force:
        processor.add_force_reprocess_option()
    
    # 处理data目录下的所有txt文件
    processor.process_data_directory(args.data_dir)
    
    print("\n" + "="*60)
    print("数据预处理完成！")
    print("="*60)
    if not args.auto_split:
        print("手动分割模式输出文件:")
        print(f"  训练集:")
        print(f"    - 总表: {os.path.join(processor.train_dir, 'all_data.csv')}")
        print(f"    - 分表: {processor.train_dir}/<合作方名称>.csv")
        print(f"  测试集:")
        print(f"    - 总表: {os.path.join(processor.test_dir, 'all_data.csv')}")
        print(f"    - 分表: {processor.test_dir}/<合作方名称>.csv")
        print(f"  处理记录: {processor.processed_files_record}")
    else:
        print("输出文件:")
        print(f"  - 总表: {os.path.join(processor.output_dir, 'all_data.csv')}")
        print(f"  - 分表: {processor.output_dir}/<合作方名称>.csv")
        print(f"  - 处理报告: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processing_report.txt')}")
        print(f"  - 特征分析: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_analysis.json')}")
    
    print("\n说明:")
    print("1. 总表包含所有合作方的数据")
    print("2. 分表按合作方自动创建")
    print("3. 支持增量处理，重复运行会追加数据")
    print("4. 所有数据已完成清洗和验证")


if __name__ == "__main__":
    main()