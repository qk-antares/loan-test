"""
工具函数模块
包含日期处理、数据转换等通用功能
"""

from datetime import datetime
from typing import Optional, Tuple, Any, Dict
import os
import re

def extract_request_date_from_id(request_id: str) -> datetime:
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
        return datetime.now()


def parse_birth_date(birth_date_str: str) -> Optional[datetime]:
    """
    解析出生日期字符串
    
    Args:
        birth_date_str: 出生日期字符串，格式如 "19930125"
        
    Returns:
        出生日期的datetime对象，解析失败返回None
    """
    try:
        return datetime.strptime(birth_date_str, '%Y%m%d')
    except ValueError:
        return None


def parse_validity_date(validity_date_str: str) -> Tuple[Optional[datetime], Optional[datetime]]:
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


def calculate_age_years(birth_date: datetime, request_date: datetime) -> Optional[float]:
    """
    计算从出生日期到请求日期的年龄（保留1位小数）
    
    Args:
        birth_date: 出生日期
        request_date: 请求日期
        
    Returns:
        年龄（年，保留1位小数），计算失败返回None
    """
    if birth_date and request_date:
        delta = request_date - birth_date
        age_years = delta.days / 365.25  # 使用365.25考虑闰年
        return round(age_years, 1)
    return None


def calculate_validity_days(validity_end_date: Optional[datetime], request_date: datetime) -> Optional[int]:
    """
    计算从请求日期到证件过期日期的天数
    
    Args:
        validity_end_date: 证件过期日期，None表示长期有效
        request_date: 请求日期
        
    Returns:
        剩余有效天数（负数表示已过期），长期有效返回99999
    """
    if validity_end_date and request_date:
        delta = validity_end_date - request_date
        return delta.days
    elif validity_end_date is None:
        # 长期有效，返回一个较大的数值
        return 99999
    return None


def process_nation_field(value: str) -> str:
    """
    处理民族字段，去掉结尾多余的"族"
    
    Args:
        value: 原始民族字段值
        
    Returns:
        处理后的民族字段值
    """
    if not value:
        return value
    
    value_str = str(value).strip()
    if value_str.endswith('族') and value_str != '族':
        # 去掉结尾的"族"字，但保留本身就是"族"的情况
        return value_str[:-1]
    else:
        return value_str

def encode_degree(degree_str: str) -> Optional[int]:
    """学历编码：从低到高编码为1-6"""
    if not degree_str:
        return None

    degree_map = {
        'JUNIOR': 1,    # 初中
        'SENIOR': 2,    # 高中
        'COLLEGE': 3,   # 大专
        'BACHELOR': 4,  # 本科
        'MASTER': 5,    # 硕士
        'DOCTOR': 6     # 博士（与硕士同级）
    }
    return degree_map.get(degree_str.upper(), None)

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

def get_valid_values_map() -> Dict[str, set]:
    """
    定义各字段的合法值集合，参考原代码
    
    Returns:
        字段名到合法值集合的映射
    """
    return {
        'degree': {'1', '2', '3', '4', '5', '6'},  # 编码后的学历等级
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

def extract_date_from_filename(input_file: str) -> str:
    # 获取文件名
    filename = os.path.basename(input_file)
    # 用正则提取日期（假设文件名格式为 2025-09-20.txt）
    match = re.match(r'(\d{4}-\d{2}-\d{2})\.txt$', filename)
    if match:
        return match.group(1)
    return None