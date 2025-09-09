"""
工具函数模块
包含日期处理、数据转换等通用功能
"""

from datetime import datetime
from typing import Optional, Tuple


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
        return datetime(2025, 9, 3)


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
