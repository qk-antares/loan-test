import pandas as pd
import json
import re
from typing import Dict, Any


class DataCheck:
        
    def __init__(self, data_file: str, original_data_file: str, unused_data_file: str):
        """
        初始化数据分析器
        
        """
        self.data_file = data_file #实际使用的数据
        self.original_data_file = original_data_file #原始数据，包含被过滤掉的特征
        self.unused_data_file = unused_data_file #使用了applyPos字段的版本
        self.masked_features = [] # 脱敏特征
        self.filtered_features = [] # 预处理被过滤掉的特征
        self.invalid_data = [] # 未通过筛选条件的数据
        self.missing_data = [] # 存在筛选条件特征值缺失的数据
        


    """
    解析JSON格式的报文
    """
    def parse_json_message(self, json_str: str) -> Dict[str, Any]:
     
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return {}
             
   
    """
    将嵌套字典扁平化为并列的键值对
    """       
    def flatten_dict(self, data, parent_key="", separator="."):
        
        items = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                
                if isinstance(value, dict):
                    # 递归处理嵌套字典
                    items.update(self.flatten_dict(value, new_key, separator))
                elif isinstance(value, list):
                    # 处理列表，将列表索引作为key的一部分
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            items.update(self.flatten_dict(item, f"{new_key}.{i}", separator))
                        else:
                            items[f"{new_key}[{i}]"] = item
                else:
                    items[new_key] = value
        else:
            items[parent_key] = data
        
        return items
    
    """
    原始数据简单处理
    """
    def data_process(self, df: pd.DataFrame):
        process_data = []
        for idx, row in df.iterrows():    
            message = row['content']
            json_data = self.parse_json_message(message)
            data = self.flatten_dict(json_data)
            process_data.append(data)
        return process_data

    """
    合作方分布与通过率分析报告
    """
    def calculate_distribution(self, df: pd.DataFrame):
        print("=" * 60)
        print("合作方分布与通过率分析报告")
        print("=" * 60)

        # 1. 基础统计信息
        print(f"数据总记录数: {len(df)} 笔")
        print(f"总体通过率: {(df['label'].sum() / len(df) * 100):.2f}%")
        print()

        # 2. 合作方分布
        partner_counts = df['partner_code'].value_counts().sort_values(ascending=False)
        print("合作方分布:")
        for partner, count in partner_counts.items():
            print(f"{partner}: {count} 笔 ({count / len(df) * 100:.2f}%)")
        print()

        # 3. 各合作方通过率分析
        print("各合作方通过率分析:")
        for partner, count in partner_counts.items():
            partner_df = df[df['partner_code'] == partner]
            pass_rate = partner_df['label'].mean()
            print(f"{partner}: {count} 笔, 通过 {partner_df['label'].sum()} 笔,未通过{count-partner_df['label'].sum()}笔, 通过率 {pass_rate * 100:.2f}%")
        print()


    """ 
    检查单个值是否看起来像脱敏数据
    """
    def is_masked_value(self, value):
        value_str = str(value)
         # 定义脱敏模式的正则表达式
        masking_patterns = {
            '手机号脱敏': r'^1[3-9]\*{4,6}\d{2,4}$',  # 13*****03, 18*****62
            '身份证脱敏': r'^\d{3}\*{4,6}\d{3,4}$',   # 320*****014, 420*****516
            '银行卡脱敏': r'^\d{3}\*{4,6}\d{3,4}$',   # 621*****076, 621*****611
            '姓名脱敏': r'^[\u4e00-\u9fa5]\*{3,5}[\u4e00-\u9fa5]$',  # 朱*****祥, 胡*****宝
            '通用脱敏': r'.*\*{3,}.*',  # 包含连续3个及以上星号的任何字符串
            '部分隐藏': r'^.{2}\*{4,}.{2}$',  # 前后保留2位，中间隐藏
        }
        
        # 检查各种脱敏模式
        for pattern_name, pattern in masking_patterns.items():
            if re.match(pattern, value_str):
                return True
        # 检查URL脱敏（包含***的URL）
        if '***' in value_str and ('http' in value_str or '//' in value_str):
            return True
            
        return False
    
    def clean_feature_name(self, original_features):
        """
        清理特征名称，去除列表带来的重复
        """
        cleaned_features = set()
        for feature in original_features:
        # 使用正则表达式去除数字索引
            cleaned_feature = re.sub(r'\.\d+\.', '.', feature)
            cleaned_features.add(cleaned_feature)
        return cleaned_features

    """
    脱敏特征分析报告
    """
    def analyze_masked_features(self, df:pd.DataFrame):
        print("=" * 60)
        print("脱敏特征分析报告")
        print("=" * 60)
        process_data = self.data_process(df)
        masked_features_set = set()
        if process_data:
            for item in process_data[:100]:# 只扫描前100条记录，通常足够覆盖所有脱敏特征
                for key, value in item.items():
                    if value is not None and self.is_masked_value(value):
                        masked_features_set.add(key)
            cleaned_masked_features = self.clean_feature_name(masked_features_set)
            self.masked_features = list(cleaned_masked_features)
            print(f"脱敏特征(共{len(self.masked_features)}个) :{self.masked_features}")
        print()


    """
    过滤特征分析报告
    """
    def analyze_filtered_features(self, df_filtered: pd.DataFrame, df_original: pd.DataFrame):
        print("=" * 60)
        print("过滤特征分析报告")
        print("=" * 60)
        
        # 处理多条记录来获取完整特征集
        process_data = self.data_process(df_original)
        
        if not process_data:
            print("错误: 无法处理原始数据")
            return
        
        # 方法1: 扫描多条记录收集所有可能的特征
        all_features_set = set()
        for i, record in enumerate(process_data):
            if i >= 100:  # 扫描100条记录，通常足够覆盖所有特征
                break
            all_features_set.update(record.keys())
        
        cleaned_original_features = self.clean_feature_name(all_features_set)
        print(f"原始特征(共{len(cleaned_original_features)}个): {sorted(list(cleaned_original_features))}")
        print()
        
        filtered_features = set(df_filtered.columns)
        cleaned_filtered_features = self.clean_feature_name(filtered_features)
        
        filtered_features_set = cleaned_original_features - cleaned_filtered_features
        added_features = cleaned_filtered_features - cleaned_original_features
        maintained_features = cleaned_original_features & cleaned_filtered_features
            
        # 更新实例变量  
        self.filtered_features = list(filtered_features_set)
        print(f"保留特征(共{len(maintained_features)}个): {list(maintained_features)}")
        print()
        print(f"新增特征(共{len(added_features)}个): {list(added_features)}")
        print()
        print(f"过滤特征(共{len(self.filtered_features)}个): {self.filtered_features}")
        print()

    """
    每个特征进行分布分析
    """
    def analyze_features(self, df: pd.DataFrame):

        analysis_results = []

        for column in df.columns:
            # 基本统计信息
            non_null_count = df[column].count()
            null_count = df[column].isnull().sum()
            total_count = len(df)
            null_ratio = null_count / total_count
            unique_count = df[column].nunique()
            
            # 数据类型
            dtype = df[column].dtype
            
            # 最常见值及其频率
            if non_null_count > 0:
                value_counts = df[column].value_counts()
                most_common_value = value_counts.index[0] if len(value_counts) > 0 else None
                most_common_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
                most_common_ratio = most_common_freq / total_count if most_common_freq > 0 else 0
            else:
                most_common_value = None
                most_common_freq = 0
                most_common_ratio = 0
        
            # 数值型特征的额外统计
            numeric_stats = {}
            if pd.api.types.is_numeric_dtype(df[column]):
                numeric_stats = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'median': df[column].median()
                }
            
            # 文本型特征的额外统计
            text_stats = {}
            if pd.api.types.is_string_dtype(df[column]):
                # 平均长度（排除空值）
                non_null_values = df[column].dropna()
                if len(non_null_values) > 0:
                    avg_length = non_null_values.str.len().mean()
                    max_length = non_null_values.str.len().max()
                    min_length = non_null_values.str.len().min()
                    text_stats = {
                        'avg_length': avg_length,
                        'max_length': max_length,
                        'min_length': min_length
                    }
            
            # 收集分析结果
            result = {
                'feature_name': column,
                'data_type': str(dtype),
                'total_count': total_count,
                'non_null_count': non_null_count,
                'null_count': null_count,
                'null_ratio': round(null_ratio, 4),
                'unique_count': unique_count,
                'most_common_value': str(most_common_value) if most_common_value is not None else None,
                'most_common_freq': most_common_freq,
                'most_common_ratio': round(most_common_ratio, 4),
                'unique_ratio': round(unique_count / total_count, 4) if total_count > 0 else 0
            }
            
            # 添加数值统计
            result.update(numeric_stats)
            # 添加文本统计
            result.update(text_stats)
            
            analysis_results.append(result)
    
        # 转换为DataFrame并排序
        analysis_df = pd.DataFrame(analysis_results)
        
        # 按特征名排序
        analysis_df = analysis_df.sort_values('feature_name')
        
        return analysis_df
    

    """
    打印详细的特征分析报告
    """
    def print_detailed_analysis(self, df: pd.DataFrame):
        
        analysis_df = self.analyze_features(df)
        
        print("=" * 80)
        print("特征分析报告")
        print("=" * 80)
        
        for _, row in analysis_df.iterrows():
            print(f"\n特征: {row['feature_name']}")
            print(f"  数据类型: {row['data_type']}")
            print(f"  总记录数: {row['total_count']}")
            print(f"  非空数量: {row['non_null_count']}")
            print(f"  空值数量: {row['null_count']}")
            print(f"  空值比例: {row['null_ratio']:.2%}")
            print(f"  唯一值数量: {row['unique_count']}")
            print(f"  唯一值比例: {row['unique_ratio']:.2%}")
            
            if pd.notna(row['most_common_value']):
                print(f"  最常见值: {row['most_common_value']}")
                print(f"  最常见值频率: {row['most_common_freq']}")
                print(f"  最常见值比例: {row['most_common_ratio']:.2%}")
            
            # 数值型特征的额外信息
            if 'mean' in row and pd.notna(row['mean']):
                print(f"  平均值: {row['mean']:.2f}")
                print(f"  标准差: {row['std']:.2f}")
                print(f"  最小值: {row['min']:.2f}")
                print(f"  最大值: {row['max']:.2f}")
                print(f"  中位数: {row['median']:.2f}")
            
            # 文本型特征的额外信息
            if 'avg_length' in row and pd.notna(row['avg_length']):
                print(f"  平均长度: {row['avg_length']:.2f}")
                print(f"  最大长度: {row['max_length']}")
                print(f"  最小长度: {row['min_length']}")
        
        return analysis_df  


    """
    部分筛选条件
    """
    def get_rules_by_partner_code(self, partner_code:str):
        if partner_code == "AWJ_CODE":
            return {
                'city': {'鸡西市', '黑河市', '鹤岗市', '大兴安岭地区', '防城港市', '河池市', 
                        '桂林市', '来宾市', '梧州市', '安阳市', '周口市', '许昌市', '鹤壁市', 
                        '濮阳市市', '新乡市', '黄石市', '鄂州市', '十堰市', '莆田市', '三明市', 
                        '漳州市', '唐山市'},
                'province': {'宁夏', '香港', '澳门', '新疆', '西藏', '内蒙古', '台湾', '青海', 
                            '广西', '贵州', '吉林'},
                'company_keywords': {'公安局', '警察', '法院', '军队', '检察院', '城市管理局', '律师', 
                                    '记者', '贷款', '金融', '执行局', '监狱', '交通警察', '派出所', 
                                    '刑事侦查部门', '交警', '刑侦'}
            }
        elif partner_code == "HXH_CODE":
            return {}
        elif partner_code == "NWD_CODE":
            return {
                'province': {'新疆', '西藏', '香港', '澳门', '台湾'}
            }
        elif partner_code == "LXJ_CODE":
            return {
                'province': {'香港', '澳门', '台湾', '新疆', '西藏'},
                'city': {'自贡', '北海', '南宁', '宁德', '莆田', '泉州', '盐城'}
            }
        elif partner_code == "XYF_CODE":
            return {}
        elif partner_code == "RONG_CODE":
            return {
                'province': {'香港', '澳门', '台湾', '新疆', '西藏'},
                'company_keywords': {'学校', '公检法'}
            }
        elif partner_code == "XY_CODE":
            return {}
        elif partner_code == "JY_CODE":
            return {
                'province': {'新疆', '西藏'},
                'city': {'萍乡市'}
            }
        elif partner_code == "FLXD_CODE":
            return {
                'province': {'香港', '澳门', '台湾', '新疆', '西藏'}
            }
        elif partner_code == "FQY_CODE":
            return {
                'province': {'香港', '澳门', '台湾', '新疆', '西藏'}
            }
        elif partner_code == "BBS_CODE":
            return {
                'province': {'香港', '澳门', '台湾', '新疆', '西藏', '内蒙古'},
                'city': {'宁德市', '安溪县', '电白县', '余干县', '双峰县', '儋州市', '龙岩市', '宾阳县', '丰宁县'}
            }
        elif partner_code == "TMDP_CODE":
            return {
                'province': {'香港', '澳门', '台湾', '新疆', '西藏', '青海'},
             
            }
        elif partner_code == "HH_CODE":
            return {
                'province': {'新疆', '西藏'},
        
            }
        else:
            return None


    def data_compliance_check(self, df: pd.DataFrame, location_mode='standard'):
        """
        条件筛选分析报告
        
        参数:
        df: 数据框
        location_mode: 地理位置筛选模式
            - 'standard': 使用 province/city 字段
            - 'gps': 使用 deviceInfo.applyPos字段  
            - 'both': 双重筛选，两种地理位置都需要满足条件
        """
        print("=" * 80)
        print(f"条件筛选报告 - 地理位置模式: {location_mode}")
        print("=" * 80)
        
        # 根据模式确定地理位置字段
        if location_mode == 'standard':
            province_field = 'province'
            city_field = 'city'
        elif location_mode == 'gps':
            province_field = 'deviceInfo.applyPos'
            city_field = 'deviceInfo.applyPos'
        elif location_mode == 'both':
            province_field = 'province'
            city_field = 'city'
            gps_province_field = 'deviceInfo.applyPos'
            gps_city_field = 'deviceInfo.applyPos'
        else:
            raise ValueError("location_mode 必须是 'standard', 'gps' 或 'both'")
        
        # 定义需要检查的必需字段
        required_fields = [
            'partner_code', 
            province_field, 
            city_field,
            'companyInfo.companyName', 
            'idInfo.birthDate',
            'idInfo.validityDate',
            'degree'
        ]
        
        # 如果是双重筛选模式，添加GPS字段到必需字段检查
        if location_mode == 'both':
            required_fields.extend([gps_province_field, gps_city_field])
        
        for index, row in df.iterrows():
            # 提前检查所有必需字段是否为空
            missing_fields = []
            for field in required_fields:
                value = row.get(field)
                if pd.isna(value) or value == '' or value is None:
                    missing_fields.append(field)
            
            # 如果有字段为空，直接记录为无效数据
            if missing_fields:
                print(f"行 {index}: 缺失字段 {missing_fields}")
                self.missing_data.append(row)
                continue

            partner_code = row.get('partner_code')
            rules = self.get_rules_by_partner_code(partner_code)
            
            # 获取地理位置信息
            if location_mode == 'both':
                province = row.get(province_field)
                city = row.get(city_field)
                gps_province = row.get(gps_province_field)
                gps_city = row.get(gps_city_field)
            else:
                province = row.get(province_field)
                city = row.get(city_field)
            
            company = row.get('companyInfo.companyName')
            age = row.get('idInfo.birthDate')
            date = row.get('idInfo.validityDate')
            degree = row.get('degree')

            if partner_code == "AWJ_CODE":
                if age < 22 or age > 49:
                    self.invalid_data.append(row)
                    continue

                # 地理位置检查
                location_valid = True
                if location_mode == 'both':
                    # 检查标准地理位置
                    standard_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    standard_valid = standard_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                    
                    # 检查GPS地理位置
                    gps_valid = not any(keyword in str(gps_province) for keyword in rules.get('province', []))
                    gps_valid = gps_valid and not any(keyword in str(gps_city) for keyword in rules.get('city', []))
                    
                    location_valid = standard_valid and gps_valid
                else:
                    location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    continue
                
                # 检查公司名称是否包含禁止关键词
                if company and any(keyword in str(company) for keyword in rules.get('company_keywords', [])):
                    self.invalid_data.append(row)
                    continue
                    
            elif partner_code == "HXH_CODE":
                if age < 22 or age > 55 or not date:
                    self.invalid_data.append(row)
                    continue

            elif partner_code == "NWD_CODE":
                if age < 23 or age > 55 or not date or date < 7:
                    self.invalid_data.append(row)
                    continue

                # 地理位置检查
                location_valid = True
                if location_mode == 'both':
                    standard_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    standard_valid = standard_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                    
                    gps_valid = not any(keyword in str(gps_province) for keyword in rules.get('province', []))
                    gps_valid = gps_valid and not any(keyword in str(gps_city) for keyword in rules.get('city', []))
                    
                    location_valid = standard_valid and gps_valid
                else:
                    location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    continue
                    
            elif partner_code == "LXJ_CODE":
                if age < 23 or age > 50 or not date or date < 30:
                    self.invalid_data.append(row)
                    continue

                # 地理位置检查
                location_valid = True
                if location_mode == 'both':
                    standard_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    standard_valid = standard_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                    
                    gps_valid = not any(keyword in str(gps_province) for keyword in rules.get('province', []))
                    gps_valid = gps_valid and not any(keyword in str(gps_city) for keyword in rules.get('city', []))
                    
                    location_valid = standard_valid and gps_valid
                else:
                    location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    continue

            elif partner_code == "XYF_CODE":
                if age < 23 or age > 55 or not date or date < 90 or degree == 'JUNIOR':
                    self.invalid_data.append(row)
                    continue

            elif partner_code == "RONG_CODE":
                if age < 22 or age > 50:
                    self.invalid_data.append(row)
                    continue
                
                # 地理位置检查
                location_valid = True
                if location_mode == 'both':
                    standard_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    standard_valid = standard_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                    
                    gps_valid = not any(keyword in str(gps_province) for keyword in rules.get('province', []))
                    gps_valid = gps_valid and not any(keyword in str(gps_city) for keyword in rules.get('city', []))
                    
                    location_valid = standard_valid and gps_valid
                else:
                    location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    continue
                
                # 检查公司名称是否包含禁止关键词
                if company and any(keyword in str(company) for keyword in rules.get('company_keywords', [])):
                    self.invalid_data.append(row)
                    continue

            elif partner_code == "XY_CODE":
                if not date or date < 30:
                    self.invalid_data.append(row)
                    continue
                    
            elif partner_code == "JY_CODE":
                if age < 23 or age > 50:
                    self.invalid_data.append(row)
                    continue

                # 地理位置检查
                location_valid = True
                if location_mode == 'both':
                    standard_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    standard_valid = standard_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                    
                    gps_valid = not any(keyword in str(gps_province) for keyword in rules.get('province', []))
                    gps_valid = gps_valid and not any(keyword in str(gps_city) for keyword in rules.get('city', []))
                    
                    location_valid = standard_valid and gps_valid
                else:
                    location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    continue

            elif partner_code == "FLXD_CODE":
                if age < 22 or age > 55 or not date or date < 90:
                    self.invalid_data.append(row)
                    continue

                # 地理位置检查
                location_valid = True
                if location_mode == 'both':
                    standard_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    standard_valid = standard_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                    
                    gps_valid = not any(keyword in str(gps_province) for keyword in rules.get('province', []))
                    gps_valid = gps_valid and not any(keyword in str(gps_city) for keyword in rules.get('city', []))
                    
                    location_valid = standard_valid and gps_valid
                else:
                    location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    continue

            elif partner_code == "FQY_CODE":
                if age < 22 or age >= 55:
                    self.invalid_data.append(row)
                    continue

                # 地理位置检查
                location_valid = True
                if location_mode == 'both':
                    standard_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    standard_valid = standard_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                    
                    gps_valid = not any(keyword in str(gps_province) for keyword in rules.get('province', []))
                    gps_valid = gps_valid and not any(keyword in str(gps_city) for keyword in rules.get('city', []))
                    
                    location_valid = standard_valid and gps_valid
                else:
                    location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    continue

            elif partner_code == "BBS_CODE":
                if age < 22 or age > 55 or not date or date < 30:
                    self.invalid_data.append(row)
                    continue

                # 地理位置检查
                location_valid = True
                if location_mode == 'both':
                    standard_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    standard_valid = standard_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                    
                    gps_valid = not any(keyword in str(gps_province) for keyword in rules.get('province', []))
                    gps_valid = gps_valid and not any(keyword in str(gps_city) for keyword in rules.get('city', []))
                    
                    location_valid = standard_valid and gps_valid
                else:
                    location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    continue

            elif partner_code == "TMDP_CODE":
                if age < 22 or age > 48:
                    self.invalid_data.append(row)
                    continue

                # 地理位置检查
                location_valid = True
                if location_mode == 'both':
                    standard_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    standard_valid = standard_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                    
                    gps_valid = not any(keyword in str(gps_province) for keyword in rules.get('province', []))
                    gps_valid = gps_valid and not any(keyword in str(gps_city) for keyword in rules.get('city', []))
                    
                    location_valid = standard_valid and gps_valid
                else:
                    location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    continue
                        
            elif partner_code == "HH_CODE":
                if age < 22:
                    self.invalid_data.append(row)
                    continue

                # 地理位置检查
                location_valid = True
                if location_mode == 'both':
                    standard_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    standard_valid = standard_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                    
                    gps_valid = not any(keyword in str(gps_province) for keyword in rules.get('province', []))
                    gps_valid = gps_valid and not any(keyword in str(gps_city) for keyword in rules.get('city', []))
                    
                    location_valid = standard_valid and gps_valid
                else:
                    location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                    location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    continue
                    
        print(len(df), "条数据中:")
        print("筛选未通过数据条数:", len(self.invalid_data)) 
        print("缺失数据条数:", len(self.missing_data))
        print("未通过比例: {:.2f}%".format((len(self.invalid_data) + len(self.missing_data)) / len(df) * 100))
        print()

    """
    合作方分布以及特征分析报告
    """
    def analyze_data(self):
        df_filtered = pd.read_csv(self.data_file) # 实际使用的数据
        df_original = pd.read_csv(self.original_data_file, sep='\|\|', engine='python') # 原始数据，包含被过滤掉的特征
        df_unused = pd.read_csv(self.unused_data_file) # 未使用的数据版本，使用了applyPos字段

        # 脱敏特征统计
        self.analyze_masked_features(df_original)

        # 特征筛选统计
        self.analyze_filtered_features(df_filtered, df_original)

        # 合作方分布与通过率分析
        self.calculate_distribution(df_filtered)

        # 各特征详细分析
        self.print_detailed_analysis(df_filtered)

        # 数据符合验证
        self.data_compliance_check(df_unused,'both')




def main():
    check = DataCheck('train/all_data.csv', 'data/2025-09-20.txt', 'data/all_data.csv')
    check.analyze_data()

if __name__ == "__main__":
    main()
