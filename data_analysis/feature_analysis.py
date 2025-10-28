import pandas as pd
import re
from typing import Dict
import os
import numpy as np
import pandas as pd
from typing import Dict, List
import numpy as np


class FeatureAnalysis:
        
    def __init__(self, data_file: str="processed"):
        """
        初始化数据分析器
        
        """
        self.data_file = data_file #实际使用的数据


    def analyze_feature_impact(self, df: pd.DataFrame, time_series_data: Dict[str, pd.DataFrame]) -> None:
        """
        特征通过率分析 - 排除YXM_CODE，统计空值
        """
        print("开始特征分析...")
        
        # 创建输出目录
        output_dir = "data/feature_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 排除合作方YXM_CODE
        df = df[df['partner_code'] != 'YXM_CODE'].copy()
        print(f"排除YXM_CODE后数据集形状: {df.shape}")
        
        # 按预定义顺序定义特征
        categorical_features = [
            'amount', 'bankCardInfo.bankCode', 
            'companyInfo.industry', 'companyInfo.occupation', 'customerSource', 'degree', 'maritalStatus',
            'idInfo.gender', 'idInfo.nation', 'income', 'jobFunctions', 'purpose', 'resideFunctions', 'term', 
            'deviceInfo.osType', 'deviceInfo.isCrossDomain','applyPos.province','applyPos.city','companyInfo.companyName','province','city'
        ]# 太长了，暂时排除 
        
        numeric_features = [
            'idInfo.birthDate', 'idInfo.validityDate', 'pictureInfo.0.faceScore'
        ]
        
        # 处理地理位置信息
        if 'deviceInfo.applyPos' in df.columns:
            print("提取地理位置信息...")
            df = self._extract_location_info(df)
        
        # 筛选实际存在的特征
        categorical_features = [f for f in categorical_features if f in df.columns]
        numeric_features = [f for f in numeric_features if f in df.columns]
        
        print(f"类别型特征: {len(categorical_features)} 个")
        print(f"数值型特征: {len(numeric_features)} 个")
        
        all_results = []
        
        # 分析总体数据
        print("分析总体数据...")
        for feature in categorical_features:
            results = self._analyze_categorical_feature(df, feature, '总体')
            all_results.extend(results)
        
        for feature in numeric_features:
            results = self._analyze_numeric_feature(df, feature, '总体')
            all_results.extend(results)
        
        # 特殊处理：联系人关系组合分析
        print("分析联系人关系组合...")
        relationship_results = self._analyze_relationship_combination(df, '总体')
        all_results.extend(relationship_results)
        
        full_data = df.copy()
        # 按合作方分析
        print("按合作方分析...")
        partners = [p for p in df['partner_code'].unique() if p != 'YXM_CODE']
        
        for partner in partners:
            partner_data = df[df['partner_code'] == partner]
            # if len(partner_data) < 50:  # 样本量要求
            #     continue
                
            for feature in categorical_features:
                results = self._analyze_categorical_feature(partner_data, feature, f'合作方_{partner}')
                all_results.extend(results)
            
            for feature in numeric_features:
                results = self._analyze_numeric_feature(partner_data, feature, f'合作方_{partner}')
                all_results.extend(results)
            
            # 合作方的联系人关系组合分析
            partner_relationship_results = self._analyze_relationship_combination(partner_data, f'合作方_{partner}')
            all_results.extend(partner_relationship_results)
        
        # 输出结果
        self._write_analysis_report(all_results, full_data, output_dir, categorical_features, numeric_features)
        print(f"分析完成！结果保存在: {output_dir}")


    def _analyze_relationship_combination(self, data: pd.DataFrame, scope: str) -> List[dict]:
        """
        分析联系人关系组合特征
        将 linkmanList.0.relationship 和 linkmanList.1.relationship 合并分析
        """
        results = []
        
        # 检查必要的列是否存在
        if 'linkmanList.0.relationship' not in data.columns or 'linkmanList.1.relationship' not in data.columns:
            return results
        
        # 创建关系组合
        def create_relationship_combination(row):
            rel1 = row['linkmanList.0.relationship']
            rel2 = row['linkmanList.1.relationship']
            
            # 处理空值
            if pd.isna(rel1) and pd.isna(rel2):
                return '双空'
            elif pd.isna(rel1):
                return f'空+{rel2}'
            elif pd.isna(rel2):
                return f'{rel1}+空'
            else:
                return f'{rel1}+{rel2}'
        
        # 应用组合创建
        data = data.copy()
        data['relationship_combination'] = data.apply(create_relationship_combination, axis=1)
        
        # 统计各组合的通过率
        combination_stats = data.groupby('relationship_combination').agg({
            'label': ['count', 'mean', 'sum']
        }).round(4)
        
        # 转换为结果格式
        for combination in combination_stats.index:
            stats = combination_stats.loc[combination]
            
            results.append({
                'scope': scope,
                'feature': 'linkmanList.relationship',
                'type': 'categorical',
                'value': str(combination),
                'pass_rate': stats[('label', 'mean')],
                'sample_count': int(stats[('label', 'count')]),
                'pass_count': int(stats[('label', 'sum')])
            })
        
        # 按通过率排序
        results.sort(key=lambda x: x['pass_rate'], reverse=True)
        
        # # 单独分析第一个联系人的关系（保持原有分析）
        # rel1_results = self._analyze_single_relationship(data, 'linkmanList.0.relationship', scope, 'linkmanList.0.relationship')
        # results.extend(rel1_results)
        
        # # 单独分析第二个联系人的关系（保持原有分析）
        # rel2_results = self._analyze_single_relationship(data, 'linkmanList.1.relationship', scope, 'linkmanList.1.relationship')
        # results.extend(rel2_results)
        
        return results
  
    def _analyze_single_relationship(self, data: pd.DataFrame, feature: str, scope: str, relationship_type: str) -> List[dict]:
        """
        分析单个联系人关系特征
        """
        results = []
        
        if feature not in data.columns:
            return results
        
        # 统计各关系的通过率
        relationship_stats = data.groupby(feature).agg({
            'label': ['count', 'mean', 'sum']
        }).round(4)
        
        # 处理空值
        null_data = data[data[feature].isna()]
        if len(null_data) > 0:
            results.append({
                'scope': scope,
                'feature': f'{relationship_type}',
                'type': 'categorical',
                'value': '空值',
                'pass_rate': null_data['label'].mean(),
                'sample_count': len(null_data),
                'pass_count': null_data['label'].sum()
            })
        
        # 转换为结果格式
        for relationship in relationship_stats.index:
            if pd.isna(relationship):
                continue
                
            stats = relationship_stats.loc[relationship]
            
            results.append({
                'scope': scope,
                'feature': f'{relationship_type}',
                'type': 'categorical',
                'value': str(relationship),
                'pass_rate': stats[('label', 'mean')],
                'sample_count': int(stats[('label', 'count')]),
                'pass_count': int(stats[('label', 'sum')])
            })
        
        return results


    def _analyze_categorical_feature(self, data: pd.DataFrame, feature: str, scope: str) -> List[dict]:
        """分析类别型特征 - 包含空值统计"""
        results = []
        
        # 确保包含空值
        feature_data = data[feature].fillna('None')  # 将NaN转换为'None'
        
        # 修复警告：添加 observed=True
        grouped = data.groupby(feature_data, observed=True).agg({
            'label': ['count', 'mean', 'sum']
        }).round(4)
        
        for value, stats in grouped.iterrows():
            # 保持原始值，包括'None'
            value_str = str(value)
            
            results.append({
                'scope': scope,
                'feature': feature,
                'type': 'categorical',
                'value': value_str,
                'pass_rate': stats[('label', 'mean')],
                'sample_count': int(stats[('label', 'count')]),
                'pass_count': int(stats[('label', 'sum')])
            })
        
        # 改进排序：先按通过率降序，通过率相同时按样本数降序
        results.sort(key=lambda x: (x['pass_rate'], x['sample_count']), reverse=True)
        return results

    def _analyze_numeric_feature(self, data: pd.DataFrame, feature: str, scope: str) -> List[dict]:
        """分析数值型特征"""
        results = []
        total_samples = len(data)
        
        # 用于跟踪已处理的样本索引
        processed_indices = set()
        
        # 1. 统计空值
        null_data = data[data[feature].isna()]
        if len(null_data) > 0:
            results.append({
                'scope': scope,
                'feature': feature,
                'type': 'numeric',
                'value': 'None',
                'pass_rate': null_data['label'].mean(),
                'sample_count': len(null_data),
                'pass_count': null_data['label'].sum()
            })
            processed_indices.update(null_data.index)
        
        # 分析非空数值数据
        non_null_data = data[data[feature].notna()]
        feature_data = non_null_data[feature]
        
        if len(feature_data) == 0:
            # 验证：如果只有空值，检查总数
            if len(processed_indices) != total_samples:
                missing_count = total_samples - len(processed_indices)
                print(f"错误: 特征 {feature} 样本数不匹配，缺失 {missing_count} 个样本")
            return results
        
        # 2. 统计无限值
        infinite_mask = np.isinf(feature_data)
        infinite_data = non_null_data[infinite_mask]
        if len(infinite_data) > 0:
            results.append({
                'scope': scope,
                'feature': feature,
                'type': 'numeric',
                'value': '无限值',
                'pass_rate': infinite_data['label'].mean(),
                'sample_count': len(infinite_data),
                'pass_count': infinite_data['label'].sum()
            })
            processed_indices.update(infinite_data.index)
        
        # 3. 处理有限数值数据
        finite_mask = ~infinite_mask
        finite_data = non_null_data[finite_mask]
        finite_feature_data = finite_data[feature]
        
        if len(finite_feature_data) == 0:
            # 验证当前总数
            current_total = len(processed_indices)
            if current_total != total_samples:
                print(f"错误: 特征 {feature} 样本数不匹配: {current_total} != {total_samples}")
            return results
        
        try:
            min_val = finite_feature_data.min()
            max_val = finite_feature_data.max()
            
            # 4. 智能分箱
            if feature == 'idInfo.birthDate':
                # 处理异常年龄值（负值或过大值）
                abnormal_age_mask = (finite_feature_data < 0) | (finite_feature_data > 120)
                abnormal_age_indices = finite_data[abnormal_age_mask].index
                
                if len(abnormal_age_indices) > 0:
                    abnormal_age_data = data.loc[abnormal_age_indices]
                    abnormal_values = abnormal_age_data[feature].unique()
                    results.append({
                        'scope': scope,
                        'feature': feature,
                        'type': 'numeric',
                        'value': f'异常年龄值({",".join([str(x) for x in abnormal_values])})',
                        'pass_rate': abnormal_age_data['label'].mean(),
                        'sample_count': len(abnormal_age_data),
                        'pass_count': abnormal_age_data['label'].sum()
                    })
                    processed_indices.update(abnormal_age_indices)
                
                # 正常年龄分箱（排除异常值）
                normal_age_mask = ~abnormal_age_mask
                normal_age_data = finite_data[normal_age_mask]
                normal_age_feature = normal_age_data[feature]
                
                if len(normal_age_feature) > 0:
                    age_bins = [0, 18, 25, 30, 35, 40, 45, 50, 55, 60, 70, 100]
                    age_labels = ['未成年(<18)', '18-25岁', '25-30岁', '30-35岁', '35-40岁', '40-45岁', '45-50岁', '50-55岁', '55-60岁', '60-70岁', '70岁以上']
                    bins = pd.cut(normal_age_feature, bins=age_bins, labels=age_labels, right=False)
                    current_finite_data = normal_age_data
                else:
                    bins = pd.Series([], dtype=object)
                    current_finite_data = normal_age_data
                    
            elif feature == 'idInfo.validityDate':
                # 处理异常有效期值（负值）
                abnormal_validity_mask = finite_feature_data < 0
                abnormal_validity_indices = finite_data[abnormal_validity_mask].index
                
                if len(abnormal_validity_indices) > 0:
                    abnormal_validity_data = data.loc[abnormal_validity_indices]
                    abnormal_values = abnormal_validity_data[feature].unique()
                    results.append({
                        'scope': scope,
                        'feature': feature,
                        'type': 'numeric',
                        'value': f'异常有效期({",".join([str(x) for x in abnormal_values])})',
                        'pass_rate': abnormal_validity_data['label'].mean(),
                        'sample_count': len(abnormal_validity_data),
                        'pass_count': abnormal_validity_data['label'].sum()
                    })
                    processed_indices.update(abnormal_validity_indices)
                
                # 正常有效期分箱（排除异常值）
                normal_validity_mask = ~abnormal_validity_mask
                normal_validity_data = finite_data[normal_validity_mask]
                normal_validity_feature = normal_validity_data[feature]
                
                if len(normal_validity_feature) > 0:
                    if max_val <= 365:
                        validity_bins = [0, 1, 30, 90, 180, 365]
                        validity_labels = ['当天', '1-30天', '1-3个月', '3-6个月', '6-12个月']
                    elif max_val <= 3650:
                        validity_bins = [0, 30, 90, 365, 730, 1825, 3650]
                        validity_labels = ['1个月内', '1-3个月', '3-12个月', '1-2年', '2-5年', '5-10年']
                    else:
                        validity_bins = [0, 365, 1825, 3650, 7300, 18250, float('inf')]
                        validity_labels = ['1年以内', '1-5年', '5-10年', '10-20年', '20-50年', '50年以上']
                    
                    bins = pd.cut(normal_validity_feature, bins=validity_bins, labels=validity_labels, right=False)
                    current_finite_data = normal_validity_data
                else:
                    bins = pd.Series([], dtype=object)
                    current_finite_data = normal_validity_data
                    
            elif feature == 'pictureInfo.0.faceScore':
                # 人脸分数已经包含了异常值处理
                score_bins = [-np.inf, 0, 50, 60, 70, 75, 80, 85, 90, 95, 100, np.inf]
                score_labels = ['小于0（异常）', '0-50', '50-60', '60-70', '70-75', 
                            '75-80', '80-85', '85-90', '90-95', '95-100', '大于100（异常）']
                bins = pd.cut(finite_feature_data, bins=score_bins, labels=score_labels)
                current_finite_data = finite_data
                
            else:
                # 对于其他数值特征，也检查异常值
                abnormal_mask = np.isnan(finite_feature_data)  # 或其他异常检测逻辑
                normal_mask = ~abnormal_mask
                normal_data = finite_data[normal_mask]
                normal_feature = normal_data[feature]
                
                if len(normal_feature) > 0:
                    n_bins = min(8, max(4, len(normal_feature) // 100))
                    try:
                        bins = pd.qcut(normal_feature, q=n_bins, duplicates='drop')
                    except:
                        bins = pd.cut(normal_feature, bins=n_bins)
                    current_finite_data = normal_data
                else:
                    bins = pd.Series([], dtype=object)
                    current_finite_data = normal_data
            
            # 5. 分组统计 - 确保不重复（只在有分箱数据时执行）
            if len(bins) > 0:
                grouped = current_finite_data.groupby(bins, observed=True).agg({
                    'label': ['count', 'mean', 'sum']
                }).round(4)
                
                # 记录分箱样本索引
                bin_indices = set()
                for bin_range in grouped.index:
                    bin_mask = (bins == bin_range)
                    bin_data = current_finite_data[bin_mask]
                    bin_indices.update(bin_data.index)
                    
                    results.append({
                        'scope': scope,
                        'feature': feature,
                        'type': 'numeric',
                        'value': str(bin_range),
                        'pass_rate': grouped.loc[bin_range, ('label', 'mean')],
                        'sample_count': len(bin_data),
                        'pass_count': int(grouped.loc[bin_range, ('label', 'sum')])
                    })
                
                processed_indices.update(bin_indices)
            
            # 6. 检查是否有分箱过程中遗漏的样本
            expected_finite_indices = set(finite_data.index)
            actual_processed_finite = processed_indices.intersection(expected_finite_indices)
            missing_finite_indices = expected_finite_indices - actual_processed_finite
            
            if len(missing_finite_indices) > 0:
                missing_finite_data = data.loc[list(missing_finite_indices)]
                missing_values = missing_finite_data[feature].unique()
                print(f"分箱警告: 特征 {feature} 在分箱过程中遗漏 {len(missing_finite_indices)} 个样本")
                print(f"遗漏样本的特征值: {missing_values}")
                
                results.append({
                    'scope': scope,
                    'feature': feature,
                    'type': 'numeric',
                    'value': f'分箱遗漏值({",".join([str(x) for x in missing_values])})',
                    'pass_rate': missing_finite_data['label'].mean(),
                    'sample_count': len(missing_finite_data),
                    'pass_count': missing_finite_data['label'].sum()
                })
                processed_indices.update(missing_finite_indices)
            
            # 7. 最终验证
            final_processed = len(processed_indices)
            if final_processed != total_samples:
                missing_count = total_samples - final_processed
                print(f"严重错误: 特征 {feature} 丢失 {missing_count} 个样本")
                print(f"  预期: {total_samples}, 实际: {final_processed}")
                
                # 找出丢失的样本
                missing_indices = set(data.index) - processed_indices
                if missing_indices:
                    missing_data = data.loc[list(missing_indices)]
                    missing_values = missing_data[feature].unique()
                    print(f"  丢失样本的特征值: {missing_values}")
                    
                    results.append({
                        'scope': scope,
                        'feature': feature,
                        'type': 'numeric',
                        'value': f'最终丢失值({",".join([str(x) for x in missing_values])})',
                        'pass_rate': missing_data['label'].mean(),
                        'sample_count': len(missing_data),
                        'pass_count': missing_data['label'].sum()
                    })
            else:
                print(f"{scope}——特征 {feature} 样本统计正确: {final_processed}/{total_samples}")
            
            # 排序
            results.sort(key=lambda x: (x['pass_rate'], x['sample_count']), reverse=True)
            
        except Exception as e:
            print(f"{scope}——特征 {feature} 分箱失败: {e}")
            
            # 失败时统计所有有限数据
            results.append({
                'scope': scope,
                'feature': feature,
                'type': 'numeric',
                'value': f'有限数据[{min_val:.1f}-{max_val:.1f}]',
                'pass_rate': finite_feature_data.mean(),
                'sample_count': len(finite_feature_data),
                'pass_count': finite_feature_data.sum()
            })
            processed_indices.update(finite_data.index)
        
        return results


    def _extract_location_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取省市区信息"""
        df = df.copy()
        
        # 用于统计未知地址
        unknown_addresses = []
        
        def extract_location(address):
            if pd.isna(address) or not isinstance(address, str):
                unknown_addresses.append(f"空值或非字符串: {address}")
                return '未知', '未知', '未知'
            
            original_address = address.strip()
            address = original_address.replace(' ', '').replace('　', '')  # 去除空格
            
            province = '未知'
            city = '未知'
            district = '未知'
            remaining = address
            
            # 1. 第一步：提取省份/直辖市
            # 直辖市
            direct_cities = ['北京市', '上海市', '天津市', '重庆市']
            for city_name in direct_cities:
                if remaining.startswith(city_name):
                    province = city_name
                    city = city_name  # 直辖市城市名与省份相同
                    remaining = remaining.replace(city_name, '', 1)
                    break
            
            # 自治区
            if province == '未知':
                autonomous_regions = [
                    '新疆维吾尔自治区', '西藏自治区', '广西壮族自治区', 
                    '宁夏回族自治区', '内蒙古自治区'
                ]
                for region in autonomous_regions:
                    if remaining.startswith(region):
                        province = region
                        remaining = remaining.replace(region, '', 1)
                        break
            
            # 特别行政区
            if province == '未知':
                special_regions = ['香港特别行政区', '澳门特别行政区', '香港', '澳门']
                for region in special_regions:
                    if remaining.startswith(region):
                        if region in ['香港', '澳门']:
                            province = f'{region}特别行政区'
                        else:
                            province = region
                        city = province  # 特别行政区城市名与省份相同
                        remaining = remaining.replace(region, '', 1)
                        break
            
            # 省
            if province == '未知':
                match = re.search(r'^(\S+?省)', remaining)
                if match:
                    province = match.group(1)
                    remaining = remaining.replace(province, '', 1)
            
            # 2. 第二步：提取城市（在剩余字符串中从左到右匹配）
            if province != '未知' and city == '未知':
                # 先尝试匹配自治州、地区、盟（这些是地级行政单位）
                prefectural_units = ['自治州', '地区', '盟']
                for unit in prefectural_units:
                    # 查找第一个出现的自治州/地区/盟
                    pattern = f'([^省区]+?{unit})'
                    match = re.search(pattern, remaining)
                    if match:
                        city_candidate = match.group(1)
                        # 验证：不能是空字符串，且长度合理
                        if len(city_candidate) > 2 and city_candidate != province:
                            city = city_candidate
                            remaining = remaining.replace(city, '', 1)
                            break
                
                # 如果没有找到自治州等地级单位，再找市
                if city == '未知':
                    # 查找第一个出现的市
                    pattern = r'([^省区县]+?市)'
                    match = re.search(pattern, remaining)
                    if match:
                        city_candidate = match.group(1)
                        # 验证：不能是空字符串，且长度合理
                        if len(city_candidate) > 2 and city_candidate != province:
                            city = city_candidate
                            remaining = remaining.replace(city, '', 1)
            
            # 3. 第三步：处理省直管县的情况
            # 如果没有找到地级市，但找到了县级单位，且省份不是直辖市
            if (province != '未知' and city == '未知' and 
                province not in direct_cities + ['香港特别行政区', '澳门特别行政区']):
                
                # 查找县级单位（县、自治县、县级市、区）
                county_units = ['自治县', '县', '市', '区']
                for unit in county_units:
                    pattern = f'^([^{unit}]+?{unit})'
                    match = re.search(pattern, remaining)
                    if match:
                        county_candidate = match.group(1)
                        if len(county_candidate) > 1 and county_candidate != province:
                            city = county_candidate  # 县级单位作为城市
                            district = county_candidate  # 同时作为区县
                            remaining = remaining.replace(city, '', 1)
                            break
            
            # 4. 第四步：提取区县（如果城市不是县级单位）
            if (district == '未知' and city != '未知' and city != province and
                not any(unit in city for unit in ['自治县', '县', '区'])):  # 如果城市不是县级单位
                
                # 在剩余字符串中查找区县级单位
                district_units = ['区', '县', '自治县', '市', '旗']
                for unit in district_units:
                    pattern = f'([^{unit}]+?{unit})'
                    match = re.search(pattern, remaining)
                    if match:
                        district_candidate = match.group(1)
                        if (len(district_candidate) > 1 and 
                            district_candidate != city and 
                            district_candidate != province):
                            district = district_candidate
                            break
            
            # 5. 第五步：最终清理和验证
            # 清理结果中的多余字符
            for field in [province, city, district]:
                if field != '未知':
                    field = field.strip('市县区 ')
            
            # 记录未知的地址用于调试
            if province == '未知' or city == '未知':
                unknown_addresses.append(f"地址: '{original_address}' -> 省份: {province}, 城市: {city}, 区县: {district}, 剩余: '{remaining}'")
            
            return province, city, district
        
        # 应用提取函数
        location_data = df['deviceInfo.applyPos'].apply(extract_location)
        df['applyPos.province'] = location_data.apply(lambda x: x[0])
        df['applyPos.city'] = location_data.apply(lambda x: x[1])
        # df['applyPos.district'] = location_data.apply(lambda x: x[2])
        
        # 打印未知地址统计
        if unknown_addresses:
            print(f"\n=== 地址提取未知情况统计 ===")
            print(f"总共发现 {len(unknown_addresses)} 个未知地址")
            print("前20个未知地址示例:")
            for i, addr in enumerate(unknown_addresses[:20]):
                print(f"  {i+1}. {addr}")
            
            province_unknown = sum(1 for addr in unknown_addresses if "省份: 未知" in addr)
            city_unknown = sum(1 for addr in unknown_addresses if "城市: 未知" in addr)
            print(f"\n未知分布: 省份未知 {province_unknown} 个, 城市未知 {city_unknown} 个")
            
            if len(unknown_addresses) > 20:
                unknown_file = "data/unknown_addresses.txt"
                os.makedirs(os.path.dirname(unknown_file), exist_ok=True)
                with open(unknown_file, "w", encoding="utf-8") as f:
                    f.write("未知地址列表\n")
                    f.write("=" * 50 + "\n")
                    for addr in unknown_addresses:
                        f.write(addr + "\n")
                print(f"完整未知地址列表已保存到: {unknown_file}")
        else:
            print("所有地址都成功提取了省市区信息！")
        
        # 打印提取结果统计
        province_counts = df['applyPos.province'].value_counts()
        city_counts = df['applyPos.city'].value_counts()
        
        print(f"\n=== 地址提取结果统计 ===")
        print(f"提取到 {len(province_counts)} 个省份")
        print(f"提取到 {len(city_counts)} 个城市")
        print(f"前10个省份:")
        for province, count in province_counts.head(10).items():
            print(f"  {province}: {count}")
        print(f"前10个城市:")
        for city, count in city_counts.head(10).items():
            print(f"  {city}: {count}")

        return df


    def _write_analysis_report(self, results: List[dict], df: pd.DataFrame, output_dir: str, 
                                categorical_features: List[str], numeric_features: List[str]):
        """生成分析报告 - 按预定义顺序，包含特征影响程度分析"""
        overall_pass_rate = df['label'].mean()
        
        # 定义特殊特征（不在原始特征列表中）
        special_features = ['linkmanList.relationship', 'linkmanList.0.relationship', 'linkmanList.1.relationship']
        
        # 计算特征影响程度（加权方差）
        def calculate_feature_impact(results, scope):
            """计算特征对通过率的影响程度"""
            scope_results = [item for item in results if item['scope'] == scope]
            feature_impacts = {}
            
            for feature in set(item['feature'] for item in scope_results):
                feature_results = [item for item in scope_results if item['feature'] == feature]
                
                # 排除特殊值
                # valid_results = [
                #     item for item in feature_results 
                #     if item['value'] not in ['None']
                # ]
                valid_results = feature_results
                
                # if len(valid_results) < 2:  # 至少需要2个分箱才能计算方差
                #     continue
                
                # 提取通过率和样本数
                pass_rates = []
                sample_counts = []
                
                for item in valid_results:
                    pass_rates.append(item['pass_rate'])
                    sample_counts.append(item['sample_count'])
                
                # 计算加权方差
                if sum(sample_counts) > 0:
                    try:
                        weighted_mean = np.average(pass_rates, weights=sample_counts)
                        weighted_variance = np.average(
                            [(pr - weighted_mean) ** 2 for pr in pass_rates], 
                            weights=sample_counts
                        )
                        
                        # 使用样本数加权的方差作为影响程度指标
                        total_samples = sum(sample_counts)
                        impact_score = weighted_variance * total_samples
                        
                        feature_impacts[feature] = {
                            'weighted_variance': weighted_variance,
                            'impact_score': impact_score,
                            'bins_count': len(valid_results),
                            'total_samples': total_samples
                        }
                    except Exception as e:
                        print(f"计算特征 {feature} 影响程度时出错: {e}")
                        continue
            
            return feature_impacts
        
        # 计算总体特征影响程度
        overall_impacts = calculate_feature_impact(results, '总体')
        
        # 总体报告
        with open(os.path.join(output_dir, "总体特征分析报告.txt"), "w", encoding="utf-8") as f:
            f.write("特征通过率分析报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"总体通过率: {overall_pass_rate:.3f}\n")
            f.write(f"总样本数: {len(df)} (已排除YXM_CODE)\n")
            f.write("=" * 60 + "\n\n")
            
            # 输出特征影响程度排序
            f.write("特征影响程度排序（基于加权方差）:\n")
            f.write("[影响分数 = 加权方差 × 总样本数]\n")
    
            f.write("-" * 50 + "\n")
            
            if overall_impacts:
                sorted_features = sorted(
                    overall_impacts.items(), 
                    key=lambda x: x[1]['impact_score'], 
                    reverse=True
                )
                
                for i, (feature, impact_info) in enumerate(sorted_features, 1):
                    f.write(f"{i:2d}. {feature:<25} ")
                    f.write(f"影响分数: {impact_info['impact_score']:8.2f} ")
                    f.write(f"方差: {impact_info['weighted_variance']:.4f} ")
                    f.write(f"分箱数: {impact_info['bins_count']:2d} ")
                    f.write(f"样本数: {impact_info['total_samples']:5d}\n")
            else:
                f.write("无法计算特征影响程度\n")
            f.write("\n")
            
            # 总体范围的结果
            scope_results = [item for item in results if item['scope'] == '总体']
            
            f.write("详细特征分析结果:\n")
            f.write("-" * 50 + "\n")
            
            # 按预定义顺序输出特征（包括常规特征和特殊特征）
            all_features = categorical_features + numeric_features + special_features
            
            for feature in all_features:
                feature_results = [item for item in scope_results if item['feature'] == feature]
                if not feature_results:  # 如果该特征没有结果，跳过
                    continue
                
                # 显示特征的影响程度信息
                impact_info = overall_impacts.get(feature, {})
                impact_score = impact_info.get('impact_score', 'N/A')
                variance = impact_info.get('weighted_variance', 'N/A')
                
                f.write(f"\n【{feature}】 - 影响分数: {impact_score}, 方差: {variance}\n")
                for i, item in enumerate(feature_results, 1):
                    if item['type'] == 'numeric':
                        f.write(f"  {i:2d}. {item['value']:<40} ")
                    else:
                        f.write(f"  {i:2d}. {item['value']:<40} ")
                    
                    f.write(f"通过率: {item['pass_rate']:6.3f} ")
                    f.write(f"样本数: {item['sample_count']:4d}")
                    if item['value'] == 'None':
                        f.write(" (空值)")
                    f.write("\n")
            
            # 计算实际分析的特征数量
            analyzed_features = len([f for f in all_features if any(item['feature'] == f for item in scope_results)])
            f.write(f"\n共计分析 {analyzed_features} 个特征\n")
        
        # 各合作方单独报告
        partner_scopes = sorted(set(item['scope'] for item in results if item['scope'].startswith('合作方_')))
        
        for scope in partner_scopes:
            partner_name = scope.replace('合作方_', '')
            scope_results = [item for item in results if item['scope'] == scope]
            
            # 计算合作方特征影响程度
            partner_impacts = calculate_feature_impact(results, scope)
            
            with open(os.path.join(output_dir, f"合作方_{partner_name}_分析报告.txt"), "w", encoding="utf-8") as f:
                f.write(f"合作方 {partner_name} 特征分析报告\n")
                f.write("=" * 60 + "\n")
                
                partner_data = df[df['partner_code'] == partner_name]
                partner_pass_rate = partner_data['label'].mean()
                
                f.write(f"合作方通过率: {partner_pass_rate:.3f}\n")
                f.write(f"样本数量: {len(partner_data)}\n")
                f.write("=" * 60 + "\n\n")
                
                # 输出合作方特征影响程度排序
                f.write(f"合作方 {partner_name} 特征影响程度排序:\n")
                f.write("[影响分数 = 加权方差 × 总样本数]\n")
                f.write("-" * 50 + "\n")
                
                if partner_impacts:
                    sorted_features = sorted(
                        partner_impacts.items(), 
                        key=lambda x: x[1]['impact_score'], 
                        reverse=True
                    )
                    
                    for i, (feature, impact_info) in enumerate(sorted_features, 1):
                        f.write(f"{i:2d}. {feature:<25} ")
                        f.write(f"影响分数: {impact_info['impact_score']:8.2f} ")
                        f.write(f"方差: {impact_info['weighted_variance']:.4f} ")
                        f.write(f"分箱数: {impact_info['bins_count']:2d} ")
                        f.write(f"样本数: {impact_info['total_samples']:5d}\n")
                else:
                    f.write("无法计算特征影响程度\n")
                f.write("\n")
                
                # 按预定义顺序输出特征（包括特殊特征）
                all_features = categorical_features + numeric_features + special_features
                
                for feature in all_features:
                    feature_results = [item for item in scope_results if item['feature'] == feature]
                    if not feature_results:
                        continue
                    
                    # 显示特征的影响程度信息
                    impact_info = partner_impacts.get(feature, {})
                    impact_score = impact_info.get('impact_score', 'N/A')
                    variance = impact_info.get('weighted_variance', 'N/A')
                    
                    f.write(f"【{feature}】 - 影响分数: {impact_score}, 方差: {variance}\n")
                    for i, item in enumerate(feature_results, 1):
                        if item['type'] == 'numeric':
                            f.write(f"  {i:2d}. {item['value']:<40} ")
                        else:
                            f.write(f"  {i:2d}. {item['value']:<40} ")
                        
                        f.write(f"通过率: {item['pass_rate']:6.3f} ")
                        f.write(f"样本数: {item['sample_count']:4d}")
                        if item['value'] == 'None':
                            f.write(" (空值)")
                        f.write("\n")
                    f.write("\n")
                
                # 合作方报告也显示分析的特征数量
                analyzed_features = len([f for f in all_features if any(item['feature'] == f for item in scope_results)])
                f.write(f"共计分析 {analyzed_features} 个特征\n")


    def _load_all_time_data(self, data_file: str, start_date: str = "2025-10-01", end_date: str = "2025-10-21"):
        """加载所有时间文件夹下的数据，限定日期范围
        
        参数:
        - data_file: 数据文件或目录路径
        - start_date: 开始日期 (包含)
        - end_date: 结束日期 (包含)
        """
        all_data_list = []
        time_series_data = {}  # 用于存储时间序列数据
        
        if os.path.isfile(data_file):
            # 如果是单个文件，检查日期是否在日期范围内
            file_date = os.path.basename(data_file).split('.')[0]
            if start_date <= file_date <= end_date:
                df = pd.read_csv(data_file)
                all_data_list.append(df)
                time_series_data[file_date] = df
                print(f"成功加载文件: {file_date}, 形状: {df.shape}")
            else:
                print(f"跳过文件 {file_date}，不在日期范围 {start_date} 到 {end_date}")
                
        elif os.path.isdir(data_file):
            # 遍历所有时间文件夹
            date_folders = [f for f in os.listdir(data_file) 
                        if os.path.isdir(os.path.join(data_file, f))]
            
            for date_folder in sorted(date_folders):
                # 检查日期是否在日期范围内
                if date_folder < start_date:
                    print(f"跳过文件夹 {date_folder}，早于开始日期 {start_date}")
                    continue
                elif date_folder > end_date:
                    print(f"跳过文件夹 {date_folder}，超过结束日期 {end_date}")
                    continue
                    
                date_path = os.path.join(data_file, date_folder)
                all_data_file = os.path.join(date_path, "all_data.csv")
                
                if os.path.exists(all_data_file):
                    try:
                        df = pd.read_csv(all_data_file)
                        all_data_list.append(df)
                        time_series_data[date_folder] = df
                        print(f"成功加载: {date_folder}/all_data.csv, 形状: {df.shape}")
                    except Exception as e:
                        print(f"加载文件 {all_data_file} 时出错: {e}")
        
        # 合并所有数据
        if all_data_list:
            all_data = pd.concat(all_data_list, ignore_index=True)
            print(f"总共加载 {len(all_data_list)} 个文件，合并后形状: {all_data.shape}")
            print(f"日期范围: {start_date} 到 {end_date}")
            return all_data, time_series_data
        else:
            print("未找到符合日期要求的数据文件")
            return pd.DataFrame(), {}


    """
    各特征通过率分析报告
    """
    def analyze_data(self):

        all_data, time_series_data = self._load_all_time_data(
            data_file=self.data_file,start_date="2025-10-01",end_date="2025-10-21") # 加载处理好的数据
        self.analyze_feature_impact(all_data, time_series_data)



def main():
    analysis = FeatureAnalysis()
    analysis.analyze_data()

if __name__ == "__main__":
    main()

  