import pandas as pd
import json
import re
from typing import Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from datetime import datetime
import numpy as np


class DataCheck:
        
    def __init__(self, data_file: str="processed", original_data_file: str="data/2025-09-20.txt", unused_data_file: str="data_for_analysis"):
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
    
    def remove_parent_fields(self,field_set):
        """
        从字段集合中删除父级字段
        如果存在 A.B，则删除 A
        """
        # 创建结果的副本
        result = set(field_set)
        
        # 找出所有父级字段
        parent_fields = set()
        for field in field_set:
            # 如果字段包含点，说明有父级字段
            if '.' in field:
                # 获取父级字段（第一个点之前的部分）
                parent_field = field.split('.')[0]
                parent_fields.add(parent_field)
        
        # 从结果中移除所有父级字段
        result -= parent_fields
        
        return result


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
    def calculate_distribution(self):
        print("=" * 60)
        print("合作方分布与通过率分析报告")
        print("=" * 60)

        # 1. 读取所有时间文件夹下的数据
        all_data, time_series_data = self._load_all_time_data(self.data_file)
        
        if all_data.empty:
            print("未找到任何数据文件")
            return

        # 2. 基础统计信息
        print(f"数据总记录数: {len(all_data)} 笔")
        print(f"总体通过率: {(all_data['label'].sum() / len(all_data) * 100):.2f}%")
        print()

        # 3. 合作方分布
        partner_counts = all_data['partner_code'].value_counts().sort_values(ascending=False)
        print("合作方分布:")
        for partner, count in partner_counts.items():
            print(f"{partner}: {count} 笔 ({count / len(all_data) * 100:.2f}%)")
        print()

        # 4. 各合作方通过率分析
        print("各合作方通过率分析:")
        for partner, count in partner_counts.items():
            partner_df = all_data[all_data['partner_code'] == partner]
            pass_rate = partner_df['label'].mean()
            print(f"{partner}: {count} 笔, 通过 {partner_df['label'].sum()} 笔, 未通过 {count-partner_df['label'].sum()} 笔, 通过率 {pass_rate * 100:.2f}%")
        print()

        # 5. 绘制图表
        self._plot_partner_trends(time_series_data)
   

    def _plot_partner_trends(self, time_series_data):
        """使用 matplotlib 的交互式功能"""

        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        if not time_series_data:
            print("没有时间序列数据可绘制图表")
            return
        
        # 准备数据
        dates = sorted(time_series_data.keys())
        partners = set()
        
        for date, df in time_series_data.items():
            partners.update(df['partner_code'].unique())
        
        partners = sorted(list(partners))
        
        # 准备数据
        partner_sample_data = {}
        partner_pass_rate_data = {}
        
        for partner in partners:
            sample_counts = []
            pass_rates = []
            for date in dates:
                df_date = time_series_data[date]
                partner_df = df_date[df_date['partner_code'] == partner]
                sample_counts.append(len(partner_df))
                if len(partner_df) > 0:
                    pass_rates.append(partner_df['label'].mean() * 100)
                else:
                    pass_rates.append(0)
            partner_sample_data[partner] = sample_counts
            partner_pass_rate_data[partner] = pass_rates
        
        # 计算总样本数和总体通过率
        total_samples = []
        total_pass_rates = []
        active_partners_count = []
        
        for date in dates:
            df_date = time_series_data[date]
            total_samples.append(len(df_date))
            if len(df_date) > 0:
                total_pass_rates.append(df_date['label'].mean() * 100)
            else:
                total_pass_rates.append(0)
            active_partners_count.append(len(df_date['partner_code'].unique()))

        # 第一张图：样本数量趋势
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        
        lines1 = []
        annotations1 = []  # 存储标注对象
        colors = plt.cm.tab20(np.linspace(0, 1, len(partners)))
        
        # 绘制各合作方样本数
        for i, partner in enumerate(partners):
            line1, = ax1.plot(dates, partner_sample_data[partner], marker='o', 
                            color=colors[i], linewidth=2, markersize=6, label=partner)
            lines1.append(line1)
            
            # 标注具体数值
            partner_annotations = []
            for j, (date, count) in enumerate(zip(dates, partner_sample_data[partner])):
                if count > 0:
                    ann = ax1.annotate(f'{count}', (date, count), 
                                    textcoords="offset points", xytext=(0,8), 
                                    ha='center', fontsize=8, alpha=0.7, color=colors[i])
                    partner_annotations.append(ann)
            annotations1.append(partner_annotations)
        
        # 绘制总样本数
        total_line1, = ax1.plot(dates, total_samples, marker='s', linewidth=3, 
                            color='black', linestyle='--', markersize=8, label='总样本数')
        
        # 标注总样本数值
        total_annotations1 = []
        for j, (date, total) in enumerate(zip(dates, total_samples)):
            ann = ax1.annotate(f'{total}', (date, total), 
                            textcoords="offset points", xytext=(0,12), 
                            ha='center', fontsize=10, fontweight='bold', color='red')
            total_annotations1.append(ann)
        
        ax1.set_title('各合作方样本数随时间变化', fontsize=14, fontweight='bold')
        ax1.set_ylabel('样本数量', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # 创建图例
        legend1 = ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                            frameon=True, fancybox=True, shadow=True)
        
        # 使图例可交互 - 修复版本
        def on_pick1(event):
            # 这个事件在点击图例项时触发
            legline = event.artist
            origline = linedict1[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
            
            # 设置图例项的透明度
            if visible:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.3)
            
            # 处理数值标注
            if origline == total_line1:
                for ann in total_annotations1:
                    ann.set_visible(visible)
            else:
                index = lines1.index(origline)
                for ann in annotations1[index]:
                    ann.set_visible(visible)
            
            fig1.canvas.draw()
        
        # 映射图例项到原始线条
        linedict1 = {}
        for legline, origline in zip(legend1.get_lines(), lines1 + [total_line1]):
            legline.set_picker(5)  # 5 points tolerance
            linedict1[legline] = origline
        
        fig1.canvas.mpl_connect('pick_event', on_pick1)
        
        plt.tight_layout()
        plt.show()
        
        # 第二张图：通过率趋势
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        
        lines2 = []
        annotations2 = []  # 存储标注对象
        
        # 绘制各合作方通过率
        for i, partner in enumerate(partners):
            line2, = ax2.plot(dates, partner_pass_rate_data[partner], marker='s', 
                            color=colors[i], linewidth=2, markersize=6, label=partner)
            lines2.append(line2)
            
            # 标注具体数值
            partner_annotations = []
            for j, (date, rate) in enumerate(zip(dates, partner_pass_rate_data[partner])):
                if rate > 0:
                    ann = ax2.annotate(f'{rate:.1f}%', (date, rate), 
                                    textcoords="offset points", xytext=(0,8), 
                                    ha='center', fontsize=8, alpha=0.7, color=colors[i])
                    partner_annotations.append(ann)
            annotations2.append(partner_annotations)
        
        # 绘制总体通过率
        total_line2, = ax2.plot(dates, total_pass_rates, marker='D', linewidth=3, 
                            color='black', linestyle='--', markersize=8, label='总体通过率')
        
        # 标注总体通过率数值
        total_annotations2 = []
        for j, (date, rate) in enumerate(zip(dates, total_pass_rates)):
            ann = ax2.annotate(f'{rate:.1f}%', (date, rate), 
                            textcoords="offset points", xytext=(0,12), 
                            ha='center', fontsize=10, fontweight='bold', color='red')
            total_annotations2.append(ann)
        
        ax2.set_title('各合作方通过率随时间变化', fontsize=14, fontweight='bold')
        ax2.set_ylabel('通过率 (%)', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(bottom=0)
        
        # 创建图例
        legend2 = ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                            frameon=True, fancybox=True, shadow=True)
        
        # 使图例可交互
        def on_pick2(event):
            legline = event.artist
            origline = linedict2[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
            
            if visible:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.3)
            
            # 处理数值标注
            if origline == total_line2:
                for ann in total_annotations2:
                    ann.set_visible(visible)
            else:
                index = lines2.index(origline)
                for ann in annotations2[index]:
                    ann.set_visible(visible)
            
            fig2.canvas.draw()
        
        # 映射图例项到原始线条
        linedict2 = {}
        for legline, origline in zip(legend2.get_lines(), lines2 + [total_line2]):
            legline.set_picker(5)  # 5 points tolerance
            linedict2[legline] = origline
        
        fig2.canvas.mpl_connect('pick_event', on_pick2)
        
        plt.tight_layout()
        plt.show()
        
        # 第三张图：活跃合作方数量
        fig3, ax3 = plt.subplots(figsize=(12, 6))

        # 绘制活跃合作方数量
        line3, = ax3.plot(dates, active_partners_count, marker='^', linewidth=3, 
                        color='green', markersize=8, label='活跃合作方数量')

        # 标注具体数值
        annotations3 = []
        for j, (date, count) in enumerate(zip(dates, active_partners_count)):
            ann = ax3.annotate(f'{count}家', (date, count), 
                            textcoords="offset points", xytext=(0,10), 
                            ha='center', fontsize=10, fontweight='bold', color='blue')
            annotations3.append(ann)

        ax3.set_title('活跃合作方数量随时间变化', fontsize=14, fontweight='bold')
        ax3.set_ylabel('合作方数量', fontsize=12)
        ax3.set_xlabel('日期', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # 创建普通图例（第三张图不需要交互）
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

        # 打印输出各阶段活跃合作方信息
        print("\n" + "="*60)
        print("各阶段活跃合作方详情")
        print("="*60)

        for i, date in enumerate(dates):
            df_date = time_series_data[date]
            active_partners = sorted(df_date['partner_code'].unique())
            count = active_partners_count[i]
            
            print(f"{date}: {count}家活跃合作方")
            print(", ".join(active_partners))
            

        print("="*60)
        
        print("使用说明：")
        print("1. 点击右侧图例中的线条或文字可以显示/隐藏对应的折线")
        print("2. 隐藏的图例会变灰显示")
        print("3. 折线上的数值会随折线一起显示或隐藏")
      
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
            cleaned_masked_features = self.remove_parent_fields(cleaned_masked_features)
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
        cleaned_original_features = self.remove_parent_fields(cleaned_original_features)
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


    def data_compliance_check(self, df: pd.DataFrame):
        """
        条件筛选分析报告
        
        参数:
        df: 数据框
        """
        print("=" * 80)
        print("条件筛选报告 - 地理位置模式: GPS")
        print("=" * 80)

        # 计算被排除的条目数量
        excluded_count = len(df[df['partner_code'] == 'YXM_CODE'])
        
        # 过滤掉 partner_code 为 YXM_CODE 的数据
        df_filtered = df[df['partner_code'] != 'YXM_CODE']

        print(f"总共排除 {excluded_count} 条 partner_code 为 YXM_CODE 的数据")
        print(f"剩余 {len(df_filtered)} 条数据进行处理")
        
        # 使用GPS地理位置字段
        province_field = 'deviceInfo.applyPos'
        city_field = 'deviceInfo.applyPos'
        
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
        
        # 初始化统计字典
        failure_stats = {}
        
        def add_failure_stat(partner_code, failure_type, reason=""):
            """添加失败统计"""
            key = f"{partner_code}_{failure_type}"
            if key not in failure_stats:
                failure_stats[key] = {
                    'partner_code': partner_code,
                    'failure_type': failure_type,
                    'reason': reason,
                    'count': 0
                }
            failure_stats[key]['count'] += 1
        
        for index, row in df_filtered.iterrows():
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
                add_failure_stat(row.get('partner_code'), '缺失字段', f"缺失字段: {missing_fields}")
                continue

            partner_code = row.get('partner_code')
            rules = self.get_rules_by_partner_code(partner_code)
            
            # 获取GPS地理位置信息
            province = row.get(province_field)
            city = row.get(city_field)
            
            company = row.get('companyInfo.companyName')
            age = row.get('idInfo.birthDate')
            date = row.get('idInfo.validityDate')
            degree = row.get('degree')

            if partner_code == "AWJ_CODE":
                if age < 22 or age > 49:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}不在22-49范围内")
                    continue

                # GPS地理位置检查
                location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '地理位置不符合', f"省份/城市包含禁止关键词")
                    continue
                
                # 检查公司名称是否包含禁止关键词
                if company and any(keyword in str(company) for keyword in rules.get('company_keywords', [])):
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '公司名称不符合', f"公司名称包含禁止关键词")
                    continue
                    
            elif partner_code == "HXH_CODE":
                if age < 22 or age > 55:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}不在22-55范围内")
                    continue
                elif not date:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期缺失', "有效期字段为空")
                    continue

            elif partner_code == "NWD_CODE":
                if age < 23 or age > 55:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}不在23-55范围内")
                    continue
                elif not date:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期缺失', "有效期字段为空")
                    continue
                elif date < 7:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期不足', f"有效期{date}天小于7天")
                    continue

                # GPS地理位置检查
                location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '地理位置不符合', f"省份/城市包含禁止关键词")
                    continue
                    
            elif partner_code == "LXJ_CODE":
                if age < 23 or age > 50:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}不在23-50范围内")
                    continue
                elif not date:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期缺失', "有效期字段为空")
                    continue
                elif date < 30:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期不足', f"有效期{date}天小于30天")
                    continue

                # GPS地理位置检查
                location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '地理位置不符合', f"省份/城市包含禁止关键词")
                    continue

            elif partner_code == "XYF_CODE":
                if age < 23 or age > 55:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}不在23-55范围内")
                    continue
                elif not date:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期缺失', "有效期字段为空")
                    continue
                elif date < 90:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期不足', f"有效期{date}天小于90天")
                    continue
                elif degree == 'JUNIOR':
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '学历不符合', "学历为初中不符合要求")
                    continue

            elif partner_code == "RONG_CODE":
                if age < 22 or age > 50:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}不在22-50范围内")
                    continue
                
                # GPS地理位置检查
                location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '地理位置不符合', f"省份/城市包含禁止关键词")
                    continue
                
                # 检查公司名称是否包含禁止关键词
                if company and any(keyword in str(company) for keyword in rules.get('company_keywords', [])):
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '公司名称不符合', f"公司名称包含禁止关键词")
                    continue

            elif partner_code == "XY_CODE":
                if not date:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期缺失', "有效期字段为空")
                    continue
                elif date < 30:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期不足', f"有效期{date}天小于30天")
                    continue
                    
            elif partner_code == "JY_CODE":
                if age < 23 or age > 50:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}不在23-50范围内")
                    continue

                # GPS地理位置检查
                location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '地理位置不符合', f"省份/城市包含禁止关键词")
                    continue

            elif partner_code == "FLXD_CODE":
                if age < 22 or age > 55:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}不在22-55范围内")
                    continue
                elif not date:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期缺失', "有效期字段为空")
                    continue
                elif date < 90:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期不足', f"有效期{date}天小于90天")
                    continue

                # GPS地理位置检查
                location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '地理位置不符合', f"省份/城市包含禁止关键词")
                    continue

            elif partner_code == "FQY_CODE":
                if age < 22 or age >= 55:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}不在22-54范围内")
                    continue

                # GPS地理位置检查
                location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '地理位置不符合', f"省份/城市包含禁止关键词")
                    continue

            elif partner_code == "BBS_CODE":
                if age < 22 or age > 55:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}不在22-55范围内")
                    continue
                elif not date:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期缺失', "有效期字段为空")
                    continue
                elif date < 30:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '有效期不足', f"有效期{date}天小于30天")
                    continue

                # GPS地理位置检查
                location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '地理位置不符合', f"省份/城市包含禁止关键词")
                    continue

            elif partner_code == "TMDP_CODE":
                if age < 22 or age > 48:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}不在22-48范围内")
                    continue

                # GPS地理位置检查
                location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '地理位置不符合', f"省份/城市包含禁止关键词")
                    continue
                        
            elif partner_code == "HH_CODE":
                if age < 22:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '年龄不符合', f"年龄{age}小于22岁")
                    continue

                # GPS地理位置检查
                location_valid = not any(keyword in str(province) for keyword in rules.get('province', []))
                location_valid = location_valid and not any(keyword in str(city) for keyword in rules.get('city', []))
                
                if not location_valid:
                    self.invalid_data.append(row)
                    add_failure_stat(partner_code, '地理位置不符合', f"省份/城市包含禁止关键词")
                    continue
        
        # 输出总体统计
        print(f"\n{len(df_filtered)} 条数据中:")
        print("筛选未通过数据条数:", len(self.invalid_data)) 
        print("缺失数据条数:", len(self.missing_data))
        print("未通过比例: {:.2f}%".format((len(self.invalid_data) + len(self.missing_data)) / len(df_filtered) * 100))
        
        # 输出详细失败统计
        print("\n" + "=" * 80)
        print("详细失败类型统计")
        print("=" * 80)
        
        if failure_stats:
            # 按partner_code和失败类型排序
            sorted_stats = sorted(failure_stats.values(), 
                                key=lambda x: (x['partner_code'], x['count']), 
                                reverse=True)
            
            current_partner = None
            for stat in sorted_stats:
                if stat['partner_code'] != current_partner:
                    current_partner = stat['partner_code']
                    print(f"\n{current_partner}:")
                    print("-" * 40)
                
                print(f"  {stat['failure_type']}: {stat['count']} 条")
                if stat['reason']:
                    print(f"    原因: {stat['reason']}")
        else:
            print("没有失败数据")
        
        print()

    def _load_all_time_data(self, data_file: str):
        """加载所有时间文件夹下的数据"""
        all_data_list = []
        time_series_data = {}  # 用于存储时间序列数据
        
        if os.path.isfile(data_file):
            # 如果是单个文件，直接读取
            df = pd.read_csv(data_file)
            date_str = os.path.basename(data_file).split('.')[0]
            # df['date'] = date_str
            all_data_list.append(df)
            
            # 添加到时间序列数据
            time_series_data[date_str] = df
            
        elif os.path.isdir(data_file):
            # 遍历所有时间文件夹
            date_folders = [f for f in os.listdir(data_file) 
                        if os.path.isdir(os.path.join(data_file, f))]
            
            for date_folder in sorted(date_folders):
                date_path = os.path.join(data_file, date_folder)
                all_data_file = os.path.join(date_path, "all_data.csv")
                
                if os.path.exists(all_data_file):
                    try:
                        df = pd.read_csv(all_data_file)
                        # df['date'] = date_folder  # 添加日期列
                        all_data_list.append(df)
                        time_series_data[date_folder] = df
                        print(f"成功加载: {date_folder}/all_data.csv, 形状: {df.shape}")
                    except Exception as e:
                        print(f"加载文件 {all_data_file} 时出错: {e}")
        
        # 合并所有数据
        if all_data_list:
            all_data = pd.concat(all_data_list, ignore_index=True)
            print(f"总共加载 {len(all_data_list)} 个文件，合并后形状: {all_data.shape}")
            return all_data, time_series_data
        else:
            return pd.DataFrame(), {}

    """
    合作方分布以及特征分析报告
    """
    def analyze_data(self):

        # all_data, time_series_data = self._load_all_time_data(self.data_file)
        unused_data, unused_time_series_data = self._load_all_time_data(self.unused_data_file)
        # original_data = pd.read_csv(self.original_data_file, sep='\|\|', engine='python') # 原始数据，包含被过滤掉的特征
        # 脱敏特征统计
        # self.analyze_masked_features(original_data)

        # 特征筛选统计
        # self.analyze_filtered_features(all_data, original_data)

        # # 合作方分布与通过率分析
        # self.calculate_distribution()

        # # 各特征详细分析
        # self.print_detailed_analysis(all_data)

        # # 数据符合验证
        self.data_compliance_check(unused_data)



def main():
    check = DataCheck()
    check.analyze_data()

if __name__ == "__main__":
    main()
