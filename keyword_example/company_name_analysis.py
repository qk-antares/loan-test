"""
公司名称分词+词袋模型分析脚本
分析 companyInfo.companyName 中的关键词与贷款通过率的关系
"""

import pandas as pd
import jieba
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.stats as stats
from texttable import Texttable
import os
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

class CompanyNameAnalyzer:
    """公司名称分词+词袋分析器"""
    
    def __init__(self, csv_file_paths):
        """
        初始化分析器
        
        Args:
            csv_file_paths: CSV文件路径列表 或 单个CSV文件路径
        """
        # 确保csv_file_paths是列表
        if isinstance(csv_file_paths, str):
            self.csv_file_paths = [csv_file_paths]
        else:
            self.csv_file_paths = csv_file_paths
            
        self.df = None
        self.company_names = None
        self.labels = None
        self.segmented_names = None
        
    def load_data(self, exclude_partners=None):
        """
        加载数据
        
        Args:
            exclude_partners: 要排除的合作方代码列表，默认排除 ['YXM_CODE', 'LYX_CODE']
        """
        if exclude_partners is None:
            exclude_partners = ['YXM_CODE', 'LYX_CODE']
        
        print("正在加载数据...")
        
        # 存储所有数据框
        dataframes = []
        file_stats = []
        
        # 逐个读取文件
        for file_path in self.csv_file_paths:
            if os.path.exists(file_path):
                print(f"读取文件: {file_path}")
                df_temp = pd.read_csv(file_path)
                dataframes.append(df_temp)
                file_stats.append({
                    'file': os.path.basename(file_path),
                    'records': len(df_temp)
                })
            else:
                print(f"警告: 文件不存在 - {file_path}")
        
        if not dataframes:
            raise FileNotFoundError("没有找到任何有效的CSV文件")
        
        # 合并所有数据框
        self.df = pd.concat(dataframes, ignore_index=True)
        
        # 打印文件统计信息
        print(f"\n文件读取统计:")
        for stat in file_stats:
            print(f"  - {stat['file']}: {stat['records']} 条记录")
        
        # 记录原始数据量
        original_count = len(self.df)
        print(f"\n合并后总计: {original_count} 条记录")
        
        # 过滤掉指定的合作方数据
        if exclude_partners and 'partner_code' in self.df.columns:
            print(f"排除合作方: {exclude_partners}")
            # 统计被排除的数据量
            excluded_counts = {}
            for partner in exclude_partners:
                count = len(self.df[self.df['partner_code'] == partner])
                if count > 0:
                    excluded_counts[partner] = count
            
            if excluded_counts:
                print(f"排除的数据统计: {excluded_counts}")
            
            # 过滤数据
            self.df = self.df[~self.df['partner_code'].isin(exclude_partners)]
            
            filtered_count = len(self.df)
            print(f"过滤后保留 {filtered_count} 条记录 (原始: {original_count} 条, 排除: {original_count - filtered_count} 条)")
        else:
            print(f"未进行合作方过滤，保留所有 {original_count} 条记录")
        
        # 提取公司名称和标签
        self.company_names = self.df['companyInfo.companyName'].fillna('').astype(str)
        self.labels = self.df['label'].astype(int)
        
        print(f"数据加载完成，共 {len(self.df)} 条记录")
        print(f"通过率: {self.labels.mean():.2%}")
        
        # 打印一些统计信息
        non_empty_company_names = self.company_names[self.company_names != '']
        print(f"非空公司名称: {len(non_empty_company_names)} 条")
        print(f"唯一公司名称: {len(non_empty_company_names.unique())} 个")
    
    def segment_company_names(self):
        """对公司名称进行分词"""
        print("正在进行分词...")
        self.segmented_names = []
        
        for company_name in self.company_names:
            if company_name and company_name.strip():
                # 使用jieba进行分词
                words = jieba.cut(company_name)
                # 过滤单字符词
                filtered_words = [
                    word for word in words 
                    if len(word) >= 2  # 只保留2个字符以上的词
                    and not word.isdigit()  # 排除纯数字
                ]
                self.segmented_names.append(' '.join(filtered_words))
            else:
                self.segmented_names.append('')
        
        print("分词完成")
    
    def analyze_single_keywords(self, min_freq=2):
        """
        分析单个关键词的通过率
        
        Args:
            min_freq: 关键词最小出现频数阈值，只保留出现次数>=该阈值的关键词
            
        Returns:
            keyword_stats: 关键词统计信息
        """
        print(f"正在分析单个关键词通过率（只保留出现频数>={min_freq}的关键词）...")
        
        # 统计所有关键词的出现次数和通过次数
        keyword_counts = defaultdict(int)  # 关键词总出现次数
        keyword_pass_counts = defaultdict(int)  # 关键词通过次数
        
        for i, segmented_name in enumerate(self.segmented_names):
            if segmented_name and segmented_name.strip():
                # 获取该条记录的标签
                label = self.labels.iloc[i]
                
                # 分词结果已经是空格分隔的
                words = segmented_name.split()
                
                # 去重，一条记录中同一个词只计算一次
                unique_words = set(words)
                
                for word in unique_words:
                    if word and len(word) >= 2:  # 只考虑2个字符以上的词
                        keyword_counts[word] += 1
                        if label == 1:  # 如果通过了
                            keyword_pass_counts[word] += 1
        
        print(f"找到 {len(keyword_counts)} 个不重复的关键词")
        
        # 筛选出现频数>=阈值的关键词
        filtered_keywords = [(keyword, count) for keyword, count in keyword_counts.items() if count >= min_freq]
        
        print(f"保留出现频数>={min_freq}的关键词: {len(filtered_keywords)} 个")
        
        # 计算统计信息
        keyword_stats = []
        overall_pass_rate = self.labels.mean()
        
        for keyword, total_count in filtered_keywords:
            pass_count = keyword_pass_counts[keyword]
            pass_rate = pass_count / total_count if total_count > 0 else 0
            
            # 计算提升度 (lift)
            lift = pass_rate / overall_pass_rate if overall_pass_rate > 0 else 0
            
            # 1. 贝叶斯平滑通过率
            # 使用先验：alpha=整体通过次数, beta=整体未通过次数
            alpha = self.labels.sum()  # 总通过次数作为先验
            beta = len(self.labels) - alpha  # 总未通过次数作为先验
            bayesian_pass_rate = (pass_count + alpha * 0.1) / (total_count + (alpha + beta) * 0.1)
            
            # 2. 统计显著性检验 (二项检验)
            if total_count >= 5:  # 最小样本量要求
                # H0: 该关键词通过率 = 整体通过率，小于0.05认为差异显著
                # 使用新版本的 binomtest (SciPy >= 1.7.0)
                result = stats.binomtest(pass_count, total_count, overall_pass_rate, alternative='two-sided')
                p_value = result.pvalue
                is_significant = p_value < 0.05
            else:
                p_value = 1.0
                is_significant = False
            
            keyword_stats.append({
                'keyword': keyword, # 关键词
                'total_count': total_count, # 总出现次数
                'pass_count': pass_count,   # 通过次数
                'pass_rate': pass_rate,     # 通过率
                'bayesian_pass_rate': bayesian_pass_rate,   # 贝叶斯平滑通过率
                'lift': lift,              # 提升度（通过率与整体通过率的比值）
                'p_value': p_value,        # p值
                'is_significant': is_significant,   # 是否统计显著
                'frequency': total_count / len(self.df)  # 占总样本的比例
            })
        
        # 转换为DataFrame
        keyword_df = pd.DataFrame(keyword_stats)
        
        print("关键词分析完成")
        
        return keyword_df
    
    def analyze_keyword_pass_rates(self, feature_matrix, feature_names, top_k=30):
        """
        分析关键词通过率（使用词袋模型的方法，已废弃）
        
        Args:
            feature_matrix: 特征矩阵
            feature_names: 特征名称列表
            top_k: 展示前k个关键词
            
        Returns:
            keyword_stats: 关键词统计信息
        """
        print("正在分析关键词通过率...")
        
        keyword_stats = []
        
        for i, keyword in enumerate(feature_names):
            # 找到包含该关键词的记录
            has_keyword = feature_matrix[:, i] > 0
            
            if np.sum(has_keyword) == 0:
                continue
            
            # 计算统计信息
            total_count = np.sum(has_keyword)
            pass_count = np.sum(self.labels[has_keyword])
            pass_rate = pass_count / total_count if total_count > 0 else 0
            
            # 整体通过率作为基准
            overall_pass_rate = self.labels.mean()
            
            # 计算提升度 (lift)
            lift = pass_rate / overall_pass_rate if overall_pass_rate > 0 else 0
            
            keyword_stats.append({
                'keyword': keyword,
                'total_count': total_count,
                'pass_count': pass_count,
                'pass_rate': pass_rate,
                'lift': lift,
                'frequency': total_count / len(self.df)  # 频率
            })
        
        # 转换为DataFrame并排序
        keyword_df = pd.DataFrame(keyword_stats)
        keyword_df = keyword_df.sort_values('pass_rate', ascending=False)
        
        print("关键词分析完成")
        
        return keyword_df
    
    def print_keyword_analysis(self, keyword_df, top_k=30):
        """打印关键词分析结果，使用多种统计学方法调和通过率和样本量"""
        overall_pass_rate = self.labels.mean()
        
        print(f"\n{'='*90}")
        print(f"单个关键词通过率分析结果 (整体通过率: {overall_pass_rate:.2%})")
        print(f"{'='*90}")
        
        # 1. 按原始通过率排序
        print(f"\n1. 按原始通过率排序 (前{top_k}个):")
        table1 = Texttable()
        table1.set_deco(Texttable.HEADER | Texttable.VLINES | Texttable.BORDER)
        table1.set_cols_align(['c', 'l', 'c', 'c', 'c', 'c', 'c'])
        table1.set_cols_width([4, 12, 6, 6, 8, 8, 6])
        
        # 设置表头
        table1.header(['排名', '关键词', '样本数', '通过数', '通过率', '提升度', '显著性'])
        
        # 添加数据行
        df_by_rate = keyword_df.sort_values('pass_rate', ascending=False)
        for i, (idx, row) in enumerate(df_by_rate.head(top_k).iterrows()):
            sig_mark = "***" if row['is_significant'] else ""
            table1.add_row([
                i+1,
                row['keyword'],
                row['total_count'],
                row['pass_count'],
                f"{row['pass_rate']:.2%}",
                f"{row['lift']:.2f}",
                sig_mark
            ])
        
        print(table1.draw())
        
        # 2. 按贝叶斯平滑通过率排序
        print(f"\n2. 按贝叶斯平滑通过率排序 (前{top_k}个) - 解决小样本偏差:")
        table2 = Texttable()
        table2.set_deco(Texttable.HEADER | Texttable.VLINES | Texttable.BORDER)
        table2.set_cols_align(['c', 'l', 'c', 'c', 'c', 'c', 'c'])
        table2.set_cols_width([4, 12, 6, 8, 8, 8, 6])
        
        # 设置表头
        table2.header(['排名', '关键词', '样本数', '原始率', '平滑率', '提升度', '显著性'])
        
        # 添加数据行
        df_by_bayesian = keyword_df.sort_values('bayesian_pass_rate', ascending=False)
        for i, (idx, row) in enumerate(df_by_bayesian.head(top_k).iterrows()):
            sig_mark = "***" if row['is_significant'] else ""
            table2.add_row([
                i+1,
                row['keyword'],
                row['total_count'],
                f"{row['pass_rate']:.2%}",
                f"{row['bayesian_pass_rate']:.2%}",
                f"{row['lift']:.2f}",
                sig_mark
            ])
        
        print(table2.draw())
        
        # 3. 只显示统计显著的关键词
        significant_keywords = keyword_df[keyword_df['is_significant']].sort_values('pass_rate', ascending=False)
        print(f"\n3. 统计显著的关键词 (p<0.05, 共{len(significant_keywords)}个):")
        if len(significant_keywords) > 0:
            table3 = Texttable()
            table3.set_deco(Texttable.HEADER | Texttable.VLINES | Texttable.BORDER)
            table3.set_cols_align(['c', 'l', 'c', 'c', 'c', 'c'])
            table3.set_cols_width([4, 12, 6, 8, 10, 8])
            
            # 设置表头
            table3.header(['排名', '关键词', '样本数', '通过率', 'p值', '提升度'])
            
            # 添加数据行
            for i, (idx, row) in enumerate(significant_keywords.head(top_k).iterrows()):
                table3.add_row([
                    i+1,
                    row['keyword'],
                    row['total_count'],
                    f"{row['pass_rate']:.2%}",
                    f"{row['p_value']:.4f}",
                    f"{row['lift']:.2f}"
                ])
            
            print(table3.draw())
        else:
            print("  没有发现统计显著的关键词")
        
        # 统计摘要
        print(f"\n统计摘要:")
        print(f"- 总计关键词数量: {len(keyword_df)}")
        print(f"- 通过率 > 整体通过率的关键词: {len(keyword_df[keyword_df['pass_rate'] > overall_pass_rate])}")
        print(f"- 统计显著的关键词 (p<0.05): {len(significant_keywords)}")
        print(f"- 样本量≥30的关键词: {len(keyword_df[keyword_df['total_count'] >= 30])}")
        print(f"- 样本量≥100的关键词: {len(keyword_df[keyword_df['total_count'] >= 100])}")
        
        # 方法说明
        print(f"\n方法说明:")
        print("1. 原始通过率: 简单的通过次数/总次数")
        print("2. 贝叶斯平滑: 使用先验分布平滑小样本估计，减少极端值")
        print("3. 统计显著性: 二项检验，*** 表示p<0.05")
        
        print()  # 空行
    
    def plot_keyword_analysis(self, keyword_df, top_k=20):
        """绘制关键词分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 贝叶斯平滑通过率最高的显著关键词（它们的实际通过率）
        top_keywords = keyword_df[keyword_df['is_significant']].sort_values('bayesian_pass_rate', ascending=False).head(top_k)
        axes[0, 0].barh(range(len(top_keywords)), top_keywords['pass_rate'])
        axes[0, 0].set_yticks(range(len(top_keywords)))
        axes[0, 0].set_yticklabels(top_keywords['keyword'])
        axes[0, 0].set_xlabel('通过率')
        axes[0, 0].set_title(f'贝叶斯平滑通过率最高的 {top_k} 个显著关键词')
        axes[0, 0].invert_yaxis()
        
        # 2. 出现频率 vs 通过率散点图
        axes[0, 1].scatter(keyword_df['frequency'], keyword_df['pass_rate'], alpha=0.6)
        axes[0, 1].set_xlabel('出现频率')
        axes[0, 1].set_ylabel('通过率')
        axes[0, 1].set_title('关键词频率 vs 通过率')
        axes[0, 1].axhline(y=self.labels.mean(), color='r', linestyle='--', label='整体通过率')
        axes[0, 1].legend()
        
        # 3. 通过率分布直方图
        axes[1, 0].hist(keyword_df['pass_rate'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('通过率')
        axes[1, 0].set_ylabel('关键词数量')
        axes[1, 0].set_title('关键词通过率分布')
        axes[1, 0].axvline(x=self.labels.mean(), color='r', linestyle='--', label='整体通过率')
        axes[1, 0].legend()
        
        # 4. 提升度最高的显著关键词
        top_lift_keywords = keyword_df[keyword_df['is_significant']].nlargest(top_k, 'lift')
        axes[1, 1].barh(range(len(top_lift_keywords)), top_lift_keywords['lift'])
        axes[1, 1].set_yticks(range(len(top_lift_keywords)))
        axes[1, 1].set_yticklabels(top_lift_keywords['keyword'])
        axes[1, 1].set_xlabel('提升度 (Lift)')
        axes[1, 1].set_title(f'提升度最高的 {top_k} 个显著关键词')
        axes[1, 1].invert_yaxis()
        axes[1, 1].axvline(x=1.0, color='r', linestyle='--', label='无提升线')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('./company_name_keyword_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, keyword_df, filename='./keyword_analysis_results.csv'):
        """保存分析结果到CSV文件"""
        # 重新排序列，把重要的指标放在前面
        columns_order = [
            'keyword', 'total_count', 'pass_count', 'pass_rate', 
            'bayesian_pass_rate', 'lift', 'p_value', 'is_significant', 'frequency'
        ]
        
        # 按贝叶斯平滑通过率排序保存
        keyword_df_sorted = keyword_df.sort_values('bayesian_pass_rate', ascending=False)
        keyword_df_sorted[columns_order].to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"分析结果已保存到: {filename}")
        print(f"  - 按贝叶斯平滑通过率排序")
        print(f"  - 包含贝叶斯平滑和统计显著性等指标")
    
    def run_analysis(self, min_freq=2, top_k=30, exclude_partners=None):
        """
        运行完整分析流程
        
        Args:
            min_freq: 关键词最小出现频数阈值
            top_k: 显示前k个结果
            exclude_partners: 要排除的合作方代码列表，默认排除 ['YXM_CODE', 'LYX_CODE']
        """
        # 1. 加载数据
        self.load_data(exclude_partners=exclude_partners)
        
        # 2. 分词
        self.segment_company_names()
        
        # 3. 分析单个关键词通过率
        keyword_df = self.analyze_single_keywords(min_freq=min_freq)
        
        if keyword_df is None or keyword_df.empty:
            print("无法分析关键词，分析终止")
            return None
        
        # 4. 打印结果
        self.print_keyword_analysis(keyword_df, top_k)
        
        # 5. 绘制图表
        try:
            self.plot_keyword_analysis(keyword_df, top_k)
        except Exception as e:
            print(f"绘图失败: {e}")
        
        # 6. 保存结果
        self.save_results(keyword_df)
        
        return keyword_df


def generate_date_file_list(start_date, end_date, base_path_template):
    """
    生成指定日期范围内的文件路径列表
    
    Args:
        start_date: 开始日期 (格式: 'YYYY-MM-DD')
        end_date: 结束日期 (格式: 'YYYY-MM-DD')
        base_path_template: 文件路径模板，使用 {date} 作为占位符
        
    Returns:
        list: 文件路径列表
    """
    file_paths = []
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    current = start
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        file_path = base_path_template.format(date=date_str)
        file_paths.append(file_path)
        current += timedelta(days=1)
    
    return file_paths


def filter_existing_files(file_paths):
    """
    过滤出存在的文件
    
    Args:
        file_paths: 文件路径列表
        
    Returns:
        tuple: (存在的文件列表, 不存在的文件列表)
    """
    existing = []
    missing = []
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            existing.append(file_path)
        else:
            missing.append(file_path)
    
    return existing, missing


def main():
    """主函数"""
    # 方案1: 使用预定义的10月文件列表
    print("开始公司名称关键词分析...")
    print("分析时间范围: 2025年10月1日 - 2025年10月31日")
    
    # 获取10月所有文件
    file_template = os.path.join('processed', '{date}', 'all_data.csv')
    csv_file_paths = generate_date_file_list('2025-10-01', '2025-10-31', file_template)
    
    # 过滤出存在的文件
    existing_files, missing_files = filter_existing_files(csv_file_paths)
    
    if missing_files:
        print(f"\n警告: 有 {len(missing_files)} 个文件不存在:")
        for missing in missing_files[:5]:  # 只显示前5个
            print(f"  - {missing}")
        if len(missing_files) > 5:
            print(f"  ... 以及其他 {len(missing_files) - 5} 个文件")
    
    if not existing_files:
        print("错误: 没有找到任何有效的数据文件!")
        return
    
    # 打印将要分析的文件
    print(f"\n找到 {len(existing_files)} 个有效文件:")
    for i, path in enumerate(existing_files[:5]):  # 只显示前5个
        print(f"  {i+1}. {os.path.basename(os.path.dirname(path))}/all_data.csv")
    if len(existing_files) > 5:
        print(f"  ... 以及其他 {len(existing_files) - 5} 个文件")
    
    # 创建分析器
    analyzer = CompanyNameAnalyzer(existing_files)
    
    # 运行分析
    keyword_df = analyzer.run_analysis(
        min_freq=10,         # 只保留出现频数>=10的关键词
        top_k=30,           # 在控制台输出前30个结果
        exclude_partners=['YXM_CODE', 'LYX_CODE']  # 排除指定合作方数据
    )
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()