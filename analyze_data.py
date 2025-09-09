"""
数据分析工具
用于分析处理后的合作方数据
"""

import pandas as pd
import os
from collections import Counter

class PartnerDataAnalyzer:
    """
    合作方数据分析器
    """
    
    def __init__(self, data_dir='processed_data'):
        """
        初始化分析器
        
        Args:
            data_dir: 处理后数据的目录
        """
        self.data_dir = data_dir
        self.partner_files = self._get_partner_files()
    
    def _get_partner_files(self):
        """获取所有合作方的CSV文件"""
        if not os.path.exists(self.data_dir):
            return {}
        
        files = {}
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                partner_code = filename.replace('.csv', '')
                files[partner_code] = os.path.join(self.data_dir, filename)
        
        return files
    
    def load_partner_data(self, partner_code):
        """
        加载指定合作方的数据
        
        Args:
            partner_code: 合作方编码
            
        Returns:
            DataFrame: 合作方数据
        """
        if partner_code not in self.partner_files:
            print(f"找不到合作方 {partner_code} 的数据")
            return None
        
        try:
            df = pd.read_csv(self.partner_files[partner_code])
            return df
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None
    
    def analyze_all_partners(self):
        """分析所有合作方的数据概览"""
        print("="*60)
        print("所有合作方数据概览")
        print("="*60)
        
        total_records = 0
        success_records = 0
        
        partner_stats = []
        
        for partner_code, file_path in self.partner_files.items():
            try:
                df = pd.read_csv(file_path)
                record_count = len(df)
                success_count = len(df[df['label'] == '成功'])
                success_rate = (success_count / record_count * 100) if record_count > 0 else 0
                
                partner_stats.append({
                    'partner': partner_code,
                    'total_records': record_count,
                    'success_count': success_count,
                    'success_rate': success_rate
                })
                
                total_records += record_count
                success_records += success_count
                
            except Exception as e:
                print(f"处理 {partner_code} 时出错: {e}")
                continue
        
        # 按总记录数排序
        partner_stats.sort(key=lambda x: x['total_records'], reverse=True)
        
        print(f"\n合作方数据统计:")
        print("-" * 80)
        print(f"{'合作方':<12} {'总记录数':<10} {'成功数':<8} {'成功率':<8} {'失败率':<8}")
        print("-" * 80)
        
        for stat in partner_stats:
            fail_rate = 100 - stat['success_rate']
            print(f"{stat['partner']:<12} {stat['total_records']:<10} {stat['success_count']:<8} "
                  f"{stat['success_rate']:<7.1f}% {fail_rate:<7.1f}%")
        
        overall_success_rate = (success_records / total_records * 100) if total_records > 0 else 0
        
        print("-" * 80)
        print(f"{'总计':<12} {total_records:<10} {success_records:<8} {overall_success_rate:<7.1f}% "
              f"{100-overall_success_rate:<7.1f}%")
        
        return partner_stats
    
    def analyze_partner_features(self, partner_code):
        """
        分析指定合作方的特征分布
        
        Args:
            partner_code: 合作方编码
        """
        df = self.load_partner_data(partner_code)
        if df is None:
            return
        
        print(f"\n{'='*60}")
        print(f"合作方 {partner_code} 特征分析")
        print(f"{'='*60}")
        
        print(f"总记录数: {len(df)}")
        
        # 分析每个特征
        feature_columns = [col for col in df.columns if col != 'label']
        
        for feature in feature_columns:
            print(f"\n特征: {feature}")
            print("-" * 40)
            
            # 处理缺失值
            non_null_count = df[feature].count()
            null_count = len(df) - non_null_count
            
            if null_count > 0:
                print(f"缺失值: {null_count} ({null_count/len(df)*100:.1f}%)")
            
            if non_null_count == 0:
                print("该特征全部为空值")
                continue
            
            # 统计值分布
            value_counts = df[feature].value_counts()
            unique_count = len(value_counts)
            
            print(f"唯一值数量: {unique_count}")
            
            if unique_count <= 20:
                # 显示所有值的分布
                print("值分布:")
                for value, count in value_counts.head(20).items():
                    percentage = count / non_null_count * 100
                    print(f"  {value}: {count} ({percentage:.1f}%)")
            else:
                # 只显示最常见的值
                print("最常见的值:")
                for value, count in value_counts.head(10).items():
                    percentage = count / non_null_count * 100
                    print(f"  {value}: {count} ({percentage:.1f}%)")
                print(f"  ... 还有 {unique_count - 10} 个其他值")
    
    def compare_partners(self, feature):
        """
        比较不同合作方在某个特征上的分布
        
        Args:
            feature: 要比较的特征名
        """
        print(f"\n{'='*60}")
        print(f"特征 '{feature}' 在不同合作方的分布比较")
        print(f"{'='*60}")
        
        partner_distributions = {}
        
        for partner_code in self.partner_files.keys():
            df = self.load_partner_data(partner_code)
            if df is not None and feature in df.columns:
                value_counts = df[feature].value_counts()
                total_count = df[feature].count()
                
                # 转换为百分比
                percentages = (value_counts / total_count * 100).round(1)
                partner_distributions[partner_code] = percentages
        
        if not partner_distributions:
            print(f"没有找到特征 '{feature}' 的数据")
            return
        
        # 获取所有可能的值
        all_values = set()
        for dist in partner_distributions.values():
            all_values.update(dist.index)
        
        all_values = sorted(list(all_values))
        
        # 创建比较表
        print(f"\n特征值分布 (百分比):")
        print("-" * (15 + len(partner_distributions) * 12))
        
        # 表头
        header = f"{'值':<15}"
        for partner in sorted(partner_distributions.keys()):
            header += f"{partner:<12}"
        print(header)
        print("-" * (15 + len(partner_distributions) * 12))
        
        # 数据行
        for value in all_values[:20]:  # 只显示前20个值
            row = f"{str(value):<15}"
            for partner in sorted(partner_distributions.keys()):
                percentage = partner_distributions[partner].get(value, 0)
                row += f"{percentage:<11.1f}%"
            print(row)
    
    def analyze_success_patterns(self):
        """分析成功模式"""
        print(f"\n{'='*60}")
        print("成功模式分析")
        print(f"{'='*60}")
        
        all_success_data = []
        all_fail_data = []
        
        # 收集所有成功和失败的数据
        for partner_code in self.partner_files.keys():
            df = self.load_partner_data(partner_code)
            if df is not None:
                success_data = df[df['label'] == '成功']
                fail_data = df[df['label'] == '失败']
                
                all_success_data.append(success_data)
                all_fail_data.append(fail_data)
        
        if not all_success_data:
            print("没有找到成功的数据")
            return
        
        # 合并数据
        success_df = pd.concat(all_success_data, ignore_index=True)
        fail_df = pd.concat(all_fail_data, ignore_index=True)
        
        print(f"成功案例总数: {len(success_df)}")
        print(f"失败案例总数: {len(fail_df)}")
        
        # 分析特征差异
        feature_columns = [col for col in success_df.columns if col != 'label']
        
        for feature in feature_columns:
            print(f"\n特征: {feature}")
            print("-" * 40)
            
            success_counts = success_df[feature].value_counts().head(5)
            fail_counts = fail_df[feature].value_counts().head(5)
            
            print("成功案例中最常见的值:")
            for value, count in success_counts.items():
                percentage = count / len(success_df) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
            
            print("失败案例中最常见的值:")
            for value, count in fail_counts.items():
                percentage = count / len(fail_df) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")


def main():
    """主函数"""
    analyzer = PartnerDataAnalyzer()
    
    if not analyzer.partner_files:
        print("没有找到处理后的数据文件。")
        print("请先运行 process_loan_data.py 生成数据。")
        return
    
    # 分析所有合作方概览
    partner_stats = analyzer.analyze_all_partners()
    
    # 分析成功模式
    analyzer.analyze_success_patterns()
    
    # 示例：分析特定合作方
    if 'LXJ_CODE' in analyzer.partner_files:
        analyzer.analyze_partner_features('LXJ_CODE')
    
    # 示例：比较特征分布
    analyzer.compare_partners('companyInfo.industry')


if __name__ == "__main__":
    main()
