"""
使用示例：演示如何使用贷款数据预处理系统
"""

from data_processor import DataProcessor
import os

def example_basic_processing():
    """示例1：基本数据处理"""
    print("="*60)
    print("示例1：基本数据处理")
    print("="*60)
    
    # 创建处理器
    processor = DataProcessor('data/20250903.csv')
    
    # 定义要提取的特征
    features = [
        'amount',                 # 贷款金额
        'income',                 # 收入
        'degree',                 # 学历
        'companyInfo.industry'    # 行业
    ]
    
    print("提取的特征:")
    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature}")
    
    # 处理数据
    partner_counts = processor.process_data_by_partner(features)
    
    print(f"\n成功处理 {sum(partner_counts.values())} 条记录")
    print("各合作方数据量:")
    for partner, count in sorted(partner_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {partner}: {count}")


def example_custom_analysis():
    """示例2：自定义特征分析"""
    print("\n" + "="*60)
    print("示例2：自定义特征分析")
    print("="*60)
    
    processor = DataProcessor('data/20250903.csv')
    
    # 金融相关特征
    financial_features = [
        'amount',
        'term',
        'income',
        'bankCardInfo.bankCode',
        'bankCardInfo.cardType'
    ]
    
    print("分析金融相关特征:")
    analysis = processor.analyze_features(financial_features, sample_size=3000)
    
    # 只显示部分分析结果
    print(f"\n金融特征摘要:")
    for feature, stats in analysis.items():
        if feature != 'label':
            print(f"{feature}: {stats['type']} 类型, {stats['unique_count']} 个唯一值")


def example_demographic_analysis():
    """示例3：人口统计学特征分析"""
    print("\n" + "="*60)
    print("示例3：人口统计学特征分析")
    print("="*60)
    
    processor = DataProcessor('data/20250903.csv')
    
    # 人口统计学特征
    demographic_features = [
        'province',
        'city',
        'degree',
        'maritalStatus',
        'idInfo.gender',
        'idInfo.nation'
    ]
    
    print("处理人口统计学特征...")
    partner_counts = processor.process_data_by_partner(demographic_features)
    
    print("分析特征分布...")
    analysis = processor.analyze_features(demographic_features[:3], sample_size=2000)  # 只分析前3个特征以节省时间
    
    print("\n人口统计学特征概览:")
    for feature, stats in analysis.items():
        if feature != 'label' and stats['type'] == 'categorical':
            top_value = list(stats['values'].keys())[0] if stats['values'] else 'N/A'
            print(f"{feature}: 最常见值为 '{top_value}'")


def example_risk_features():
    """示例4：风险评估相关特征"""
    print("\n" + "="*60)
    print("示例4：风险评估相关特征")
    print("="*60)
    
    processor = DataProcessor('data/20250903.csv')
    
    # 风险评估特征
    risk_features = [
        'amount',
        'income',
        'purpose',
        'jobFunctions',
        'resideFunctions',
        'companyInfo.occupation',
        'deviceInfo.osType'
    ]
    
    print("提取风险评估特征...")
    partner_counts = processor.process_data_by_partner(risk_features)
    
    # 简单的成功率分析
    print("\n风险特征处理完成")
    print("生成的文件可用于:")
    print("- 机器学习模型训练")
    print("- 风险评分模型开发")
    print("- A/B测试分析")


def example_data_quality_check():
    """示例5：数据质量检查"""
    print("\n" + "="*60)
    print("示例5：数据质量检查")
    print("="*60)
    
    processor = DataProcessor('data/20250903.csv')
    
    # 检查关键字段的数据质量
    key_features = [
        'amount',
        'phone',
        'bankCardInfo.bankCardNo',
        'idInfo.idNumber'
    ]
    
    print("检查关键字段数据质量...")
    analysis = processor.analyze_features(key_features, sample_size=1000)
    
    print("\n数据质量报告:")
    for feature, stats in analysis.items():
        if feature != 'label':
            missing_rate = (1000 - stats['count']) / 1000 * 100 if stats['count'] < 1000 else 0
            print(f"{feature}:")
            print(f"  有效数据: {stats['count']}/1000")
            print(f"  缺失率: {missing_rate:.1f}%")
            print(f"  唯一值: {stats['unique_count']}")


def main():
    """运行所有示例"""
    print("贷款数据预处理系统 - 使用示例")
    print("本脚本演示了系统的各种使用方式")
    
    try:
        # 检查数据文件是否存在
        if not os.path.exists('data/20250903.csv'):
            print("错误: 找不到数据文件 data/20250903.csv")
            return
        
        # 运行示例
        example_basic_processing()
        
        # 询问是否继续运行其他示例
        print("\n是否继续运行其他示例? (y/n):", end=" ")
        response = input().strip().lower()
        
        if response in ['y', 'yes', '是']:
            example_custom_analysis()
            example_demographic_analysis()
            example_risk_features()
            example_data_quality_check()
        
        print("\n" + "="*60)
        print("示例演示完成!")
        print("="*60)
        print("\n生成的文件保存在 processed_data/ 目录下")
        print("使用 'python analyze_data.py' 进行详细分析")
        
    except KeyboardInterrupt:
        print("\n\n用户中断了程序执行")
    except Exception as e:
        print(f"\n发生错误: {e}")


if __name__ == "__main__":
    main()
