"""
贷款数据预处理主程序
用于演示如何使用 DataProcessor 类进行数据处理和分析
"""

from data_processor import DataProcessor
import os

def main():
    """
    主函数：演示如何使用数据处理器
    """
    # 初始化数据处理器
    processor = DataProcessor('data/20250903.csv')
    
    # 定义要提取的特征列表
    # 用户可以根据需要修改这个列表
    feature_list = [
        'amount',                    # 贷款金额
        'area',                      # 所在区域
        'bankCardInfo.bankCardNo',   # 银行卡号
        'bankCardInfo.bankCode',     # 银行代码
        'bankCardInfo.cardType',     # 银行卡类型
        'city',                      # 城市
        'companyInfo.industry',      # 所属行业
        'companyInfo.occupation',    # 职业
        'degree',                    # 学历
        'deviceInfo.osType',         # 设备系统类型
        'deviceInfo.phoneType',      # 手机型号
        'income',                    # 收入
        'maritalStatus',             # 婚姻状况
        'province',                  # 省份
        'purpose',                   # 借款用途
        'term',                      # 贷款期数
        'idInfo.gender',             # 性别
        'customerSource'             # 客户来源
    ]
    
    print("="*60)
    print("贷款数据预处理系统")
    print("="*60)
    
    print("\n当前配置的特征列表:")
    for i, feature in enumerate(feature_list, 1):
        print(f"{i:2d}. {feature}")
    
    # 处理数据并按合作方生成CSV文件
    print("\n开始数据处理...")
    partner_counts = processor.process_data_by_partner(feature_list)
    
    # 打印处理结果统计
    print("\n处理结果统计:")
    print("-" * 40)
    total_records = 0
    for partner, count in sorted(partner_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{partner}: {count:,} 条记录")
        total_records += count
    
    print(f"\n总共处理: {total_records:,} 条记录")
    print(f"生成的CSV文件保存在: processed_data/ 目录下")
    
    # 进行特征分析
    print("\n开始特征分析...")
    analysis_results = processor.analyze_features(feature_list, sample_size=5000)
    
    # 打印分析报告
    processor.print_analysis_report(analysis_results)
    
    print("\n处理完成！")


def custom_feature_demo():
    """
    演示如何使用自定义特征列表
    """
    processor = DataProcessor('data/20250903.csv')
    
    # 自定义特征列表示例
    custom_features = [
        'amount',
        'bankCardInfo.bankCode',
        'bankCardInfo.cardType',
        'companyInfo.industry'
    ]
    
    print("使用自定义特征列表:")
    for i, feature in enumerate(custom_features, 1):
        print(f"{i}. {feature}")
    
    # 处理数据
    partner_counts = processor.process_data_by_partner(custom_features)
    
    # 分析特征
    analysis_results = processor.analyze_features(custom_features)
    processor.print_analysis_report(analysis_results)


def interactive_mode():
    """
    交互式模式，允许用户自定义特征列表
    """
    print("="*60)
    print("交互式特征选择模式")
    print("="*60)
    
    # 显示可用的特征选项
    available_features = [
        'amount',                    # 贷款金额
        'term',                      # 贷款期数
        'area',                      # 所在区域
        'city',                      # 城市
        'province',                  # 省份
        'degree',                    # 学历
        'maritalStatus',             # 婚姻状况
        'income',                    # 收入
        'purpose',                   # 借款用途
        'customerSource',            # 客户来源
        'jobFunctions',              # 工作性质
        'resideFunctions',           # 居住性质
        'bankCardInfo.bankCode',     # 银行代码
        'bankCardInfo.cardType',     # 银行卡类型
        'companyInfo.industry',      # 所属行业
        'companyInfo.occupation',    # 职业
        'deviceInfo.osType',         # 设备系统类型
        'deviceInfo.phoneType',      # 手机型号
        'deviceInfo.phoneMaker',     # 手机厂商
        'idInfo.gender',             # 性别
        'idInfo.nation',             # 民族
        'linkmanList.0.relationship' # 第一个联系人关系
    ]
    
    print("\n可选择的特征:")
    for i, feature in enumerate(available_features, 1):
        print(f"{i:2d}. {feature}")
    
    print("\n请输入要选择的特征编号，用逗号分隔 (例如: 1,3,5,7):")
    print("或直接按Enter使用默认特征集合")
    
    user_input = input().strip()
    
    if not user_input:
        # 使用默认特征
        selected_features = [
            'amount',
            'bankCardInfo.bankCode', 
            'bankCardInfo.cardType',
            'companyInfo.industry'
        ]
        print("使用默认特征集合")
    else:
        try:
            # 解析用户输入
            indices = [int(x.strip()) - 1 for x in user_input.split(',')]
            selected_features = [available_features[i] for i in indices if 0 <= i < len(available_features)]
            
            if not selected_features:
                print("无效的选择，使用默认特征集合")
                selected_features = ['amount', 'bankCardInfo.bankCode', 'bankCardInfo.cardType', 'companyInfo.industry']
                
        except (ValueError, IndexError):
            print("输入格式错误，使用默认特征集合")
            selected_features = ['amount', 'bankCardInfo.bankCode', 'bankCardInfo.cardType', 'companyInfo.industry']
    
    print(f"\n选择的特征:")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    # 处理数据
    processor = DataProcessor('data/20250903.csv')
    
    print("\n开始处理数据...")
    partner_counts = processor.process_data_by_partner(selected_features)
    
    print("\n处理结果统计:")
    print("-" * 40)
    for partner, count in sorted(partner_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{partner}: {count:,} 条记录")
    
    # 分析特征
    print("\n开始特征分析...")
    analysis_results = processor.analyze_features(selected_features)
    processor.print_analysis_report(analysis_results)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        # 交互式模式
        interactive_mode()
    elif len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # 演示模式
        custom_feature_demo()
    else:
        # 默认模式
        main()
    
    print("\n运行选项:")
    print("python process_loan_data.py          # 使用默认特征集合")
    print("python process_loan_data.py demo     # 运行简单演示")
    print("python process_loan_data.py interactive  # 交互式选择特征")
