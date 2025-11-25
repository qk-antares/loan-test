import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from typing import Dict, List, Tuple, Any
import warnings
from ml_data_loader import DataLoader
warnings.filterwarnings('ignore')
import shap, os, matplotlib.pyplot as plt
class LoanDistributionModel:
    """
    贷款分发智能决策模型 - 重构版本

    主要功能：
    1. 多标签分类：预测每个合作方的通过概率
    2. 智能排序：基于通过概率进行排序推荐
    3. 处理类别不平衡：多种策略应对正负样本不均衡
    """

    def __init__(self, train_data_dir: str = None, test_data_dir: str = None):
        """
        初始化模型

        Args:
            train_data_dir: 训练数据目录，如果为None则自动寻找上级目录
            test_data_dir: 测试数据目录，如果为None则自动寻找上级目录
        """


        self.partners = []  # 合作方列表
        self.feature_columns = []  # 特征列名
        self.models = {}  # 每个合作方的模型
        self.encoders = {}  # 编码器
        self.scaler = StandardScaler()

        # 定义使用的特征列表（保持不变）
        self.feature_list = [
            'amount',
            'bankCardInfo.bankCode',
            'city',
            'companyInfo.companyName',
            'companyInfo.industry',
            'companyInfo.occupation',
            'customerSource',
            'degree',
            'idInfo.birthDate',
            'idInfo.gender',
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
            'deviceInfo.osType',
            # 'deviceInfo.isCrossDomain',
            'deviceInfo.applyPos'
        ]

        # 公司名称过滤规则
        self.company_filter_rules = self._init_company_filters()

    def _init_company_filters(self) -> Dict[str, List[str]]:
        """
        初始化公司名称过滤规则

        Returns:
            过滤规则字典
        """
        return {
            'AWJ': ['公安局', '警察', '法院', '军队', '检察院', '城市管理局', '律师', '记者', '贷款', '金融', '执行局',
                    '监狱', '交通警察', '派出所', '刑事侦查部门', '交警', '刑侦'],
            'RONG': ['学校', '小学', '中学', '大学', '学院', '公检法'],
        }


    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征预处理

        Args:
            df: 原始数据

        Returns:
            处理后的特征数据
        """
        print("开始特征预处理...")
        processed_df = df.copy()

        # 1. 只保留选择的特征列
        available_features = [col for col in self.feature_list if col in processed_df.columns]
        missing_features = [col for col in self.feature_list if col not in processed_df.columns]

        if missing_features:
            print(f"警告: 以下特征在数据中缺失: {missing_features}")

        # 添加必要的标识列
        # FIX_BUG: 列名
        required_cols = ['partner_code', 'label']
        columns_to_keep = available_features + required_cols

        processed_df = processed_df[columns_to_keep]
        print(f"使用 {len(available_features)} 个特征")

        # 2. 公司名称过滤规则特征
        if 'companyInfo.companyName' in processed_df.columns:
            company_series = processed_df['companyInfo.companyName'].astype(str).fillna("")

            # 遍历所有类别（如 AWJ、RONG）
            for rule_name, keywords in self.company_filter_rules.items():
                for keyword in keywords:
                    # 列名格式示例： company_AWJ_公安局
                    col_name = f"company_{rule_name}_{keyword}"
                    processed_df[col_name] = company_series.apply(
                        lambda x: 1 if keyword in x else 0
                    )

        # 3. 学历 one-hot 编码 (JUNIOR=1, ..., DOCTOR=6)
        if 'degree' in processed_df.columns:
            processed_df['degree'] = processed_df['degree'].fillna(0)  # 缺失值填0

            def simplify_degree(degree_val):
                mapping = {
                    1.0: 'JUNIOR',
                    2.0: 'SENIOR',
                    3.0: 'COLLEGE',
                    4.0: 'BACHELOR',
                    5.0: 'MASTER',
                    6.0: 'DOCTOR'
                }
                return mapping.get(degree_val, 'UNKNOWN')

            processed_df['degree_group'] = processed_df['degree'].apply(simplify_degree)
            degree_dummies = pd.get_dummies(processed_df['degree_group'], prefix='degree')
            processed_df = pd.concat([processed_df, degree_dummies], axis=1)
            processed_df = processed_df.drop(['degree', 'degree_group'], axis=1)
            print("学历 one-hot 列：", degree_dummies.columns.tolist())
            print("学历类别分布：\n", degree_dummies.sum())

        # 收入
        if 'income' in processed_df.columns:
            processed_df['income'] = processed_df['income'].fillna(0)  # 缺失值填0
            def simplify_income(income_val):
                mapping = {
                    1.0: 'A',
                    2.0: 'B',
                    3.0: 'C',
                    4.0: 'D'
                }
                return mapping.get(income_val, 'UNKNOWN')

            processed_df['income_group'] = processed_df['income'].apply(simplify_income)
            income_dummies = pd.get_dummies(processed_df['income_group'], prefix='income')
            processed_df = pd.concat([processed_df, income_dummies], axis=1)
            processed_df = processed_df.drop(['income', 'income_group'], axis=1)
            print("收入 one-hot 列：", income_dummies.columns.tolist())
            print("收入类别分布：\n", income_dummies.sum())

        # 4.5 城市与省份自定义编码（新增部分） ==============================
        if 'city' in processed_df.columns:
            processed_df['city'] = processed_df['city'].fillna('UNKNOWN').astype(str)

            first_tier = ['北京市', '上海市', '广州市', '深圳市']
            new_first_tier = ['杭州市', '南京市', '成都市', '武汉市', '重庆市', '苏州市', '天津市', '西安市', '长沙市',
                              '青岛市']
            # 分类函数
            def simplify_city(city):
                if city in first_tier:
                    return 'first_tier'
                elif city in new_first_tier:
                    return 'new_first_tier'
                else:
                    return 'others'
            # 应用分组
            processed_df['city_group'] = processed_df['city'].apply(simplify_city)
            # one-hot 编码
            city_dummies = pd.get_dummies(processed_df['city_group'], prefix='city')
            # 合并到原数据
            processed_df = pd.concat([processed_df, city_dummies], axis=1)
            # 删除原始列和中间列
            processed_df = processed_df.drop(['city', 'city_group'], axis=1)
            # 打印调试信息
            print("城市分组 one-hot 列：", city_dummies.columns.tolist())
            print("城市类别分布：\n", city_dummies.sum())


        # 🗺️ 省份 one-hot 编码：重点省份、省外、未知
        if 'province' in processed_df.columns:
            processed_df['province'] = processed_df['province'].fillna('UNKNOWN').astype(str)
            key_provinces = ['北京市', '上海市', '广东省', '浙江省', '江苏省']
            def simplify_province(province):
                if province == 'UNKNOWN':
                    return 'UNKNOWN'
                elif province in key_provinces:
                    return 'key_province'
                else:
                    return 'other_province'
            # 创建新列
            processed_df['province_group'] = processed_df['province'].apply(simplify_province)
            # One-hot 编码
            province_dummies = pd.get_dummies(processed_df['province_group'], prefix='province')
            # 合并结果
            processed_df = pd.concat([processed_df, province_dummies], axis=1)
            # 删除原始列
            processed_df = processed_df.drop(['province', 'province_group'], axis=1)
            # 输出结果查看
            print("省份分组 one-hot 列：", province_dummies.columns.tolist())
            print("省份类别分布：\n", province_dummies.sum())


        # 银行编码 one-hot 编码
        if 'bankCardInfo.bankCode' in processed_df.columns:
            # 填充缺失值并转字符串
            processed_df['bankCardInfo.bankCode'] = processed_df['bankCardInfo.bankCode'].fillna('UNKNOWN').astype(str)
            # 清洗数据：去掉浮点 '.0' 和空格
            processed_df['bankCardInfo.bankCode'] = processed_df['bankCardInfo.bankCode'].str.replace('.0', '',
                                                                                                      regex=False).str.strip()
            # 定义各类银行代码
            state_owned_banks = ['102', '103', '104', '105', '301', '402']  # 国有大行
            joint_stock_banks = ['302', '303', '304', '305', '306', '307', '308', '309', '310']  # 股份制银行
            small_local_banks = ['313', '403', '404', '501']  # 小银行、农商行、外资行等

            # 分类函数，带打印检查
            def categorize_bank(bank_code):
                if bank_code in state_owned_banks:
                    category = 'state_owned'
                elif bank_code in joint_stock_banks:
                    category = 'joint_stock'
                elif bank_code in small_local_banks:
                    category = 'small_local'
                else:
                    category = 'unknown'
                return category

            processed_df['bank_category'] = processed_df['bankCardInfo.bankCode'].apply(categorize_bank)

            # one-hot 编码
            bank_dummies = pd.get_dummies(processed_df['bank_category'], prefix='bank')
            processed_df = pd.concat([processed_df, bank_dummies], axis=1)

            # 删除原始列和分类列
            processed_df = processed_df.drop(['bankCardInfo.bankCode', 'bank_category'], axis=1)


        if 'deviceInfo.applyPos' in processed_df.columns:
            processed_df['deviceInfo.applyPos'] = processed_df['deviceInfo.applyPos'].fillna('UNKNOWN').astype(str)
            # 提取前三个字符作为省份
            processed_df['device_province'] = processed_df['deviceInfo.applyPos'].apply(
                lambda x: x[:3] if x != 'UNKNOWN' else 'UNKNOWN'
            )

            # 分组函数
            key_provinces = ['北京市', '上海市', '广东省', '浙江省', '江苏省']
            def simplify_device_province(prov):
                if prov == 'UNKNOWN':
                    return 'unknown'
                elif prov in key_provinces:
                    return 'key_province'
                else:
                    return 'others'
            processed_df['device_province_group'] = processed_df['device_province'].apply(simplify_device_province)
            # one-hot 编码
            device_province_dummies = pd.get_dummies(processed_df['device_province_group'], prefix='device_province')
            processed_df = pd.concat([processed_df, device_province_dummies], axis=1)

            # 删除原始列和中间列
            processed_df = processed_df.drop(['deviceInfo.applyPos', 'device_province', 'device_province_group'],
                                             axis=1)
            # 打印调试信息
            print("设备省份 one-hot 列：", device_province_dummies.columns.tolist())
            print("设备省份类别分布：\n", device_province_dummies.sum())


        if 'idInfo.birthDate' in processed_df.columns:
            # 年龄已经是数值
            def simplify_age(age):
                if pd.isna(age):
                    return 'unknown'  # 缺失值
                elif 30 <= age <= 45:
                    return '30-45'
                elif 45 < age <= 60:
                    return '45-60'
                else:
                    return 'other'  # 小于30或大于60

            processed_df['age_simple'] = processed_df['idInfo.birthDate'].apply(simplify_age)

            # one-hot 编码
            age_dummies = pd.get_dummies(processed_df['age_simple'], prefix='age')
            processed_df = pd.concat([processed_df, age_dummies], axis=1)

            # 删除原始列和简化列
            processed_df = processed_df.drop(['idInfo.birthDate', 'age_simple'], axis=1)

            print("年龄分组 one-hot 列：", age_dummies.columns.tolist())
            print("年龄分组总分布：\n", age_dummies.sum())


        if 'pictureInfo.0.faceScore' in processed_df.columns:
            # 转换为数值型
            processed_df['pictureInfo.0.faceScore'] = pd.to_numeric(
                processed_df['pictureInfo.0.faceScore'], errors='coerce'
            )

            # 定义分组函数
            def simplify_face_score(score):
                if pd.isna(score):
                    return 'unknown'
                elif 90 < score <= 100:
                    return '90-100'
                elif 80 <= score <= 90:
                    return '80-90'
                elif 0 <= score < 80:
                    return '0-80'
                else:
                    return 'unknown'
            # 应用分组
            processed_df['faceScore_group'] = processed_df['pictureInfo.0.faceScore'].apply(simplify_face_score)
            # one-hot 编码
            face_dummies = pd.get_dummies(processed_df['faceScore_group'], prefix='face')
            # 合并编码列
            processed_df = pd.concat([processed_df, face_dummies], axis=1)
            # 删除原始列与中间列
            processed_df = processed_df.drop(['pictureInfo.0.faceScore', 'faceScore_group'], axis=1)
            # 调试信息
            print("脸部分数分组 one-hot 列：", face_dummies.columns.tolist())
            print("脸部分组总分布：\n", face_dummies.sum())


        if 'idInfo.validityDate' in processed_df.columns:
            # 定义分组函数
            def simplify_validity(days_left):
                if pd.isna(days_left):
                    return 'invalid_or_missing'
                elif days_left <= 365:
                    return 'within_1y'
                elif days_left <= 365 * 5:
                    return '1_5y'
                else:
                    return 'over_5y'
            # 应用分类函数
            processed_df['validity_group'] = processed_df['idInfo.validityDate'].apply(simplify_validity)
            # one-hot 编码
            validity_dummies = pd.get_dummies(processed_df['validity_group'], prefix='validity')
            # 合并编码列
            processed_df = pd.concat([processed_df, validity_dummies], axis=1)
            # 删除原始列与中间列
            processed_df = processed_df.drop(['idInfo.validityDate', 'validity_group'], axis=1)
            # 输出调试信息
            print("身份证有效期分组 one-hot 列：", validity_dummies.columns.tolist())
            print("身份证有效期分布：\n", validity_dummies.sum())

        # jobFunctions
        # if 'jobFunctions' in processed_df.columns:
        #     processed_df['jobFunctions'] = processed_df['jobFunctions'].fillna('UNKNOWN').astype(str)
        #     job_dummies = pd.get_dummies(processed_df['jobFunctions'], prefix='job')
        #     processed_df = pd.concat([processed_df, job_dummies], axis=1)
        #     processed_df = processed_df.drop('jobFunctions', axis=1)

        # resideFunctions
        if 'resideFunctions' in processed_df.columns:
            processed_df['resideFunctions'] = processed_df['resideFunctions'].fillna('UNKNOWN').astype(str)
            reside_dummies = pd.get_dummies(processed_df['resideFunctions'], prefix='reside')
            processed_df = pd.concat([processed_df, reside_dummies], axis=1)
            processed_df = processed_df.drop('resideFunctions', axis=1)

        if 'maritalStatus' in processed_df.columns:
            processed_df['maritalStatus'] = processed_df['maritalStatus'].fillna('UNKNOWN').astype(str)
            marital_dummies = pd.get_dummies(processed_df['maritalStatus'], prefix='marital')
            processed_df = pd.concat([processed_df, marital_dummies], axis=1)
            processed_df = processed_df.drop('maritalStatus', axis=1)

        if 'purpose' in processed_df.columns:
            processed_df['purpose'] = processed_df['purpose'].fillna('UNKNOWN').astype(str)
            purpose_dummies = pd.get_dummies(processed_df['purpose'], prefix='purpose')
            processed_df = pd.concat([processed_df, purpose_dummies], axis=1)
            processed_df = processed_df.drop('purpose', axis=1)

        if 'customerSource' in processed_df.columns:
            processed_df['customerSource'] = processed_df['customerSource'].fillna('UNKNOWN').astype(str)
            customerSource_dummies = pd.get_dummies(processed_df['customerSource'], prefix='customerSource')
            processed_df = pd.concat([processed_df, customerSource_dummies], axis=1)
            processed_df = processed_df.drop('customerSource', axis=1)

        if 'companyInfo.occupation' in processed_df.columns:
            processed_df['companyInfo.occupation'] = processed_df['companyInfo.occupation'].fillna('UNKNOWN').astype(
                str)
            occupation_dummies = pd.get_dummies(processed_df['companyInfo.occupation'], prefix='occupation')
            processed_df = pd.concat([processed_df, occupation_dummies], axis=1)
            processed_df = processed_df.drop('companyInfo.occupation', axis=1)

        if 'companyInfo.industry' in processed_df.columns:
            processed_df['companyInfo.industry'] = processed_df['companyInfo.industry'].fillna(
                'UNKNOWN').astype(
                str)
            industry_dummies = pd.get_dummies(processed_df['companyInfo.industry'], prefix='industry')
            processed_df = pd.concat([processed_df, industry_dummies], axis=1)
            processed_df = processed_df.drop('companyInfo.industry', axis=1)

        if 'deviceInfo.osType' in processed_df.columns:
            processed_df['deviceInfo.osType'] = processed_df['deviceInfo.osType'].fillna('UNKNOWN').astype(str)
            ostype_dummies = pd.get_dummies(processed_df['deviceInfo.osType'], prefix='ostype')
            processed_df = pd.concat([processed_df, ostype_dummies], axis=1)
            processed_df = processed_df.drop('deviceInfo.osType', axis=1)

        if 'idInfo.gender' in processed_df.columns:
            processed_df['idInfo.gender'] = processed_df['idInfo.gender'].fillna('UNKNOWN').astype(str)
            gender_dummies = pd.get_dummies(processed_df['idInfo.gender'], prefix='gender')
            processed_df = pd.concat([processed_df, gender_dummies], axis=1)
            processed_df = processed_df.drop('idInfo.gender', axis=1)

        if 'linkmanList.0.relationship' in processed_df.columns:
            # 填充缺失值并转字符串
            processed_df['linkmanList.0.relationship'] = processed_df['linkmanList.0.relationship'].fillna(
                'UNKNOWN').astype(str)
            # one-hot 编码
            relationship_dummies = pd.get_dummies(processed_df['linkmanList.0.relationship'], prefix='relationship')
            processed_df = pd.concat([processed_df, relationship_dummies], axis=1)
            processed_df = processed_df.drop('linkmanList.0.relationship', axis=1)

        if 'linkmanList.0.relationship' in processed_df.columns:
            # 填充缺失值并转字符串
            processed_df['linkmanList.0.relationship'] = processed_df['linkmanList.0.relationship'].fillna(
                'UNKNOWN').astype(str)
            # one-hot 编码
            relationship0_dummies = pd.get_dummies(processed_df['linkmanList.0.relationship'], prefix='relationship0')
            processed_df = pd.concat([processed_df, relationship0_dummies], axis=1)
            processed_df = processed_df.drop('linkmanList.0.relationship', axis=1)

        if 'linkmanList.1.relationship' in processed_df.columns:
            # 填充缺失值并转字符串
            processed_df['linkmanList.1.relationship'] = processed_df['linkmanList.1.relationship'].fillna(
                'UNKNOWN').astype(str)
            # one-hot 编码
            relationship1_dummies = pd.get_dummies(processed_df['linkmanList.1.relationship'], prefix='relationship1')
            processed_df = pd.concat([processed_df, relationship1_dummies], axis=1)
            processed_df = processed_df.drop('linkmanList.1.relationship', axis=1)


        if 'idInfo.nation' in processed_df.columns:
            # 1. 填充缺失值并转字符串
            processed_df['idInfo.nation'] = processed_df['idInfo.nation'].fillna('UNKNOWN').astype(str).str.strip()
            # 2. 简化分类：汉族为 'han'，其他为 'other'
            def simplify_nation(nation):
                if nation == '汉':
                    return 'han'
                elif nation == 'UNKNOWN':
                    return 'unknown'
                else:
                    return 'minority'
            processed_df['nation_simple'] = processed_df['idInfo.nation'].apply(simplify_nation)
            # 3. one-hot 编码
            nation_dummies = pd.get_dummies(processed_df['nation_simple'], prefix='nation')
            processed_df = pd.concat([processed_df, nation_dummies], axis=1)
            processed_df = processed_df.drop(['idInfo.nation', 'nation_simple'], axis=1)


        # if 'deviceInfo.isCrossDomain' in processed_df.columns:
        #     processed_df['deviceInfo.isCrossDomain'] = processed_df['deviceInfo.isCrossDomain'].fillna('UNKNOWN').astype(str)
        #     isCrossDomain_dummies = pd.get_dummies(processed_df['deviceInfo.isCrossDomain'], prefix='isCrossDomain')
        #     processed_df = pd.concat([processed_df, isCrossDomain_dummies], axis=1)
        #     processed_df = processed_df.drop('deviceInfo.isCrossDomain', axis=1)


        # # 5. 处理分类特征
        # categorical_features = [
        #     'linkmanList.0.relationship',
        #     'linkmanList.1.relationship',
        # ]
        #
        # # 只处理实际存在的分类特征
        # categorical_features = [f for f in categorical_features if f in processed_df.columns]
        #
        # for feature in categorical_features:
        #     if feature in processed_df.columns:
        #         # 1. 处理缺失值 (用字符串'UNKNOWN')
        #         processed_df[feature] = processed_df[feature].fillna('UNKNOWN')
        #
        #         # 2. 确保整个列都是字符串类型
        #         processed_df[feature] = processed_df[feature].astype(str)
        #
        #         # 标签编码
        #         if feature not in self.encoders:
        #             self.encoders[feature] = LabelEncoder()
        #
        #             # 获取所有唯一值，并确保 'UNKNOWN' 包含在内，以便编码器能够识别它
        #             unique_values_for_fit = list(processed_df[feature].unique())
        #             if 'UNKNOWN' not in unique_values_for_fit:
        #                 unique_values_for_fit.append('UNKNOWN')
        #
        #             self.encoders[feature].fit(unique_values_for_fit)
        #         else:
        #             # 预测阶段：处理测试集中的未见类别
        #             unseen_labels_set = set(processed_df[feature].unique()) - set(self.encoders[feature].classes_)
        #
        #             if unseen_labels_set:
        #                 total_unseen = len(unseen_labels_set)
        #                 unseen_labels_list = sorted(list(unseen_labels_set))
        #                 display_labels = unseen_labels_list[:10]
        #
        #                 print(
        #                     f"  警告: 特征 '{feature}' 在测试集中发现 {total_unseen} 个未见类别，前10个为: {display_labels}，将替换为 'UNKNOWN'")
        #                 processed_df[feature] = processed_df[feature].replace(list(unseen_labels_set), 'UNKNOWN')
        #
        #         # 转换列
        #         processed_df[f'{feature}_encoded'] = self.encoders[feature].transform(
        #             processed_df[feature]
        #         )
        #
        #         # 移除原始列
        #         processed_df = processed_df.drop(feature, axis=1)

        # 6. 处理数值特征的缺失值
        numerical_features = ['amount', 'idInfo.birthDate', 'idInfo.validityDate',
                              'pictureInfo.0.faceScore', 'term']
        numerical_features = [f for f in numerical_features if f in processed_df.columns]

        for feature in numerical_features:
            if feature in processed_df.columns:
                processed_df[feature] = pd.to_numeric(processed_df[feature], errors='coerce')
                # 再次使用中位数填充，确保在所有转换后依然保持
                processed_df[feature] = processed_df[feature].fillna(processed_df[feature].median())

        # 7. 移除公司名称原始列（如果存在）
        if 'companyInfo.companyName' in processed_df.columns:
            processed_df = processed_df.drop('companyInfo.companyName', axis=1)

        # === 关键修改部分：更鲁棒的最终NaN处理 ===
        # 确保所有数值型列没有NaN，使用中位数填充
        for col in processed_df.select_dtypes(include=np.number).columns:
            if processed_df[col].isnull().any():
                # 使用该列的中位数填充，而不是整个DataFrame的中位数
                median_val = processed_df[col].median()
                if pd.isna(median_val):  # 如果所有值都是NaN，中位数也会是NaN，此时填充0
                    processed_df[col] = processed_df[col].fillna(0)
                else:
                    processed_df[col] = processed_df[col].fillna(median_val)

        # 确保所有非数值型列（如'partner_code'列，虽然不是特征，但也要干净）没有NaN
        for col in processed_df.select_dtypes(exclude=np.number).columns:
            if processed_df[col].isnull().any():
                processed_df[col] = processed_df[col].fillna('UNKNOWN')  # 用一个特殊的字符串填充

        print(f"预处理完成，特征维度: {processed_df.shape[1]}")
        return processed_df

    def _check_company_keywords(self, company_name: str, keywords: List[str]) -> int:
        """
        检查公司名称是否包含关键词

        Args:
            company_name: 公司名称
            keywords: 关键词列表

        Returns:
            1 if 包含关键词, 0 otherwise
        """
        if not company_name or pd.isna(company_name):
            return 0

        company_name_str = str(company_name)
        for keyword in keywords:
            if keyword in company_name_str:
                return 1
        return 0

    def prepare_training_data_global(self, df: pd.DataFrame, fit_scaler: bool = False):
        """
        准备训练/测试数据（全局版本，不区分合作方）
        """
        df = df.reset_index(drop=True)

        # 去掉 partner_code，只保留特征和 label
        feature_columns = [col for col in df.columns if col not in ['partner_code', 'label']]
        self.feature_columns = feature_columns

        X_raw = df[feature_columns].values
        y = df['label'].values

        # 标准化
        if fit_scaler:
            X = self.scaler.fit_transform(X_raw)
        else:
            try:
                X = self.scaler.transform(X_raw)
            except Exception:
                X = self.scaler.fit_transform(X_raw)

        print(f"全局训练数据准备完成: 特征维度: {X.shape[1]}, 样本数: {X.shape[0]}, 正类比例: {y.mean():.3f}")
        return X, y

    def train_all_partners_as_one(self, df, strategy="class_weight"):
        """
        将所有合作方数据合并为一个整体进行训练，只保存AUC更高的整体模型。
        """
        # 准备全局数据
        X, y = self.prepare_training_data_global(df, fit_scaler=True)

        # 数据划分
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 根据策略训练模型
        if strategy == "class_weight":
            model, test_results = self._train_with_class_weight(X_train, y_train, X_test, y_test, partner="Overall")
        elif strategy == "smote":
            model, test_results = self._train_with_smote(X_train, y_train, X_test, y_test, partner="Overall")
        elif strategy == "combine":
            model, test_results = self._train_with_combine_sampling(X_train, y_train, X_test, y_test, partner="Overall")
        elif strategy == "threshold":
            model, test_results = self._train_with_threshold_tuning(X_train, y_train, X_test, y_test, partner="Overall")
        else:
            model, test_results = self._train_baseline(X_train, y_train, X_test, y_test, partner="Overall")

        # 只保存AUC更高的整体模型
        current_auc = test_results.get('test_roc_auc', 0)
        save_model = False

        if "overall" in self.models:
            previous_auc = getattr(self.models["overall"], 'best_auc', 0)
            if current_auc > previous_auc:
                save_model = True
        else:
            save_model = True

        if save_model and model is not None:
            self.models["overall"] = model
            setattr(self.models["overall"], 'best_auc', current_auc)
            setattr(self.models["overall"], 'best_strategy', strategy)

            # 保存模型文件
            save_dir = "ml_result"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "overall_model.pkl")
            joblib.dump(model, save_path)
            print(f"✅ 已保存整体模型: {save_path}")

        return model, test_results

    def _train_baseline(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray, partner: str):
        """
        基线模型训练，不进行特殊不平衡处理。
        """
        print(f"  策略: 基线模型 (无特殊不平衡处理)")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=min(5, len(y_train) // 10),
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # 训练模型
        model.fit(X_train, y_train)

        # 在测试集上评估
        test_results = self._evaluate_on_test_set(model, X_test, y_test)
        self._print_test_results(partner, test_results)

        return model, test_results

    def _train_with_class_weight(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray, partner: str):
        """使用类别权重处理不平衡"""
        print(f"  策略: 类别权重平衡")

        # 计算类别权重
        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        except:
            class_weight_dict = 'balanced'

        print(f"  类别权重: {class_weight_dict}")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=min(5, len(y_train) // 10),
            min_samples_leaf=2,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        )

        # 训练模型
        model.fit(X_train, y_train)

        # 在测试集上评估
        test_results = self._evaluate_on_test_set(model, X_test, y_test)
        self._print_test_results(partner, test_results)

        return model, test_results

    def _train_with_smote(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray, partner: str):
        """使用SMOTE过采样处理不平衡"""
        print(f"  策略: SMOTE过采样")

        try:
            # 检查是否有足够的少数类样本进行SMOTE
            min_samples = min(np.bincount(y_train.astype(int)))
            if min_samples < 2:
                print(f"  少数类样本太少，回退到类别权重方法")
                return self._train_with_class_weight(X_train, y_train, X_test, y_test, partner)

            # 使用SMOTE进行过采样
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples - 1))

            # 在训练集上应用SMOTE
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"  SMOTE后: 原始 {len(y_train)} -> 平衡 {len(y_train_resampled)} 样本")

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            # 训练模型
            model.fit(X_train_resampled, y_train_resampled)

            # 在测试集上评估
            test_results = self._evaluate_on_test_set(model, X_test, y_test)
            self._print_test_results(partner, test_results)

            return model, test_results

        except Exception as e:
            print(f"  SMOTE失败: {e}, 回退到类别权重方法")
            return self._train_with_class_weight(X_train, y_train, X_test, y_test, partner)

    def _train_with_combine_sampling(self, X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray, partner: str):
        """使用SMOTE+Tomek组合采样处理不平衡"""
        print(f"  策略: SMOTE+Tomek组合采样")

        try:
            min_samples = min(np.bincount(y_train.astype(int)))
            if min_samples < 2:
                return self._train_with_class_weight(X_train, y_train, X_test, y_test, partner)

            # 使用SMOTETomek组合方法
            smote_tomek = SMOTETomek(random_state=42)

            # 在训练集上应用组合采样
            X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
            print(f"  组合采样后: 原始 {len(y_train)} -> 处理后 {len(y_train_resampled)} 样本")

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            # 训练模型
            model.fit(X_train_resampled, y_train_resampled)

            # 在测试集上评估
            test_results = self._evaluate_on_test_set(model, X_test, y_test)
            self._print_test_results(partner, test_results)

            return model, test_results

        except Exception as e:
            print(f"  组合采样失败: {e}, 回退到类别权重方法")
            return self._train_with_class_weight(X_train, y_train, X_test, y_test, partner)

    def _train_with_threshold_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray, partner: str):
        """通过调整分类阈值处理不平衡"""
        print(f"  策略: 分类阈值调优")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=min(5, len(y_train) // 10),
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # 训练模型
        model.fit(X_train, y_train)

        # 在测试集上获取预测概率
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 寻找最优阈值（在测试集上寻找）
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0

        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"  最优阈值: {best_threshold:.3f}")

        # 保存最优阈值供预测使用
        setattr(model, 'optimal_threshold', best_threshold)

        # 使用 _evaluate_on_test_set 进行评估（它会自动使用保存的最优阈值）
        test_results = self._evaluate_on_test_set(model, X_test, y_test)
        self._print_test_results(partner, test_results)

        return model, test_results

    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray):
        """评估模型性能"""
        scoring = {
            'roc_auc': 'roc_auc',
            'accuracy': 'accuracy',
            'recall': 'recall',
            'f1': 'f1',
            'precision': 'precision',
        }

        cv_results = cross_validate(
            model, X, y, scoring=scoring,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            error_score='raise'
        )

        return cv_results

    def _print_results(self, partner: str, cv_results: dict):
        """打印结果"""
        print(f"  {partner} 交叉验证结果:")
        print(f"    AUC: {cv_results['test_roc_auc'].mean():.3f} (±{cv_results['test_roc_auc'].std():.3f})")
        print(f"    准确率: {cv_results['test_accuracy'].mean():.3f} (±{cv_results['test_accuracy'].std():.3f})")
        print(f"    查准率: {cv_results['test_precision'].mean():.3f} (±{cv_results['test_precision'].std():.3f})")
        print(f"    召回率: {cv_results['test_recall'].mean():.3f} (±{cv_results['test_recall'].std():.3f})")
        print(f"    F1分数: {cv_results['test_f1'].mean():.3f} (±{cv_results['test_f1'].std():.3f})")

    def compare_imbalance_strategies_all(self, processed_df):
        """
        比较不同的不平衡处理策略（针对所有合作方合并的数据）
        """
        print("=== 比较不同的不平衡处理策略（整体模型） ===\n")

        strategies = ["baseline", "class_weight", "smote", "combine", "threshold"]
        all_results = {}

        for strategy in strategies:
            print(f"\n{'=' * 50}")
            print(f"策略: {strategy.upper()}")
            print(f"{'=' * 50}")

            model, test_results = self.train_all_partners_as_one(processed_df, strategy=strategy)
            all_results[strategy] = test_results

        # 汇总打印结果
        self._summarize_overall_strategy_comparison(all_results)
        return all_results

    def _summarize_overall_strategy_comparison(self, all_results):
        """
        打印整体模型不同策略的比较结果
        """
        print(f"\n{'=' * 80}")
        print("整体策略比较总结")
        print(f"{'=' * 80}")

        sorted_strategies = sorted(all_results.items(),
                                   key=lambda x: (x[1]['test_roc_auc'], x[1]['test_f1'], x[1]['test_recall']),
                                   reverse=True)

        for i, (strategy, metrics) in enumerate(sorted_strategies):
            status = "🏆 最佳" if i == 0 else f"  #{i + 1}"
            print(f"  {status} {strategy:15} "
                  f"AUC: {metrics['test_roc_auc']:.3f}  "
                  f"F1: {metrics['test_f1']:.3f}  "
                  f"查准率: {metrics['test_precision']:.3f}  "
                  f"召回率: {metrics['test_recall']:.3f}  "
                  f"准确率: {metrics['test_accuracy']:.3f}")

    def predict_partner_probabilities(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        预测每个合作方的通过概率
        """
        probabilities = {}

        for partner in self.partners:
            if partner in self.models:
                proba = self.models[partner].predict_proba(X)[:, 1]
                probabilities[partner] = proba

        return probabilities

    def recommend_partners(self, user_features: np.ndarray, k: int = 3,
                           min_probability: float = 0.3) -> List[Tuple[str, float]]:
        """
        为用户推荐合作方
        """
        probabilities = self.predict_partner_probabilities(user_features)

        filtered_partners = [
            (partner, prob[0]) for partner, prob in probabilities.items()
            if prob[0] >= min_probability
        ]

        filtered_partners.sort(key=lambda x: x[1], reverse=True)
        return filtered_partners[:k]

    def _evaluate_on_test_set(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        在测试集上评估模型性能
        """
        # 获取预测概率
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 根据模型是否有最优阈值来决定预测
        if hasattr(model, 'optimal_threshold'):
            threshold = getattr(model, 'optimal_threshold')
            y_pred = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int)

        return self._calculate_metrics(y_test, y_pred, y_pred_proba)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标
        """
        return {
            'test_roc_auc': roc_auc_score(y_true, y_pred_proba),
            'test_accuracy': accuracy_score(y_true, y_pred),
            'test_recall': recall_score(y_true, y_pred, zero_division=0),
            'test_f1': f1_score(y_true, y_pred, zero_division=0),
            'test_precision': precision_score(y_true, y_pred, zero_division=0)
        }

    def _print_test_results(self, partner: str, test_results: Dict[str, float]):
        """
        打印测试集结果
        """
        print(f"  {partner} 测试集性能:")
        print(f"    AUC: {test_results['test_roc_auc']:.4f}")
        print(f"    准确率: {test_results['test_accuracy']:.4f}")
        print(f"    召回率: {test_results['test_recall']:.4f}")
        print(f"    F1: {test_results['test_f1']:.4f}")
        print(f"    精确率: {test_results['test_precision']:.4f}")

    def explain_rf_model_shap(self, partner, model, X, feature_names, top_n=6):
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 拼接目标路径：当前文件的上一级目录 + SHAP/shap_plots
        save_dir = os.path.join(current_dir, "..", "SHAP", "shap_plots")

        os.makedirs(save_dir, exist_ok=True)
        # 1. 创建解释器
        explainer = shap.Explainer(model, X)

        # 2. 计算 SHAP 值
        shap_values = explainer(X, check_additivity=False)

        # 只取正类（比如二分类里类别1），如果是分类任务
        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            shap_values_to_plot = shap_values[:, :, 1]  # 取类别1
        else:
            shap_values_to_plot = shap_values

        # 3. 绘制全局重要性（从上到下展示各特征）
        shap.summary_plot(
            shap_values_to_plot,
            X,
            feature_names=feature_names,
            max_display=top_n,
            show=False
        )

        plt.savefig(os.path.join(save_dir, f"{partner}_rf_shap_summary.png"))
        plt.close()
        print(f"图已保存：{partner}_rf_shap_summary.png")


def main():
    """主函数：演示完整的训练和评估流程"""
    print("=== 贷款分发智能决策模型 - 自动数据加载版本 ===\n")

    try:
        # 1️⃣ 初始化 DataLoader（指定 processed 数据目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_path = os.path.join(project_root, "processed")
        loader = DataLoader(processed_root=processed_path)

        # 2️⃣ 自动划分训练 / 测试日期
        train_start, train_end, test_start, test_end = loader.get_train_test_dates(scheme=2)

        # 3️⃣ 加载训练集数据
        print("\n=== 加载训练数据 ===")
        train_data = loader.load_data_range(train_start, train_end)

        # 4️⃣ 加载测试集数据
        print("\n=== 加载测试数据 ===")
        test_data = loader.load_data_range(test_start, test_end)

        # 5️⃣ 初始化模型
        model = LoanDistributionModel()
        model.partners = train_data["partner_code"].dropna().unique().tolist()

        # 6️⃣ 数据预处理（合并训练和测试数据可选，如果仅用训练集训练，保留训练集即可）
        processed_train_data = model.preprocess_features(train_data)
        processed_test_data = model.preprocess_features(test_data)  # 测试集可用于评估，但整体模型训练只需训练集

        # 7️⃣ 整体训练 + 策略比较
        # 注意：这里直接传入 processed_train_data（整个训练集合并）
        overall_results = model.compare_imbalance_strategies_all(processed_train_data)

        print("\n=== 生成 SHAP 特征重要性图（整体模型） ===")

        # 排除非特征列
        feature_names = [
            c for c in processed_train_data.columns
            if c not in ['partner_code', 'label']
        ]

        # 模型路径（整体模型）
        model_path = os.path.join("ml_result", "overall_model.pkl")

        if not os.path.exists(model_path):
            print(f"⚠️ 未找到整体模型文件：{model_path}")
        else:
            # ✅ 从文件加载整体模型
            rf_model = joblib.load(model_path)
            print(f"✅ 成功加载整体模型：{model_path}")

            # 整体训练数据 X
            X_overall = processed_train_data[feature_names].values

            # === 抽样，最多 2000 条样本 ===
            sample_size = 4000
            if X_overall.shape[0] > sample_size:
                idx = np.random.choice(X_overall.shape[0], sample_size, replace=False)
                X_overall = X_overall[idx]

            # 调用解释函数
            model.explain_rf_model_shap(
                partner="overall",
                model=rf_model,
                X=X_overall,
                feature_names=feature_names,
                top_n=60,
            )

        return model

    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
