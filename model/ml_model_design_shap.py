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

        # 3. 学历编码 (JUNIOR=1, ..., DOCTOR=6)
        if 'degree' in processed_df.columns:
            # FIX_BUG: 现在数据预处理脚本已经对学历进行编码，这里只用填充NaN为0
            processed_df['degree_encoded'] = processed_df['degree'].fillna(0)
            processed_df = processed_df.drop('degree', axis=1)

        # 4. 收入等级编码 (A=1, B=2, C=3, D=4)
        if 'income' in processed_df.columns:
            # FIX_BUG: 现在数据预处理脚本已经对收入进行编码，这里只用填充NaN为0
            processed_df['income_encoded'] = processed_df['income'].fillna(0)
            processed_df = processed_df.drop('income', axis=1)


        # 4.5 城市与省份自定义编码（新增部分） ==============================
        # 🏙️ 城市编码：一线=2，新一线=1，其他=0，未知=-1
        if 'city' in processed_df.columns:
            processed_df['city'] = processed_df['city'].fillna('UNKNOWN').astype(str)
            first_tier = ['北京市', '上海市', '广州市', '深圳市']
            new_first_tier = ['杭州市', '南京市', '成都市', '武汉市', '重庆市', '苏州市', '天津市', '西安市', '长沙市','青岛市']
            def encode_city(city):
                if city in first_tier:
                    return 2
                elif city in new_first_tier:
                    return 1
                elif city == 'UNKNOWN':
                    return -1
                else:
                    return 0

            processed_df['city_encoded'] = processed_df['city'].apply(encode_city)
            processed_df = processed_df.drop('city', axis=1)


        # 🗺️ 省份编码：北京上海广东浙江江苏=1，其他=0，未知=-1
        if 'province' in processed_df.columns:
            processed_df['province'] = processed_df['province'].fillna('UNKNOWN').astype(str)
            # 重点省份定义
            key_provinces = ['北京市', '上海市', '广东省', '浙江省', '江苏省']

            def encode_province(province):
                if province in key_provinces:
                    return 1
                elif province == 'UNKNOWN':
                    return -1
                else:
                    return 0

            processed_df['province_encoded'] = processed_df['province'].apply(encode_province)
            processed_df = processed_df.drop('province', axis=1)


        # 银行编码分类（国有大行=2，股份制银行=1，小银行/农商行=0）
        # if 'bankCardInfo.bankCode' in processed_df.columns:
        #     processed_df['bankCardInfo.bankCode'] = processed_df['bankCardInfo.bankCode'].fillna('UNKNOWN').astype(
        #         str)
        #
        #     # 定义各类银行代码
        #     state_owned_banks = ['102', '103', '104', '105', '301', '402']  # 国有大行
        #     joint_stock_banks = ['302', '303', '304', '305', '306', '307', '308', '309', '310']  # 股份制银行
        #     small_local_banks = ['313', '403', '404', '501']  # 小银行、农商行、外资行等
        #
        #     def encode_bank_code(bank_code):
        #         if bank_code in state_owned_banks:
        #             return 2
        #         elif bank_code in joint_stock_banks:
        #             return 1
        #         elif bank_code in small_local_banks:
        #             return 0
        #         elif bank_code == 'UNKNOWN':
        #             return -1
        #         else:
        #             # 未知银行默认归入风险较高的一类
        #             return 0
        #
        #     processed_df['bankCardInfo.bankCode_encoded'] = processed_df['bankCardInfo.bankCode'].apply(
        #         encode_bank_code)
        #     processed_df = processed_df.drop('bankCardInfo.bankCode', axis=1)

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

            # 直接提取前三个字符作为省份
            processed_df['device_province'] = processed_df['deviceInfo.applyPos'].apply(
                lambda x: x[:3] if x != 'UNKNOWN' else 'UNKNOWN')

            # 省份编码规则
            key_provinces = ['北京市', '上海市', '广东省', '浙江省', '江苏省']
            def encode_province(prov):
                if prov == 'UNKNOWN':
                    return -1
                elif prov in key_provinces:
                    return 1
                else:
                    return 0
            processed_df['device_province_encoded'] = processed_df['device_province'].apply(encode_province)
            processed_df = processed_df.drop('device_province', axis=1)

            # 删除原始 applyPos 列
            processed_df = processed_df.drop('deviceInfo.applyPos', axis=1)

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
            processed_df['pictureInfo.0.faceScore'] = pd.to_numeric(processed_df['pictureInfo.0.faceScore'],
                                                                    errors='coerce')
            def encode_face_score(score):
                if 90 <score <= 100:
                    return 2
                elif 80 <= score <= 90:
                    return 1
                elif score < 80:
                    return 0
                else:
                    return -1
            processed_df['faceScore_group'] = processed_df['pictureInfo.0.faceScore'].apply(encode_face_score)
            print("脸部识别分数总分布：")
            print(processed_df['faceScore_group'].value_counts())
            processed_df = processed_df.drop('pictureInfo.0.faceScore', axis=1)


        if 'idInfo.validityDate' in processed_df.columns:
            def encode_validity_days(days_left):
                if days_left is None:
                    return -1  # 无效或缺失
                elif days_left <= 365:
                    return 0  # 1年以内
                elif days_left <= 365 * 5:
                    return 1  # 1-5年
                else:
                    return 2  # 超过5年或长期有效

            processed_df['validity_encoded'] = processed_df['idInfo.validityDate'].apply(encode_validity_days)
            # 打印总的类别数量
            print("有效期编码总分布：")
            print(processed_df['validity_encoded'].value_counts())
            processed_df = processed_df.drop('idInfo.validityDate', axis=1)


        # jobFunctions
        if 'jobFunctions' in processed_df.columns:
            processed_df['jobFunctions'] = processed_df['jobFunctions'].fillna('UNKNOWN').astype(str)
            job_dummies = pd.get_dummies(processed_df['jobFunctions'], prefix='job')
            processed_df = pd.concat([processed_df, job_dummies], axis=1)
            processed_df = processed_df.drop('jobFunctions', axis=1)

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

    def prepare_training_data(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, Dict[str, Dict]]:
        """
        准备训练/测试数据
        Args:
            df: 预处理后的 DataFrame
            fit_scaler: 如果 True 则对 self.scaler 执行 fit_transform（用于训练集）
                        如果 False 则对 self.scaler 执行 transform（用于测试集/推理）
        Returns:
            X (np.ndarray)，Y_dict 每个 partner 对应 {'X_indices': [positions], 'labels': np.array}
        """
        # 确保行号是连续整数，与 X 的行号对齐
        df = df.reset_index(drop=True)

        feature_columns = [col for col in df.columns if col not in ['partner_code', 'label']]
        self.feature_columns = feature_columns

        X_raw = df[feature_columns].values

        # 标准化：训练集 fit_scaler=True，测试集 fit_scaler=False
        if fit_scaler:
            X = self.scaler.fit_transform(X_raw)
        else:
            # 如果 scaler 尚未 fit（例如误用），退回到 fit
            try:
                X = self.scaler.transform(X_raw)
            except Exception:
                X = self.scaler.fit_transform(X_raw)

        # 为每个合作方准备标签和对应的行位置（位置为 0..n-1）
        Y_dict = {}
        for partner in self.partners:
            partner_mask = (df['partner_code'] == partner)
            positions = df.index[partner_mask].tolist()  # index 已 reset，是位置
            if len(positions) > 0:
                Y_dict[partner] = {
                    'X_indices': positions,
                    'labels': df.loc[partner_mask, 'label'].values
                }

        print(f"训练数据准备完成: 特征维度: {X.shape}, 合作方数: {len(Y_dict)}")
        for partner, pdata in Y_dict.items():
            labels = pdata['labels']
            print(f"  {partner}: {len(labels)} 样本, 通过率 {labels.mean():.3f}")

        return X, Y_dict


    def train_models_with_imbalance_handling(self, X_train: np.ndarray, Y_train_dict: Dict[str, Dict],
                                             X_test: np.ndarray, Y_test_dict: Dict[str, Dict],
                                             strategy: str = "class_weight") -> Dict[str, Dict]:
        """
        训练模型 - 处理类别不平衡问题（直接在训练集训练，测试集评估）
        """
        print(f"开始训练模型 - 使用 {strategy} 策略处理类别不平衡...")
        print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

        strategies_results = {}
        trained_models = {}

        # 为每个合作方训练单独的分类器
        for partner in self.partners:
            if partner not in Y_train_dict or partner not in Y_test_dict:
                print(f"\n{partner}: 无训练或测试数据，跳过")
                continue

            print(f"\n训练 {partner} 模型...")

            # 获取训练数据
            train_data = Y_train_dict[partner]
            X_train_partner = X_train[train_data['X_indices']]
            y_train_partner = train_data['labels']

            # 获取测试数据
            test_data = Y_test_dict[partner]
            X_test_partner = X_test[test_data['X_indices']]
            y_test_partner = test_data['labels']

            # 检查数据量
            if len(y_train_partner) < 10 or len(y_test_partner) < 5:
                print(f"  数据量太少 (训练: {len(y_train_partner)}, 测试: {len(y_test_partner)}), 跳过训练")
                continue

            pos_count_train = np.sum(y_train_partner)
            pos_count_test = np.sum(y_test_partner)

            if pos_count_train < 2 or pos_count_test < 1:
                print(f"  正类样本太少 (训练: {pos_count_train}, 测试: {pos_count_test}), 跳过训练")
                continue

            print(f"  训练集: {len(y_train_partner)} 样本, 正类 {pos_count_train}")
            print(f"  测试集: {len(y_test_partner)} 样本, 正类 {pos_count_test}")

            # 根据策略选择不同的处理方法
            if strategy == "class_weight":
                model, test_results = self._train_with_class_weight(X_train_partner, y_train_partner,
                                                                    X_test_partner, y_test_partner, partner)
            elif strategy == "smote":
                model, test_results = self._train_with_smote(X_train_partner, y_train_partner,
                                                             X_test_partner, y_test_partner, partner)
            elif strategy == "combine":
                model, test_results = self._train_with_combine_sampling(X_train_partner, y_train_partner,
                                                                        X_test_partner, y_test_partner, partner)
            elif strategy == "threshold":
                model, test_results = self._train_with_threshold_tuning(X_train_partner, y_train_partner,
                                                                        X_test_partner, y_test_partner, partner)
            else:  # 捕获 "baseline" 或其他未定义的策略
                model, test_results = self._train_baseline(X_train_partner, y_train_partner,
                                                           X_test_partner, y_test_partner, partner)

            if model is not None:
                # 保存当前策略的结果
                strategies_results[partner] = test_results

                # 只保存AUC更高的模型
                current_auc = test_results['test_roc_auc']

                if partner in self.models:
                    previous_auc = getattr(self.models[partner], 'best_auc', 0)
                    if current_auc > previous_auc:
                        trained_models[partner] = model
                        setattr(trained_models[partner], 'best_auc', current_auc)
                        setattr(trained_models[partner], 'best_strategy', strategy)
                else:
                    trained_models[partner] = model
                    setattr(trained_models[partner], 'best_auc', current_auc)
                    setattr(trained_models[partner], 'best_strategy', strategy)

        # 更新模型字典（只更新有改进的模型）
        self.models.update(trained_models)

        save_dir = "ml_result"
        os.makedirs(save_dir, exist_ok=True)  # 如果文件夹不存在则创建

        for partner, model in trained_models.items():
            save_path = os.path.join(save_dir, f"{partner}_model.pkl")
            joblib.dump(model, save_path)
            print(f"✅ 已保存模型: {save_path}")

        return strategies_results

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

    def compare_imbalance_strategies(self, X_train: np.ndarray, Y_train_dict: Dict[str, Dict],
                                     X_test: np.ndarray, Y_test_dict: Dict[str, Dict]):
        """
        比较不同的不平衡处理策略
        """
        print("=== 比较不同的类别不平衡处理策略 ===\n")

        strategies = ["baseline", "class_weight", "smote", "combine", "threshold"]
        all_results = {}

        for strategy in strategies:
            print(f"\n{'=' * 50}")
            print(f"策略: {strategy.upper()}")
            print(f"{'=' * 50}")

            results = self.train_models_with_imbalance_handling(X_train, Y_train_dict, X_test, Y_test_dict, strategy)
            all_results[strategy] = results

        # 总结比较结果
        self._summarize_strategy_comparison(all_results)

        return all_results

    def _summarize_strategy_comparison(self, all_results: Dict[str, Dict]):
        """总结策略比较结果"""
        print(f"\n{'=' * 80}")
        print("策略比较总结")
        print(f"{'=' * 80}")

        for partner in self.partners:
            partner_results = {}

            for strategy, results in all_results.items():
                if partner in results:
                    test_result = results[partner]

                    partner_results[strategy] = {
                        'auc': test_result['test_roc_auc'],
                        'f1': test_result['test_f1'],
                        'recall': test_result['test_recall'],
                        'precision': test_result['test_precision'],
                        'accuracy': test_result['test_accuracy']
                    }

            if partner_results:
                print(f"\n{partner}:")
                sorted_strategies = sorted(partner_results.items(),
                                           key=lambda x: (x[1]['auc'], x[1]['f1'], x[1]['recall']),
                                           reverse=True)

                for i, (strategy, metrics) in enumerate(sorted_strategies):
                    status = "🏆 最佳" if i == 0 else f"  #{i + 1}"
                    print(f"  {status} {strategy:15} "
                          f"AUC: {metrics['auc']:.3f}  "
                          f"F1: {metrics['f1']:.3f}  "
                          f"查准率: {metrics['precision']:.3f}  "
                          f"召回率: {metrics['recall']:.3f}  "
                          f"准确率: {metrics['accuracy']:.3f}")

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

        # 6️⃣ 数据预处理
        processed_train_data = model.preprocess_features(train_data)
        processed_test_data = model.preprocess_features(test_data)

        # 7️⃣ 准备训练特征和测试特征
        X_train, Y_train_dict = model.prepare_training_data(processed_train_data, fit_scaler=True)
        X_test, Y_test_dict = model.prepare_training_data(processed_test_data, fit_scaler=False)

        print(processed_train_data.columns)

        # 8️⃣ 比较不同不平衡策略
        comparison_results = model.compare_imbalance_strategies(X_train, Y_train_dict, X_test, Y_test_dict)

        # partner = "LXJ_CODE"  # 根据你的数据修改
        #
        # if partner not in processed_train_data['partner_code'].unique():
        #     print(f"⚠️ 数据中没有 {partner} 合作方，跳过规则分析")
        # else:
        #     # 取出该合作方的数据
        #     df_partner = processed_train_data[processed_train_data['partner_code'] == partner].copy()
        #
        #     # 统计总体通过率
        #     total_samples = len(df_partner)
        #     total_passed = df_partner['label'].sum()  # label=1表示通过
        #     overall_pass_rate = total_passed / total_samples if total_samples > 0 else 0
        #     print(f"\n=== {partner} 总体通过率 ===")
        #     print(f"样本总数: {total_samples}")
        #     print(f"真实通过数: {total_passed}")
        #     print(f"总体通过率: {overall_pass_rate:.2%}")
        #
        #     # 定义规则函数
        #     # 定义特征偏好方向
        #     feature_directions = {
        #         'bankCardInfo.bankCode_encoded': 'low',
        #         'idInfo.birthDate': 'high',  # 数据处理后为年龄
        #         'degree_encoded': 'high',
        #         'maritalStatus_encoded': 'low',
        #         'income_encoded': 'high',
        #         'idInfo.gender_encoded': 'low',
        #         'companyInfo.industry_encoded': 'low',
        #         'companyInfo.occupation_encoded': 'low',
        #         'deviceInfo.isCrossDomain_encoded': 'high'
        #     }
        #
        #     # feature_directions = {
        #     #     'bankCardInfo.bankCode_encoded': 'low',
        #     #     'degree_encoded': 'high',
        #     #     'pictureInfo.0.faceScore': 'high',
        #     #     'idInfo.birthDate': 'high',  # 数据处理后为年龄
        #     #     # 'degree_encoded': 'high',
        #     #     # 'maritalStatus_encoded': 'low',
        #     #     'income_encoded': 'high',
        #     #     'idInfo.gender_encoded': 'low',
        #     #     # 'companyInfo.industry_encoded': 'low',
        #     #     # 'companyInfo.occupation_encoded': 'low'
        #     # }
        #
        #     # 定义规则函数，基于均值判断
        #     def is_pass_candidate(row, df, feature_directions):
        #         for feature, direction in feature_directions.items():
        #             if feature not in df.columns:
        #                 continue
        #             mean_val = df[feature].mean()
        #             if direction == 'high' and row[feature] < mean_val:
        #                 return False
        #             if direction == 'low' and row[feature] > mean_val:
        #                 return False
        #         return True
        #
        #     # 应用规则
        #     df_partner['rule_match'] = df_partner.apply(is_pass_candidate, axis=1,
        #                                                 args=(df_partner, feature_directions))
        #
        #     # 统计匹配情况
        #     df_matched = df_partner[df_partner['rule_match']]
        #     total_matched = len(df_matched)
        #     correct_matches = df_matched['label'].sum()
        #     accuracy = correct_matches / total_matched if total_matched > 0 else 0
        #
        #     print(f"\n=== {partner} 规则匹配分析结果 ===")
        #     print(f"匹配的样本数: {total_matched}")
        #     print(f"正确预测数: {correct_matches}")
        #     print(f"规则匹配准确率: {accuracy:.2%}")

        print("\n=== 生成 SHAP 特征重要性图 ===")
        feature_names = [
            c for c in processed_train_data.columns
            if c not in ['partner_code', 'label']
        ]

        for partner in model.partners:
            model_path = os.path.join("ml_result", f"{partner}_model.pkl")
            print(model_path)
            if not os.path.exists(model_path):
                print(f"⚠️ 跳过 {partner}（未找到模型文件）")
                continue

            # ✅ 从文件加载模型
            rf_model = joblib.load(model_path)
            print(f"✅ 成功加载模型：{model_path}")

            # ✅ 获取该合作方的训练数据
            if partner in Y_train_dict and 'X_indices' in Y_train_dict[partner]:
                X_partner = X_train[Y_train_dict[partner]['X_indices']]
            else:
                print(f"⚠️ 找不到 {partner} 的 X_indices，跳过")
                continue

            # === 抽样，最多 2000 条样本 ===
            sample_size = 2000
            if X_partner.shape[0] > sample_size:
                idx = np.random.choice(X_partner.shape[0], sample_size, replace=False)
                X_partner = X_partner[idx]

            # 调用解释函数
            model.explain_rf_model_shap(
                partner=partner,
                model=rf_model,
                X=X_partner,
                feature_names=feature_names,
                top_n=40,
            )
        return model

    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
