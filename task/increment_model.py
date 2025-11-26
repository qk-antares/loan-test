import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight
import os
from typing import Dict, List, Tuple, Any
import warnings
from pathlib import Path
from datetime import datetime
import joblib
from .data_loader import DataLoader

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
            'deviceInfo.osType',
            'deviceInfo.isCrossDomain',
            'deviceInfo.applyPos'
        ]

        self.process_feature_list=[
            # ========== 基础字段 ==========
            'amount',

            # ========== 公司信息 ==========
            'occupation_11.0', 'occupation_13.0', 'occupation_17.0', 'occupation_24.0', 'occupation_27.0',
            'occupation_54.0', 'occupation_90.0', 'occupation_UNKNOWN', 'industry_A', 'industry_C', 'industry_D',
            'industry_E', 'industry_F', 'industry_G', 'industry_H', 'industry_I', 'industry_J', 'industry_K',
            'industry_L', 'industry_M', 'industry_N', 'industry_O', 'industry_P', 'industry_Q',
             'industry_R', 'industry_S', 'industry_UNKNOWN', 'industry_Z',

            # ========== 身份证信息 ==========
            'age_30-45', 'age_45-60', 'age_other', 'age_unknown',
            'validity_1_5y', 'validity_invalid_or_missing', 'validity_over_5y', 'validity_within_1y',

            # ========== 人脸信息 ==========
            'pictureInfo.0.faceScore',

            # ========== 公司：AWJ 类 ==========
            'company_AWJ_公安局', 'company_AWJ_警察', 'company_AWJ_法院', 'company_AWJ_军队',
            'company_AWJ_检察院', 'company_AWJ_城市管理局', 'company_AWJ_律师', 'company_AWJ_记者',
            'company_AWJ_贷款', 'company_AWJ_金融', 'company_AWJ_执行局', 'company_AWJ_监狱',
            'company_AWJ_交通警察', 'company_AWJ_派出所', 'company_AWJ_刑事侦查部门',
            'company_AWJ_交警', 'company_AWJ_刑侦',

            # ========== 公司：RONG 类 ==========
            'company_RONG_学校', 'company_RONG_小学', 'company_RONG_中学', 'company_RONG_大学',
            'company_RONG_学院', 'company_RONG_公检法',

            # ========== 编码字段 ==========
            'degree_encoded', 'income_encoded',

            # ========== 城市特征 ==========
            'city_first_tier', 'city_new_first_tier', 'city_others',

            # ========== 省份特征 ==========
            'province_key_province', 'province_other_province',

            # ========== 银行类别 ==========
            'bank_joint_stock', 'bank_others', 'bank_small_local', 'bank_state_owned',

            # ========== 设备省份特征 ==========
            'device_province_key_province', 'device_province_others',

            # ========== 职业特征 ==========
            'job_1.0', 'job_2.0', 'job_3.0', 'job_UNKNOWN',

            # ========== 居住特征 ==========
            'reside_1.0', 'reside_2.0', 'reside_3.0', 'reside_4.0', 'reside_UNKNOWN',

            # ========== 婚姻特征 ==========
            'marital_1.0', 'marital_2.0', 'marital_3.0', 'marital_UNKNOWN',

            # ========== 借款用途 ==========
            'purpose_CONSUME', 'purpose_UNKNOWN',

            # ========== 客户来源 ==========
            'customerSource_APP', 'customerSource_UNKNOWN', 'customerSource_XCX',

            # ========== 设备系统类型 ==========
            'ostype_ANDROID', 'ostype_IOS', 'ostype_UNKNOWN',

            # ========== 性别特征 ==========
            'gender_F', 'gender_M', 'gender_UNKNOWN',

            # ========== 联系人关系（第 0 位） ==========
            'relationship_CHILDREN', 'relationship_COLLEAGUE', 'relationship_FATHER',
            'relationship_FRIENDS', 'relationship_MATE', 'relationship_MOTHER',
            'relationship_OTHER', 'relationship_PARENTS', 'relationship_RELATIVES',
            'relationship_SIBLING', 'relationship_UNKNOWN',

            # ========== 联系人关系（第 1 位） ==========
            'relationship1_CHILDREN', 'relationship1_COLLEAGUE', 'relationship1_FATHER',
            'relationship1_FRIENDS', 'relationship1_MATE', 'relationship1_MOTHER',
            'relationship1_OTHER', 'relationship1_PARENTS', 'relationship1_RELATIVES',
            'relationship1_SIBLING', 'relationship1_UNKNOWN',

            # ========== 民族特征 ==========
            'nation_han', 'nation_minority', 'nation_unknown',

            # ========== 异地申请特征 ==========
            'isCrossDomain_False', 'isCrossDomain_True', 'isCrossDomain_UNKNOWN',
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
                if province in key_provinces:
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
                    category = 'others'
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
                if prov in key_provinces:
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


        if 'deviceInfo.isCrossDomain' in processed_df.columns:
            processed_df['deviceInfo.isCrossDomain'] = processed_df['deviceInfo.isCrossDomain'].fillna('UNKNOWN').astype(str)
            isCrossDomain_dummies = pd.get_dummies(processed_df['deviceInfo.isCrossDomain'], prefix='isCrossDomain')
            processed_df = pd.concat([processed_df, isCrossDomain_dummies], axis=1)
            processed_df = processed_df.drop('deviceInfo.isCrossDomain', axis=1)

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


        # 6. 处理数值特征的缺失值
        numerical_features = ['amount', 'pictureInfo.0.faceScore']
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

        # === 直接使用类内置的特征列表 self.process_feature_list ===

        full_feature_list = self.process_feature_list

        # 找出 DataFrame 中缺失的列
        missing_cols = [col for col in full_feature_list if col not in processed_df.columns]

        for col in missing_cols:
            processed_df[col] = 0

        print(f"已补全 {len(missing_cols)} 列，缺失列为：{missing_cols}")
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
        保存每日所有策略模型，并维护每个合作方AUC最高的最佳模型
        """

        # 自动生成当天日期
        file_date = datetime.today().strftime("%Y-%m-%d")

        print(f"开始训练模型 - 使用 {strategy} 策略处理类别不平衡（日期: {file_date}）")
        print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

        strategies_results = {}
        trained_models = {}
        skip_partners = ['FZ_CODE', 'LYX_CODE', 'YXG_CODE', 'BHYP_CODE']


        # 为每个合作方训练单独的分类器
        for partner in self.partners:

            if partner in skip_partners:
                print(f"\n{partner}: 被设置为忽略，跳过训练")
                continue

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

                # 保存每日策略模型到 task/ml_result/日期/合作方/策略_model.pkl
                partner_folder = Path("task/ml_result") / file_date / partner
                partner_folder.mkdir(parents=True, exist_ok=True)
                save_path = partner_folder / f"{strategy}_model.pkl"
                joblib.dump(model, save_path)
                print(f"✅ 已保存模型: {save_path}")

                # 维护每个合作方AUC最高的最佳模型
                current_auc = test_results['test_roc_auc']
                if partner in self.models:
                    previous_auc = getattr(self.models[partner], 'best_auc', 0)
                    if current_auc > previous_auc:
                        self.models[partner] = model
                        setattr(self.models[partner], 'best_auc', current_auc)
                        setattr(self.models[partner], 'best_strategy', strategy)
                else:
                    self.models[partner] = model
                    setattr(self.models[partner], 'best_auc', current_auc)
                    setattr(self.models[partner], 'best_strategy', strategy)

        for partner in self.partners:
            if partner in self.models:
                model = self.models[partner]
                partner_folder = Path("ml_result") / file_date / partner
                partner_folder.mkdir(parents=True, exist_ok=True)
                save_path = partner_folder / f"best_model.pkl"
                joblib.dump(model, save_path)
                print(f"✅ 已保存最佳模型: {save_path}")

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
        # self._summarize_strategy_comparison(all_results)
        print(all_results)

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



def main():
    """主函数：演示完整的训练和评估流程"""
    print("=== 贷款分发智能决策模型 - 自动数据加载版本 ===\n")

    try:
        # 1️⃣ 初始化 DataLoader（指定 processed 数据目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_path = os.path.join(project_root, "processed")
        loader = DataLoader(processed_root=processed_path)

        # 2️⃣ 自动划分训练 / 测试日期
        train_start, train_end, test_start, test_end = loader.get_train_test_dates()

        # 3️⃣ 加载训练集数据
        print("\n=== 加载训练数据 ===")
        train_data = loader.load_data_range(train_start, train_end)
        # 👉 打印训练集 jobFunctions 原始值和类型
        # if "jobFunctions" in train_data.columns:
        #     print("\n训练集 jobFunctions 原始 dtype:", train_data["jobFunctions"].dtype)
        #     print("训练集 jobFunctions 前 10 个原始值:", train_data["jobFunctions"].head(10).tolist())

        # 4️⃣ 加载测试集数据
        print("\n=== 加载测试数据 ===")
        test_data = loader.load_data_range(test_start, test_end)

        # 5️⃣ 初始化模型
        model = LoanDistributionModel()
        model.partners = train_data["partner_code"].dropna().unique().tolist()

        # 6️⃣ 数据预处理
        processed_train_data = model.preprocess_features(train_data)
        processed_test_data = model.preprocess_features(test_data)
        processed_test_data = processed_test_data.reindex(columns=processed_train_data.columns, fill_value=0)

        # 7️⃣ 准备训练特征和测试特征
        X_train, Y_train_dict = model.prepare_training_data(processed_train_data, fit_scaler=True)
        X_test, Y_test_dict = model.prepare_training_data(processed_test_data, fit_scaler=False)

        print(f"训练集特征数: {processed_train_data.shape[1]}")
        print(f"测试集特征数: {processed_test_data.shape[1]}")

        # 显示完整列名
        pd.set_option('display.max_columns', None)
        print("完整列名",processed_train_data.columns.tolist())
        print("列数为",len(processed_train_data.columns))
        train_cols = set(processed_train_data.columns)

        test_cols = set(processed_test_data.columns)

        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols

        print("\n=== 测试集中缺少的列（训练集有但测试集没有） ===")
        print(missing_in_test if missing_in_test else "无")

        print("\n=== 测试集多出来的列（测试集有但训练集没有） ===")
        print(extra_in_test if extra_in_test else "无")

        print(processed_train_data.columns)

        # 8️⃣ 比较不同不平衡策略
        comparison_results = model.compare_imbalance_strategies(X_train, Y_train_dict, X_test, Y_test_dict)

        return model

    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
