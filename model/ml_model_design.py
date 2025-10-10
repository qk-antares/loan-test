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

warnings.filterwarnings('ignore')


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
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 如果未指定目录，则自动寻找上级目录中的train和test文件夹
        if train_data_dir is None:
            # 当前脚本在backup文件夹中，上级目录就是项目根目录
            parent_dir = os.path.dirname(current_dir)
            self.train_data_dir = os.path.join(parent_dir, "train")
        else:
            self.train_data_dir = train_data_dir

        if test_data_dir is None:
            parent_dir = os.path.dirname(current_dir)
            self.test_data_dir = os.path.join(parent_dir, "test")
        else:
            self.test_data_dir = test_data_dir

        print(f"训练数据目录: {self.train_data_dir}")
        print(f"测试数据目录: {self.test_data_dir}")

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
            'deviceInfo.isCrossDomain',
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

    def load_data(self, data_type: str = "train") -> pd.DataFrame:
        """
        加载训练或测试数据
        """
        print(f"开始加载{data_type}数据...")

        data_dir = self.train_data_dir if data_type == "train" else self.test_data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 获取所有合作方文件
        partner_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

        if not partner_files:
            raise FileNotFoundError(f"在 {data_dir} 中未找到CSV文件")

        self.partners = [f.replace('.csv', '') for f in partner_files]
        print(f"发现 {len(self.partners)} 个合作方: {self.partners}")

        # 加载所有合作方数据并合并
        all_data = []

        for partner_file in partner_files:
            partner_name = partner_file.replace('.csv', '')
            file_path = os.path.join(data_dir, partner_file)
            partner_df = pd.read_csv(file_path)

            # 添加合作方标识
            partner_df['partner'] = partner_name
            partner_df['partner_label'] = partner_df['label']

            # 移除原始label列
            if 'label' in partner_df.columns:
                partner_df = partner_df.drop('label', axis=1)

            all_data.append(partner_df)
            print(f"  {partner_name}: {len(partner_df)} 条记录")

        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)

        print(f"{data_type}数据合并完成，数据形状: {combined_df.shape}")
        return combined_df

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
        required_cols = ['partner', 'partner_label']
        columns_to_keep = available_features + required_cols

        processed_df = processed_df[columns_to_keep]
        print(f"使用 {len(available_features)} 个特征")

        # 2. 公司名称过滤规则特征
        if 'companyInfo.companyName' in processed_df.columns:
            for rule_name, keywords in self.company_filter_rules.items():
                col_name = f'company_filter_{rule_name}'
                processed_df[col_name] = processed_df['companyInfo.companyName'].apply(
                    lambda x: self._check_company_keywords(x, keywords) if pd.notna(x) else 0
                )

        # 3. 学历编码 (JUNIOR=1, ..., DOCTOR=6)
        if 'degree' in processed_df.columns:
            degree_mapping = {
                'JUNIOR': 1, 'SENIOR': 2, 'COLLEGE': 3,
                'BACHELOR': 4, 'MASTER': 5, 'DOCTOR': 6
            }
            # 确保原始NaN在映射前被处理，或映射后再次处理
            processed_df['degree_encoded'] = processed_df['degree'].map(degree_mapping)
            processed_df['degree_encoded'] = processed_df['degree_encoded'].fillna(0)  # 填充为0，表示未知或最低学历
            processed_df = processed_df.drop('degree', axis=1)

        # 4. 收入等级编码 (A=1, B=2, C=3, D=4)
        if 'income' in processed_df.columns:
            income_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
            processed_df['income_encoded'] = processed_df['income'].map(income_mapping)
            processed_df['income_encoded'] = processed_df['income_encoded'].fillna(0)  # 填充为0，表示未知或最低收入
            processed_df = processed_df.drop('income', axis=1)

        # 5. 处理分类特征
        categorical_features = [
            'bankCardInfo.bankCode', 'city', 'companyInfo.industry',
            'companyInfo.occupation', 'customerSource', 'idInfo.gender',
            'idInfo.nation', 'jobFunctions', 'linkmanList.0.relationship',
            'linkmanList.1.relationship', 'maritalStatus', 'province',
            'purpose', 'resideFunctions', 'deviceInfo.osType',
            'deviceInfo.isCrossDomain', 'deviceInfo.applyPos'
        ]

        # 只处理实际存在的分类特征
        categorical_features = [f for f in categorical_features if f in processed_df.columns]

        for feature in categorical_features:
            if feature in processed_df.columns:
                # 1. 处理缺失值 (用字符串'UNKNOWN')
                processed_df[feature] = processed_df[feature].fillna('UNKNOWN')

                # 2. 确保整个列都是字符串类型
                processed_df[feature] = processed_df[feature].astype(str)

                # 标签编码
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()

                    # 获取所有唯一值，并确保 'UNKNOWN' 包含在内，以便编码器能够识别它
                    unique_values_for_fit = list(processed_df[feature].unique())
                    if 'UNKNOWN' not in unique_values_for_fit:
                        unique_values_for_fit.append('UNKNOWN')

                    self.encoders[feature].fit(unique_values_for_fit)
                else:
                    # 预测阶段：处理测试集中的未见类别
                    unseen_labels_set = set(processed_df[feature].unique()) - set(self.encoders[feature].classes_)

                    if unseen_labels_set:
                        total_unseen = len(unseen_labels_set)
                        unseen_labels_list = sorted(list(unseen_labels_set))
                        display_labels = unseen_labels_list[:10]

                        print(
                            f"  警告: 特征 '{feature}' 在测试集中发现 {total_unseen} 个未见类别，前10个为: {display_labels}，将替换为 'UNKNOWN'")
                        processed_df[feature] = processed_df[feature].replace(list(unseen_labels_set), 'UNKNOWN')

                # 转换列
                processed_df[f'{feature}_encoded'] = self.encoders[feature].transform(
                    processed_df[feature]
                )

                # 移除原始列
                processed_df = processed_df.drop(feature, axis=1)

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

        # 确保所有非数值型列（如'partner'列，虽然不是特征，但也要干净）没有NaN
        for col in processed_df.select_dtypes(exclude=np.number).columns:
            if processed_df[col].isnull().any():
                processed_df[col] = processed_df[col].fillna('UNKNOWN_CATEGORY')  # 用一个特殊的字符串填充

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

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Dict]]:
        """
        准备训练数据

        Args:
            df: 预处理后的数据

        Returns:
            特征矩阵X和每个合作方的标签字典
        """
        # 分离特征和标签
        feature_columns = [col for col in df.columns
                           if col not in ['partner', 'partner_label']]

        self.feature_columns = feature_columns
        X = df[feature_columns].values

        # 特征标准化
        X = self.scaler.fit_transform(X)

        # 为每个合作方准备标签
        Y_dict = {}
        for partner in self.partners:
            partner_mask = df['partner'] == partner
            partner_indices = df.index[partner_mask].tolist()

            if len(partner_indices) > 0:
                Y_dict[partner] = {
                    'X_indices': partner_indices,
                    'labels': df.loc[partner_mask, 'partner_label'].values
                }

        print(f"训练数据准备完成:")
        print(f"  特征维度: {X.shape}")
        print(f"  各合作方数据分布:")
        for partner in self.partners:
            if partner in Y_dict:
                partner_data = Y_dict[partner]
                pass_rate = partner_data['labels'].mean()
                print(f"    {partner}: {len(partner_data['labels'])} 样本, 通过率 {pass_rate:.3f}")
            else:
                print(f"    {partner}: 无数据")

        return X, Y_dict

    def train_models_with_imbalance_handling(self, X: np.ndarray, Y_dict: Dict[str, Dict],
                                             strategy: str = "class_weight") -> Dict[str, Dict]:
        """
        训练模型 - 处理类别不平衡问题

        Args:
            X: 特征矩阵
            Y_dict: 每个合作方的标签字典
            strategy: 处理不平衡的策略

        Returns:
            训练结果字典
        """
        print(f"开始训练模型 - 使用 {strategy} 策略处理类别不平衡...")

        strategies_results = {}
        trained_models = {}

        # 为每个合作方训练单独的分类器
        for partner in self.partners:
            if partner not in Y_dict:
                print(f"\n{partner}: 无训练数据，跳过")
                continue

            print(f"\n训练 {partner} 模型...")

            partner_data = Y_dict[partner]
            X_partner = X[partner_data['X_indices']]
            y_partner = partner_data['labels']

            # 检查数据量和类别分布
            if len(y_partner) < 10:
                print(f"  数据量太少 ({len(y_partner)} 样本), 跳过训练")
                continue

            pos_count = np.sum(y_partner)
            neg_count = len(y_partner) - pos_count
            pos_rate = pos_count / len(y_partner)

            print(f"  数据分布: 正类 {pos_count}, 负类 {neg_count}, 正类率 {pos_rate:.3f}")

            if pos_count < 2:
                print(f"  正类样本太少 ({pos_count} 个), 跳过训练")
                continue

            # 根据策略选择不同的处理方法
            if strategy == "class_weight":
                model, cv_results = self._train_with_class_weight(X_partner, y_partner, partner)
            elif strategy == "smote":
                model, cv_results = self._train_with_smote(X_partner, y_partner, partner)
            elif strategy == "combine":
                model, cv_results = self._train_with_combine_sampling(X_partner, y_partner, partner)
            elif strategy == "threshold":
                model, cv_results = self._train_with_threshold_tuning(X_partner, y_partner, partner)
            else: # 捕获 "baseline" 或其他未定义的策略
                model, cv_results = self._train_baseline(X_partner, y_partner, partner)

            if model is not None:
                trained_models[partner] = model
                strategies_results[partner] = cv_results

        self.models = trained_models
        return strategies_results

    def _train_baseline(self, X: np.ndarray, y: np.ndarray, partner: str):
        """
        基线模型训练，不进行特殊不平衡处理。
        """
        print(f"  策略: 基线模型 (无特殊不平衡处理)")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=min(5, len(y) // 10), # 保持与其他模型一致的参数
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # 交叉验证
        cv_results = self._evaluate_model(model, X, y)
        self._print_results(partner, cv_results)

        # 在全量数据上训练最终模型
        model.fit(X, y)

        return model, cv_results

    def _train_with_class_weight(self, X: np.ndarray, y: np.ndarray, partner: str):
        """使用类别权重处理不平衡"""
        print(f"  策略: 类别权重平衡")

        # 计算类别权重
        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        except:
            class_weight_dict = 'balanced'

        print(f"  类别权重: {class_weight_dict}")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=min(5, len(y) // 10),
            min_samples_leaf=2,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        )

        # 交叉验证
        cv_results = self._evaluate_model(model, X, y)
        self._print_results(partner, cv_results)

        # 在全量数据上训练最终模型
        model.fit(X, y)

        return model, cv_results

    def _train_with_smote(self, X: np.ndarray, y: np.ndarray, partner: str):
        """使用SMOTE过采样处理不平衡"""
        print(f"  策略: SMOTE过采样")

        try:
            # 检查是否有足够的少数类样本进行SMOTE
            min_samples = min(np.bincount(y.astype(int)))
            if min_samples < 2:
                print(f"  少数类样本太少，回退到类别权重方法")
                return self._train_with_class_weight(X, y, partner)

            # 使用SMOTE进行过采样
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples - 1))

            # 分层交叉验证
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = {'auc': [], 'accuracy': [], 'recall': [], 'f1': [], 'precision': []}

            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # 在训练集上应用SMOTE
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)

                # 训练模型
                model_fold = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                model_fold.fit(X_train_resampled, y_train_resampled)

                # 在验证集上评估
                y_pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)

                cv_scores['auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
                cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                cv_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
                cv_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
                cv_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))

            # 计算平均性能
            cv_results = {f'test_{k}': np.array(v) for k, v in cv_scores.items()}
            cv_results['test_roc_auc'] = cv_results.pop('test_auc')

            self._print_results(partner, cv_results)

            # 在全量数据上应用SMOTE并训练最终模型
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"  SMOTE后: 原始 {len(y)} -> 平衡 {len(y_resampled)} 样本")

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_resampled, y_resampled)

            return model, cv_results

        except Exception as e:
            print(f"  SMOTE失败: {e}, 回退到类别权重方法")
            return self._train_with_class_weight(X, y, partner)

    def _train_with_combine_sampling(self, X: np.ndarray, y: np.ndarray, partner: str):
        """使用SMOTE+Tomek组合采样处理不平衡"""
        print(f"  策略: SMOTE+Tomek组合采样")

        try:
            min_samples = min(np.bincount(y.astype(int)))
            if min_samples < 2:
                return self._train_with_class_weight(X, y, partner)

            # 使用SMOTETomek组合方法
            smote_tomek = SMOTETomek(random_state=42)

            # 交叉验证
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = {'auc': [], 'accuracy': [], 'recall': [], 'f1': [], 'precision': []}

            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_fold, y_train_fold)

                model_fold = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                model_fold.fit(X_train_resampled, y_train_resampled)

                y_pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)

                cv_scores['auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
                cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                cv_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
                cv_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
                cv_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))

            cv_results = {f'test_{k}': np.array(v) for k, v in cv_scores.items()}
            cv_results['test_roc_auc'] = cv_results.pop('test_auc')

            self._print_results(partner, cv_results)

            # 训练最终模型
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
            print(f"  组合采样后: 原始 {len(y)} -> 处理后 {len(y_resampled)} 样本")

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_resampled, y_resampled)

            return model, cv_results

        except Exception as e:
            print(f"  组合采样失败: {e}, 回退到类别权重方法")
            return self._train_with_class_weight(X, y, partner)

    def _train_with_threshold_tuning(self, X: np.ndarray, y: np.ndarray, partner: str):
        """通过调整分类阈值处理不平衡"""
        print(f"  策略: 分类阈值调优")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=min(5, len(y) // 10),
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # 交叉验证寻找最优阈值
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_threshold = 0.5
        best_f1 = 0

        cv_scores = {'auc': [], 'accuracy': [], 'recall': [], 'f1': [], 'precision': []}

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model_fold = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model_fold.fit(X_train_fold, y_train_fold)

            # 获取预测概率
            y_pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]

            # 寻找最优阈值
            thresholds = np.arange(0.1, 0.9, 0.05)
            fold_best_f1 = 0
            fold_best_threshold = 0.5

            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                f1 = f1_score(y_val_fold, y_pred_thresh, zero_division=0)
                if f1 > fold_best_f1:
                    fold_best_f1 = f1
                    fold_best_threshold = threshold

            # 使用最优阈值进行预测
            y_pred = (y_pred_proba >= fold_best_threshold).astype(int)

            cv_scores['auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))

            if fold_best_f1 > best_f1:
                best_f1 = fold_best_f1
                best_threshold = fold_best_threshold

        print(f"  最优阈值: {best_threshold:.3f}")

        cv_results = {f'test_{k}': np.array(v) for k, v in cv_scores.items()}
        cv_results['test_roc_auc'] = cv_results.pop('test_auc')

        self._print_results(partner, cv_results)

        # 训练最终模型
        model.fit(X, y)

        # 保存最优阈值供预测使用
        setattr(model, 'optimal_threshold', best_threshold)

        return model, cv_results

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

    def compare_imbalance_strategies(self, X: np.ndarray, Y_dict: Dict[str, Dict]):
        """
        比较不同的不平衡处理策略
        """
        print("=== 比较不同的类别不平衡处理策略 ===\n")

        # 修正：在策略列表中添加 "baseline"
        strategies = ["baseline", "class_weight", "smote", "combine", "threshold"]

        all_results = {}

        for strategy in strategies:
            print(f"\n{'=' * 50}")
            print(f"策略: {strategy.upper()}")
            print(f"{'=' * 50}")

            results = self.train_models_with_imbalance_handling(X, Y_dict, strategy)
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
                    cv_result = results[partner]

                    avg_f1 = cv_result['test_f1'].mean()
                    std_f1 = cv_result['test_f1'].std()
                    avg_auc = cv_result['test_roc_auc'].mean()
                    std_auc = cv_result['test_roc_auc'].std()
                    avg_recall = cv_result['test_recall'].mean()
                    std_recall = cv_result['test_recall'].std()
                    avg_precision = cv_result['test_precision'].mean()
                    std_precision = cv_result['test_precision'].std()
                    avg_accuracy = cv_result['test_accuracy'].mean()
                    std_accuracy = cv_result['test_accuracy'].std()

                    partner_results[strategy] = {
                        'f1': avg_f1, 'f1_std': std_f1,
                        'auc': avg_auc, 'auc_std': std_auc,
                        'recall': avg_recall, 'recall_std': std_recall,
                        'precision': avg_precision, 'precision_std': std_precision,
                        'accuracy': avg_accuracy, 'accuracy_std': std_accuracy
                    }

            if partner_results:
                print(f"\n{partner}:")
                sorted_strategies = sorted(partner_results.items(),
                                           key=lambda x: (x[1]['f1'], x[1]['auc'], x[1]['recall']),
                                           reverse=True)

                for i, (strategy, metrics) in enumerate(sorted_strategies):
                    status = "🏆 最佳" if i == 0 else f"  #{i + 1}"
                    print(f"  {status} {strategy:15} "
                          f"F1: {metrics['f1']:.3f} (±{metrics['f1_std']:.3f})  "
                          f"AUC: {metrics['auc']:.3f} (±{metrics['auc_std']:.3f})  "
                          f"查准率: {metrics['precision']:.3f} (±{metrics['precision_std']:.3f})  "
                          f"召回率: {metrics['recall']:.3f} (±{metrics['recall_std']:.3f})  "
                          f"准确率: {metrics['accuracy']:.3f} (±{metrics['accuracy_std']:.3f})")

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

    def evaluate_on_test(self, test_data_dir: str = None):
        """
        在测试集上评估模型性能
        """
        if test_data_dir is None:
            test_data_dir = self.test_data_dir

        print("=== 在测试集上评估模型性能 ===")

        # 加载测试数据
        test_df = self.load_data("test")
        processed_test_df = self.preprocess_features(test_df)

        # 准备测试数据
        feature_columns = [col for col in processed_test_df.columns
                           if col not in ['partner', 'partner_label']]
        X_test = processed_test_df[feature_columns].values
        X_test = self.scaler.transform(X_test)

        results = {}

        for partner in self.partners:
            if partner not in self.models:
                continue

            partner_mask = processed_test_df['partner'] == partner
            if partner_mask.sum() == 0:
                continue

            X_partner = X_test[partner_mask]
            y_true = processed_test_df.loc[partner_mask, 'partner_label'].values

            model = self.models[partner]
            y_pred_proba = model.predict_proba(X_partner)[:, 1]

            # 使用最优阈值（如果存在）或默认阈值
            if hasattr(model, 'optimal_threshold'):
                threshold = model.optimal_threshold
            else:
                threshold = 0.5

            y_pred = (y_pred_proba >= threshold).astype(int)

            results[partner] = {
                'auc': roc_auc_score(y_true, y_pred_proba),
                'accuracy': accuracy_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'threshold': threshold
            }

        # 打印测试结果
        print("\n测试集性能:")
        for partner, metrics in results.items():
            print(f"{partner}: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}, "
                  f"准确率={metrics['accuracy']:.3f}, 查准率={metrics['precision']:.3f}, "
                  f"召回率={metrics['recall']:.3f}, 阈值={metrics['threshold']:.3f}")

        return results

    def save_model(self, model_path: str = "loan_distribution_model"):
        """保存模型"""
        os.makedirs(model_path, exist_ok=True)

        joblib.dump(self.models, os.path.join(model_path, 'models.pkl'))
        joblib.dump(self.encoders, os.path.join(model_path, 'encoders.pkl'))
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.pkl'))
        joblib.dump(self.partners, os.path.join(model_path, 'partners.pkl'))
        joblib.dump(self.feature_columns, os.path.join(model_path, 'feature_columns.pkl'))

        print(f"模型已保存到: {model_path}")

    def load_model(self, model_path: str = "loan_distribution_model"):
        """加载模型"""
        self.models = joblib.load(os.path.join(model_path, 'models.pkl'))
        self.encoders = joblib.load(os.path.join(model_path, 'encoders.pkl'))
        self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        self.partners = joblib.load(os.path.join(model_path, 'partners.pkl'))
        self.feature_columns = joblib.load(os.path.join(model_path, 'feature_columns.pkl'))

        print(f"模型已从 {model_path} 加载")


def main():
    """主函数：演示完整的训练和评估流程"""
    print("=== 贷款分发智能决策模型 - 重构版本 ===\n")

    try:
        # 1. 初始化模型（不指定路径，使用自动检测）
        model = LoanDistributionModel()

        # 2. 加载和预处理训练数据
        train_data = model.load_data("train")
        processed_train_data = model.preprocess_features(train_data)

        # 3. 准备训练数据
        X_train, Y_dict = model.prepare_training_data(processed_train_data)

        # 4. 比较不同的不平衡处理策略并选择最佳策略训练
        comparison_results = model.compare_imbalance_strategies(X_train, Y_dict)

        # 5. 在测试集上评估模型
        test_results = model.evaluate_on_test()

        # 6. 保存模型（保存到backup目录内）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_path = os.path.join(current_dir, "trained_model")
        model.save_model(model_save_path)

        print(f"\n=== 模型训练和评估完成 ===")
        print(f"模型保存在: {model_save_path}")

        return model

    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
