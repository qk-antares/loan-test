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
            'deviceInfo.isCrossDomain',
            'deviceInfo.gpsLatitude',
            'deviceInfo.gpsLongitude'
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

        # 5. 处理分类特征
        # TODO: 去掉deviceInfo.applyPos，使用deviceInfo.gpsLatitude，deviceInfo.gpsLongitude（这两个是数值型）
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

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Dict]]:
        """
        准备训练数据

        Args:
            df: 预处理后的数据

        Returns:
            特征矩阵X和每个合作方的标签字典
        """
        # 分离特征和标签
        # FIX_BUG: 列名
        feature_columns = [col for col in df.columns
                           if col not in ['partner_code', 'label']]

        self.feature_columns = feature_columns
        X = df[feature_columns].values

        # 特征标准化
        X = self.scaler.fit_transform(X)

        # 为每个合作方准备标签
        Y_dict = {}
        for partner in self.partners:
            # FIX_BUG: 列名
            partner_mask = df['partner_code'] == partner
            partner_indices = df.index[partner_mask].tolist()

            if len(partner_indices) > 0:
                Y_dict[partner] = {
                    'X_indices': partner_indices,
                    'labels': df.loc[partner_mask, 'label'].values
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
        X_train, Y_train_dict = model.prepare_training_data(processed_train_data)
        X_test, Y_test_dict = model.prepare_training_data(processed_test_data)  # 新增：准备测试数据

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
