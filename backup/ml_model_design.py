"""
贷款分发智能决策模型设计

业务场景：
- 平台作为中介，将用户贷款请求分发到多个下游合作方
- 每个合作方有自己的审核规则（黑盒）
- 平台目标：在数量约束下最大化通过率和收益
- 数据：11个合作方的历史审核结果

模型选择：Multi-label Classification + Ranking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, fbeta_score, hamming_loss, accuracy_score, make_scorer, precision_recall_curve
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import re
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class LoanDistributionModel:
    """
    贷款分发智能决策模型
    
    主要功能：
    1. 多标签分类：预测每个合作方的通过概率
    2. 智能排序：基于通过概率和历史收益进行排序
    3. 约束分发：在数量约束下选择最优合作方组合
    """
    
    def __init__(self, processed_data_dir: str = "processed_data"):
        """
        初始化模型
        
        Args:
            processed_data_dir: 处理后数据目录
        """
        self.processed_data_dir = processed_data_dir
        self.partners = []  # 合作方列表
        self.feature_columns = []  # 特征列名
        self.models = {}  # 每个合作方的模型
        self.encoders = {}  # 编码器
        self.scaler = StandardScaler()
        self.company_filter_rules = self._init_company_filters()
        
    def _init_company_filters(self) -> Dict[str, List[str]]:
        """
        初始化公司名称过滤规则
        
        Returns:
            过滤规则字典
        """
        return {
            'AWJ': ['公安局', '警察', '法院', '军队', '检察院', '城市管理局', '律师', '记者', '贷款', '金融', '执行局', '监狱', '交通警察', '派出所', '刑事侦查部门', '交警', '刑侦'],
            'RONG': ['学校', '小学', '中学', '大学', '学院', '公检法'],
        }
    
    def load_and_combine_data(self) -> pd.DataFrame:
        """
        加载并合并所有合作方数据
        
        Returns:
            合并后的数据集，每行包含一个样本及其对应合作方的标签
        """
        print("开始加载合作方数据...")
        
        # 获取所有合作方文件
        partner_files = [f for f in os.listdir(self.processed_data_dir) 
                        if f.endswith('.csv')]
        
        self.partners = [f.replace('.csv', '') for f in partner_files]
        print(f"发现 {len(self.partners)} 个合作方: {self.partners}")
        
        # 加载所有合作方数据并合并
        all_data = []
        
        for partner_file in partner_files:
            partner_name = partner_file.replace('.csv', '')
            partner_df = pd.read_csv(os.path.join(self.processed_data_dir, partner_file))
            
            # 添加合作方标识
            partner_df['partner'] = partner_name
            partner_df['partner_label'] = partner_df['label']
            
            all_data.append(partner_df)
            print(f"  {partner_name}: {len(partner_df)} 条记录")
        
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 移除原始label列
        if 'label' in combined_df.columns:
            combined_df = combined_df.drop('label', axis=1)
        
        # 获取特征列名（排除partner和partner_label）
        self.feature_columns = [col for col in combined_df.columns 
                               if col not in ['partner', 'partner_label']]
        
        print(f"合并完成，数据形状: {combined_df.shape}")
        print(f"特征列数: {len(self.feature_columns)}")
        print(f"总样本数: {len(combined_df)}")
        
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
        
        # 1. 公司名称过滤规则特征
        if 'companyInfo.companyName' in processed_df.columns:
            for rule_name, keywords in self.company_filter_rules.items():
                col_name = f'company_filter_{rule_name}'
                processed_df[col_name] = processed_df['companyInfo.companyName'].apply(
                    lambda x: self._check_company_keywords(x, keywords) if pd.notna(x) else 0
                )
            # 移除原始公司名称列
            processed_df = processed_df.drop('companyInfo.companyName', axis=1)
        
        # 2. 学历编码 (JUNIOR=1, ..., DOCTOR=6)
        if 'degree' in processed_df.columns:
            degree_mapping = {
                'JUNIOR': 1, 'SENIOR': 2, 'COLLEGE': 3, 
                'BACHELOR': 4, 'MASTER': 5, 'DOCTOR': 6
            }
            processed_df['degree_encoded'] = processed_df['degree'].map(degree_mapping)
            processed_df = processed_df.drop('degree', axis=1)
        
        # 3. 收入等级编码 (A=1, B=2, C=3, D=4)
        if 'income' in processed_df.columns:
            income_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
            processed_df['income_encoded'] = processed_df['income'].map(income_mapping)
            processed_df = processed_df.drop('income', axis=1)
        
        # 4. 处理分类特征
        categorical_features = [
            'bankCardInfo.bankCode', 'city', 'companyInfo.industry', 
            'companyInfo.occupation', 'customerSource', 'idInfo.gender',
            'idInfo.nation', 'jobFunctions', 'linkmanList.0.relationship',
            'linkmanList.1.relationship', 'maritalStatus', 'province',
            'purpose', 'resideFunctions'
        ]
        
        for feature in categorical_features:
            if feature in processed_df.columns:
                # 处理缺失值
                processed_df[feature] = processed_df[feature].fillna('UNKNOWN')
                
                # 标签编码
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                    processed_df[f'{feature}_encoded'] = self.encoders[feature].fit_transform(
                        processed_df[feature].astype(str)
                    )
                else:
                    # 处理新的类别
                    try:
                        processed_df[f'{feature}_encoded'] = self.encoders[feature].transform(
                            processed_df[feature].astype(str)
                        )
                    except ValueError:
                        # 新类别用-1表示
                        processed_df[f'{feature}_encoded'] = processed_df[feature].apply(
                            lambda x: self.encoders[feature].transform([x])[0] 
                            if x in self.encoders[feature].classes_ else -1
                        )
                
                # 移除原始列
                processed_df = processed_df.drop(feature, axis=1)
        
        # 5. 处理数值特征的缺失值
        numerical_features = ['amount', 'idInfo.birthDate', 'idInfo.validityDate', 'pictureInfo.0.faceScore', 'term']
        for feature in numerical_features:
            if feature in processed_df.columns:
                processed_df[feature] = pd.to_numeric(processed_df[feature], errors='coerce')
                processed_df[feature] = processed_df[feature].fillna(processed_df[feature].median())
        
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
        if not company_name:
            return 0
        
        for keyword in keywords:
            if keyword in company_name:
                return 1
        return 0
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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
        
        X = df[feature_columns].values
        
        # 特征标准化
        X = self.scaler.fit_transform(X)
        
        # 为每个合作方准备标签
        Y_dict = {}
        for partner in self.partners:
            partner_mask = df['partner'] == partner
            partner_indices = df.index[partner_mask].tolist()
            Y_dict[partner] = {
                'X_indices': partner_indices,
                'labels': df.loc[partner_mask, 'partner_label'].values
            }
        
        print(f"训练数据准备完成:")
        print(f"  特征维度: {X.shape}")
        print(f"  各合作方数据分布:")
        for partner in self.partners:
            partner_data = Y_dict[partner]
            pass_rate = partner_data['labels'].mean()
            print(f"    {partner}: {len(partner_data['labels'])} 样本, 通过率 {pass_rate:.3f}")
        
        return X, Y_dict
    
    def train_models_with_imbalance_handling(self, X: np.ndarray, Y_dict: Dict[str, np.ndarray], 
                                           strategy: str = "class_weight"):
        """
        训练模型 - 处理类别不平衡问题
        
        Args:
            X: 特征矩阵
            Y_dict: 每个合作方的标签字典
            strategy: 处理不平衡的策略
                - "class_weight": 使用类别权重
                - "smote": 使用SMOTE过采样
                - "combine": 使用SMOTE+Tomek组合方法
                - "threshold": 调整分类阈值
        """
        print(f"开始训练模型 - 使用 {strategy} 策略处理类别不平衡...")
        
        strategies_results = {}
        
        # 为每个合作方训练单独的分类器
        for partner in self.partners:
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
            
            if pos_count < 5:
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
            else:
                model, cv_results = self._train_baseline(X_partner, y_partner, partner)
            
            if model is not None:
                self.models[partner] = model
                strategies_results[partner] = cv_results
        
        return strategies_results
    
    def _train_with_class_weight(self, X: np.ndarray, y: np.ndarray, partner: str):
        """使用类别权重处理不平衡"""
        print(f"  策略: 类别权重平衡")
        
        # 计算类别权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print(f"  类别权重: {class_weight_dict}")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=min(5, len(y) // 10),
            min_samples_leaf=2,
            class_weight=class_weight_dict,  # 关键：使用类别权重
            random_state=42,
            n_jobs=-1
        )
        
        # 交叉验证
        cv_results = self._evaluate_model(model, X, y)
        
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
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))
            
            # 分层交叉验证，每折内部使用SMOTE
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
                
                from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
                cv_scores['auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
                cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                cv_scores['recall'].append(recall_score(y_val_fold, y_pred))
                cv_scores['f1'].append(f1_score(y_val_fold, y_pred))
                cv_scores['precision'].append(precision_score(y_val_fold, y_pred))
            
            # 计算平均性能
            cv_results = {f'test_{k}': np.array(v) for k, v in cv_scores.items()}
            # 统一AUC键名
            if 'test_auc' in cv_results:
                cv_results['test_roc_auc'] = cv_results.pop('test_auc')
            
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
            smote_tomek = SMOTETomek(
                smote=SMOTE(random_state=42, k_neighbors=min(5, min_samples-1)),
                random_state=42
            )
            
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
                
                from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
                cv_scores['auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
                cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                cv_scores['recall'].append(recall_score(y_val_fold, y_pred))
                cv_scores['f1'].append(f1_score(y_val_fold, y_pred))
                cv_scores['precision'].append(precision_score(y_val_fold, y_pred))
            
            cv_results = {f'test_{k}': np.array(v) for k, v in cv_scores.items()}
            # 统一AUC键名
            if 'test_auc' in cv_results:
                cv_results['test_roc_auc'] = cv_results.pop('test_auc')
            
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
                from sklearn.metrics import f1_score
                f1 = f1_score(y_val_fold, y_pred_thresh)
                if f1 > fold_best_f1:
                    fold_best_f1 = f1
                    fold_best_threshold = threshold
            
            # 使用最优阈值进行预测
            y_pred = (y_pred_proba >= fold_best_threshold).astype(int)
            
            from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
            cv_scores['auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred))
            
            if fold_best_f1 > best_f1:
                best_f1 = fold_best_f1
                best_threshold = fold_best_threshold
        
        print(f"  最优阈值: {best_threshold:.3f}")
        
        cv_results = {f'test_{k}': np.array(v) for k, v in cv_scores.items()}
        # 统一AUC键名
        if 'test_auc' in cv_results:
            cv_results['test_roc_auc'] = cv_results.pop('test_auc')
        
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
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        )
        
        return cv_results
    
    def _print_results(self, partner: str, cv_results: dict):
        """打印结果"""
        print(f"  {partner} 交叉验证结果:")
        auc_key = 'test_roc_auc' if 'test_roc_auc' in cv_results else 'test_auc'
        print(f"    AUC: {cv_results[auc_key].mean():.3f} (±{cv_results[auc_key].std():.3f})")
        print(f"    准确率: {cv_results['test_accuracy'].mean():.3f} (±{cv_results['test_accuracy'].std():.3f})")
        print(f"    查准率: {cv_results['test_precision'].mean():.3f} (±{cv_results['test_precision'].std():.3f})")
        print(f"    召回率: {cv_results['test_recall'].mean():.3f} (±{cv_results['test_recall'].std():.3f})")
        print(f"    F1分数: {cv_results['test_f1'].mean():.3f} (±{cv_results['test_f1'].std():.3f})")
    
    def compare_imbalance_strategies(self, X: np.ndarray, Y_dict: Dict[str, np.ndarray]):
        """
        比较不同的不平衡处理策略
        
        Args:
            X: 特征矩阵
            Y_dict: 每个合作方的标签字典
        """
        print("=== 比较不同的类别不平衡处理策略 ===\n")
        
        strategies = [
            "baseline",
            "class_weight", 
            "smote",
            "combine",
            "threshold"
        ]
        
        all_results = {}
        
        for strategy in strategies:
            print(f"\n{'='*50}")
            print(f"策略: {strategy.upper()}")
            print(f"{'='*50}")
            
            if strategy == "baseline":
                results = self._train_baseline_all(X, Y_dict)
            else:
                results = self.train_models_with_imbalance_handling(X, Y_dict, strategy)
            
            all_results[strategy] = results
        
        # 总结比较结果
        self._summarize_strategy_comparison(all_results)
        
        return all_results
    
    def _train_baseline_all(self, X: np.ndarray, Y_dict: Dict[str, np.ndarray]):
        """训练基线模型（原始方法）"""
        results = {}
        
        for partner in self.partners:
            partner_data = Y_dict[partner]
            X_partner = X[partner_data['X_indices']]
            y_partner = partner_data['labels']
            
            if len(y_partner) < 10 or np.sum(y_partner) < 5:
                continue
            
            model, cv_results = self._train_baseline(X_partner, y_partner, partner)
            if model is not None:
                results[partner] = cv_results
        
        return results
    
    def _train_baseline(self, X: np.ndarray, y: np.ndarray, partner: str):
        """基线模型训练"""
        print(f"  策略: 基线模型（无特殊处理）")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=min(5, len(y) // 10),
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        cv_results = self._evaluate_model(model, X, y)
        self._print_results(partner, cv_results)
        
        model.fit(X, y)
        return model, cv_results
    
    def _summarize_strategy_comparison(self, all_results: Dict[str, Dict]):
        """总结策略比较结果"""
        print(f"\n{'='*80}")
        print("策略比较总结")
        print(f"{'='*80}")
        
        # 为每个合作方找到最佳策略
        for partner in self.partners:
            partner_results = {}
            
            for strategy, results in all_results.items():
                if partner in results:
                    cv_result = results[partner]
                    # 获取所有评估指标的均值和标准差
                    avg_f1 = cv_result['test_f1'].mean() if 'test_f1' in cv_result else 0
                    std_f1 = cv_result['test_f1'].std() if 'test_f1' in cv_result else 0
                    
                    if 'test_auc' in cv_result:
                        avg_auc = cv_result['test_auc'].mean()
                        std_auc = cv_result['test_auc'].std()
                    else:
                        avg_auc = cv_result['test_roc_auc'].mean()
                        std_auc = cv_result['test_roc_auc'].std()
                    
                    avg_recall = cv_result['test_recall'].mean()
                    std_recall = cv_result['test_recall'].std()
                    
                    avg_precision = cv_result['test_precision'].mean() if 'test_precision' in cv_result else 0
                    std_precision = cv_result['test_precision'].std() if 'test_precision' in cv_result else 0
                    
                    avg_accuracy = cv_result['test_accuracy'].mean() if 'test_accuracy' in cv_result else 0
                    std_accuracy = cv_result['test_accuracy'].std() if 'test_accuracy' in cv_result else 0
                    
                    partner_results[strategy] = {
                        'f1': avg_f1, 'f1_std': std_f1,
                        'auc': avg_auc, 'auc_std': std_auc,
                        'recall': avg_recall, 'recall_std': std_recall,
                        'precision': avg_precision, 'precision_std': std_precision,
                        'accuracy': avg_accuracy, 'accuracy_std': std_accuracy
                    }
            
            if partner_results:
                print(f"\n{partner}:")
                # 按照F1, AUC, Recall, Precision, ACC的优先级排序
                sorted_strategies = sorted(partner_results.items(), 
                                         key=lambda x: (x[1]['f1'], x[1]['auc'], x[1]['recall'], 
                                                       x[1]['precision'], x[1]['accuracy']), 
                                         reverse=True)
                
                for i, (strategy, metrics) in enumerate(sorted_strategies):
                    status = "🏆 最佳" if i == 0 else f"  #{i+1}"
                    print(f"  {status} {strategy:15} F1: {metrics['f1']:.3f} (±{metrics['f1_std']:.3f})  AUC: {metrics['auc']:.3f} (±{metrics['auc_std']:.3f})  查准率: {metrics['precision']:.3f} (±{metrics['precision_std']:.3f})  召回率: {metrics['recall']:.3f} (±{metrics['recall_std']:.3f})  准确率: {metrics['accuracy']:.3f} (±{metrics['accuracy_std']:.3f})")
    
    def train_models(self, X: np.ndarray, Y_dict: Dict[str, np.ndarray]):
        """
        保持原有接口，默认使用类别权重策略
        """
        return self.train_models_with_imbalance_handling(X, Y_dict, strategy="class_weight")
        
    
    def predict_partner_probabilities(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        预测每个合作方的通过概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            每个合作方的通过概率
        """
        probabilities = {}
        
        for partner in self.partners:
            if partner in self.models:
                # 获取正类概率
                proba = self.models[partner].predict_proba(X)[:, 1]
                probabilities[partner] = proba
            
        return probabilities
    
    def recommend_partners(self, user_features: np.ndarray, k: int = 3, 
                          min_probability: float = 0.3) -> List[Tuple[str, float]]:
        """
        为用户推荐合作方
        
        Args:
            user_features: 用户特征 (1, n_features)
            k: 推荐的合作方数量上限
            min_probability: 最小通过概率阈值
            
        Returns:
            推荐的合作方列表 [(partner_name, probability), ...]
        """
        # 预测所有合作方的通过概率
        probabilities = self.predict_partner_probabilities(user_features)
        
        # 过滤低概率合作方
        filtered_partners = [
            (partner, prob[0]) for partner, prob in probabilities.items() 
            if prob[0] >= min_probability
        ]
        
        # 按概率排序并取前k个
        filtered_partners.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_partners[:k]
    
    def save_model(self, model_path: str = "loan_distribution_model"):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        os.makedirs(model_path, exist_ok=True)
        
        # 保存各个组件
        joblib.dump(self.models, os.path.join(model_path, 'models.pkl'))
        joblib.dump(self.encoders, os.path.join(model_path, 'encoders.pkl'))
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.pkl'))
        joblib.dump(self.partners, os.path.join(model_path, 'partners.pkl'))
        
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: str = "loan_distribution_model"):
        """
        加载模型
        
        Args:
            model_path: 模型加载路径
        """
        self.models = joblib.load(os.path.join(model_path, 'models.pkl'))
        self.encoders = joblib.load(os.path.join(model_path, 'encoders.pkl'))
        self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        self.partners = joblib.load(os.path.join(model_path, 'partners.pkl'))
        
        print(f"模型已从 {model_path} 加载")

def main():
    """
    主函数：演示完整的训练和预测流程
    """
    print("=== 贷款分发智能决策模型 - 处理类别不平衡 ===")
    
    # 1. 初始化模型
    model = LoanDistributionModel()
    
    # 2. 加载和预处理数据
    combined_data = model.load_and_combine_data()
    processed_data = model.preprocess_features(combined_data)
    
    # 3. 准备训练数据
    X, Y_dict = model.prepare_training_data(processed_data)
    
    # 4. 比较不同的不平衡处理策略
    comparison_results = model.compare_imbalance_strategies(X, Y_dict)


def demo_single_strategy():
    """
    演示单一策略的详细效果
    """
    print("=== 单一策略演示：类别权重方法 ===")
    
    model = LoanDistributionModel()
    combined_data = model.load_and_combine_data()
    processed_data = model.preprocess_features(combined_data)
    X, Y_dict = model.prepare_training_data(processed_data)

    # 使用SMOTE策略训练
    results = model.train_models_with_imbalance_handling(X, Y_dict, strategy="smote")
    
    # 打印详细结果
    for partner, cv_result in results.items():
        print(f"\n{partner} 详细结果:")
        auc_key = 'test_roc_auc' if 'test_roc_auc' in cv_result else 'test_auc'
        print(f"  AUC: {cv_result[auc_key].mean():.3f} ± {cv_result[auc_key].std():.3f}")
        print(f"  准确率: {cv_result['test_accuracy'].mean():.3f} ± {cv_result['test_accuracy'].std():.3f}")
        print(f"  召回率: {cv_result['test_recall'].mean():.3f} ± {cv_result['test_recall'].std():.3f}")
        print(f"  F1分数: {cv_result['test_f1'].mean():.3f} ± {cv_result['test_f1'].std():.3f}")
    
    return model

if __name__ == "__main__":
    main()
