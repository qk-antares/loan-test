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
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, fbeta_score, hamming_loss, accuracy_score, make_scorer
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
        numerical_features = ['amount', 'idInfo.birthDate', 'idInfo.validityDate', 'term']
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
    
    def train_models(self, X: np.ndarray, Y_dict: Dict[str, np.ndarray]):
        """
        训练模型 - 修正版：避免数据泄露
        
        Args:
            X: 特征矩阵
            Y_dict: 每个合作方的标签字典
        """
        print("开始训练模型...")
        
        # 为每个合作方训练单独的分类器
        for partner in self.partners:
            print(f"训练 {partner} 模型...")
            
            partner_data = Y_dict[partner]
            X_partner = X[partner_data['X_indices']]
            y_partner = partner_data['labels']
            
            # 检查数据量是否足够
            if len(y_partner) < 10:
                print(f"  {partner} 数据量太少 ({len(y_partner)} 样本), 跳过训练")
                continue
            
            # 使用随机森林分类器
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=min(5, len(y_partner) // 2),
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            
            scoring = {
                'roc_auc': 'roc_auc',
                'accuracy': 'accuracy',
                'recall': 'recall',
                'f1': 'f1',
            }

            cv_results = cross_validate(
                model, X_partner, y_partner, cv=5, scoring=scoring
            )

            print(f"  {partner} 交叉验证 AUC: {cv_results['test_roc_auc'].mean():.3f}")
            print(f"  {partner} 交叉验证 准确率: {cv_results['test_accuracy'].mean():.3f}")
            print(f"  {partner} 交叉验证 召回率: {cv_results['test_recall'].mean():.3f}")
            print(f"  {partner} 交叉验证 F1: {cv_results['test_f1'].mean():.3f}")
        
    
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
    print("=== 贷款分发智能决策模型 ===")
    
    # 1. 初始化模型
    model = LoanDistributionModel()
    
    # 2. 加载和预处理数据
    combined_data = model.load_and_combine_data()
    processed_data = model.preprocess_features(combined_data)
    
    # 3. 准备训练数据
    X, Y_dict = model.prepare_training_data(processed_data)
    
    # 4. 训练模型
    model.train_models(X, Y_dict)

if __name__ == "__main__":
    main()
