"""
贷款导流分发系统实际实现示例
基于scikit-learn实现的简化版本
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SimpleLoanDistributionSystem:
    """
    简化版贷款导流分发系统
    """
    
    def __init__(self, data_dir='processed_data'):
        self.data_dir = data_dir
        self.partner_models = {}
        self.feature_processors = {}
        self.performance_metrics = {}
        self.partner_data = {}
        
    def load_and_preprocess_data(self):
        """加载并预处理所有合作方数据"""
        print("加载合作方数据...")
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                partner_id = filename.replace('.csv', '')
                filepath = os.path.join(self.data_dir, filename)
                
                try:
                    df = pd.read_csv(filepath)
                    print(f"加载 {partner_id}: {len(df)} 条记录")
                    
                    # 基本数据清洗
                    df = self._clean_data(df)
                    self.partner_data[partner_id] = df
                    
                except Exception as e:
                    print(f"加载 {partner_id} 数据时出错: {e}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 处理缺失值
        for col in df.columns:
            if col != 'label':
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('unknown')
                else:
                    df[col] = df[col].fillna(0)
        
        return df
    
    def train_partner_models(self):
        """为每个合作方训练专门的模型"""
        print("\n开始训练合作方模型...")
        
        for partner_id, df in self.partner_data.items():
            print(f"\n训练 {partner_id} 的模型...")
            
            if len(df) < 50:  # 数据太少，跳过
                print(f"  数据量不足，跳过 {partner_id}")
                continue
            
            # 准备特征和标签
            X, y = self._prepare_features_labels(df)
            
            if X is None:
                continue
            
            # 分割训练测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # 训练模型
            model, scaler = self._train_model(X_train, y_train)
            
            # 评估模型
            metrics = self._evaluate_model(model, scaler, X_test, y_test)
            
            # 保存模型和处理器
            self.partner_models[partner_id] = model
            self.feature_processors[partner_id] = scaler
            self.performance_metrics[partner_id] = metrics
            
            print(f"  {partner_id} 模型训练完成")
            print(f"  AUC: {metrics['auc']:.3f}, Precision: {metrics['precision']:.3f}")
    
    def _prepare_features_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备特征和标签"""
        try:
            # 分离特征和标签
            feature_cols = [col for col in df.columns if col != 'label']
            X = df[feature_cols].copy()
            y = df['label'].values
            
            # 处理分类特征
            le = LabelEncoder()
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # 转换为数值数组
            X = X.values.astype(float)
            
            return X, y
            
        except Exception as e:
            print(f"  特征准备出错: {e}")
            return None, None
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练模型"""
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 计算类权重处理不平衡
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # 训练随机森林模型
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=weight_dict,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        return model, scaler
    
    def _evaluate_model(self, model, scaler, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型性能"""
        X_test_scaled = scaler.transform(X_test)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 计算指标
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
        
        # 计算精确率和召回率
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        return {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'positive_samples': sum(y_test),
            'total_samples': len(y_test)
        }
    
    def predict_success_probability(self, loan_features: Dict, partner_id: str) -> float:
        """预测指定合作方的成功概率"""
        if partner_id not in self.partner_models:
            return 0.1  # 默认较低概率
        
        model = self.partner_models[partner_id]
        scaler = self.feature_processors[partner_id]
        
        try:
            # 准备特征向量
            feature_vector = self._prepare_single_request(loan_features, partner_id)
            if feature_vector is None:
                return 0.1
            
            # 标准化和预测
            feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))
            prob = model.predict_proba(feature_vector_scaled)[0][1]
            
            return prob
            
        except Exception as e:
            print(f"预测 {partner_id} 时出错: {e}")
            return 0.1
    
    def _prepare_single_request(self, loan_features: Dict, partner_id: str) -> np.ndarray:
        """为单个请求准备特征向量"""
        try:
            # 获取该合作方的特征列
            sample_data = self.partner_data[partner_id]
            feature_cols = [col for col in sample_data.columns if col != 'label']
            
            # 构建特征向量
            feature_vector = []
            for col in feature_cols:
                if col in loan_features:
                    value = loan_features[col]
                    # 简单的类型处理
                    if isinstance(value, str):
                        # 使用样本数据中的映射
                        unique_values = sample_data[col].unique()
                        if value in unique_values:
                            feature_vector.append(list(unique_values).index(value))
                        else:
                            feature_vector.append(0)  # 未知值
                    else:
                        feature_vector.append(float(value) if value is not None else 0)
                else:
                    feature_vector.append(0)  # 缺失特征
            
            return np.array(feature_vector)
            
        except Exception as e:
            print(f"准备特征向量时出错: {e}")
            return None
    
    def optimize_distribution(self, loan_features: Dict, max_partners: int = 3) -> List[Dict]:
        """优化分发策略"""
        # 获取所有合作方的预测
        partner_predictions = {}
        for partner_id in self.partner_models.keys():
            prob = self.predict_success_probability(loan_features, partner_id)
            partner_predictions[partner_id] = prob
        
        # 计算期望收益
        amount = float(loan_features.get('amount', 100000))
        commission_rate = 0.05
        
        partner_scores = []
        for partner_id, success_prob in partner_predictions.items():
            expected_revenue = success_prob * amount * commission_rate
            
            # 添加探索奖励（简化版UCB）
            exploration_bonus = 0.1 * np.sqrt(2 * np.log(100) / max(1, 10))  # 简化
            
            final_score = expected_revenue + exploration_bonus
            
            partner_scores.append({
                'partner_id': partner_id,
                'success_probability': success_prob,
                'expected_revenue': expected_revenue,
                'final_score': final_score
            })
        
        # 排序并选择Top-K
        partner_scores.sort(key=lambda x: x['final_score'], reverse=True)
        selected = partner_scores[:max_partners]
        
        return selected
    
    def print_model_summary(self):
        """打印模型摘要"""
        print("\n" + "="*60)
        print("合作方模型训练摘要")
        print("="*60)
        
        print(f"{'合作方':<12} {'数据量':<8} {'AUC':<8} {'精确率':<8} {'召回率':<8} {'正样本':<8}")
        print("-" * 60)
        
        for partner_id in sorted(self.performance_metrics.keys()):
            metrics = self.performance_metrics[partner_id]
            data_size = len(self.partner_data[partner_id])
            
            print(f"{partner_id:<12} {data_size:<8} {metrics['auc']:<7.3f} "
                  f"{metrics['precision']:<7.3f} {metrics['recall']:<7.3f} "
                  f"{metrics['positive_samples']:<8}")
    
    def simulate_distribution(self, num_simulations: int = 10):
        """模拟分发过程"""
        print(f"\n模拟 {num_simulations} 个贷款请求的分发...")
        
        # 生成模拟请求
        simulated_requests = self._generate_simulated_requests(num_simulations)
        
        total_expected_revenue = 0
        distribution_results = []
        
        for i, request in enumerate(simulated_requests):
            print(f"\n请求 {i+1}:")
            print(f"  金额: {request['amount']:,}")
            print(f"  行业: {request.get('companyInfo.industry', 'unknown')}")
            
            # 优化分发
            selected_partners = self.optimize_distribution(request, max_partners=3)
            
            print(f"  推荐合作方:")
            request_revenue = 0
            for j, partner_info in enumerate(selected_partners):
                prob = partner_info['success_probability']
                revenue = partner_info['expected_revenue']
                request_revenue += revenue
                
                print(f"    {j+1}. {partner_info['partner_id']}: "
                      f"成功率 {prob:.1%}, 期望收益 ¥{revenue:,.0f}")
            
            total_expected_revenue += request_revenue
            distribution_results.append({
                'request_id': i+1,
                'selected_partners': [p['partner_id'] for p in selected_partners],
                'expected_revenue': request_revenue
            })
        
        print(f"\n总期望收益: ¥{total_expected_revenue:,.0f}")
        print(f"平均每请求期望收益: ¥{total_expected_revenue/num_simulations:,.0f}")
        
        return distribution_results
    
    def _generate_simulated_requests(self, num_requests: int) -> List[Dict]:
        """生成模拟贷款请求"""
        requests = []
        
        # 从现有数据中采样
        all_data = []
        for df in self.partner_data.values():
            all_data.append(df)
        
        if all_data:
            combined_data = pd.concat(all_data).drop_duplicates()
            sample_data = combined_data.sample(n=min(num_requests, len(combined_data)))
            
            for _, row in sample_data.iterrows():
                request = {}
                for col in row.index:
                    if col != 'label':
                        request[col] = row[col]
                requests.append(request)
        
        return requests


def main():
    """主函数：演示贷款导流分发系统"""
    print("贷款导流分发系统演示")
    print("="*50)
    
    # 初始化系统
    system = SimpleLoanDistributionSystem()
    
    # 加载数据
    system.load_and_preprocess_data()
    
    if not system.partner_data:
        print("没有找到数据文件，请先运行数据预处理脚本")
        return
    
    # 训练模型
    system.train_partner_models()
    
    # 打印模型摘要
    system.print_model_summary()
    
    # 模拟分发过程
    if system.partner_models:
        system.simulate_distribution(num_simulations=5)
    
    print("\n" + "="*50)
    print("系统演示完成!")
    print("\n核心功能:")
    print("1. ✓ 为每个合作方训练专门的预测模型")
    print("2. ✓ 处理数据不平衡问题（类权重调整）")
    print("3. ✓ 智能分发优化（期望收益最大化）")
    print("4. ✓ 异质特征处理（自动编码和标准化）")
    print("5. ✓ 模型性能评估和监控")


if __name__ == "__main__":
    main()
