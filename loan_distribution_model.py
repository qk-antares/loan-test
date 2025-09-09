"""
贷款导流分发模型设计方案
针对多合作方贷款平台的智能导流系统
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime, timedelta
from collections import defaultdict

class LoanDistributionModel:
    """
    贷款导流分发模型
    
    核心思想：
    1. 将问题建模为多臂老虎机（Multi-Armed Bandit）+ 推荐系统
    2. 每个合作方是一个"臂"，用户-合作方匹配是推荐问题
    3. 考虑时间动态性和合作方规则变化
    """
    
    def __init__(self):
        self.partner_models = {}  # 每个合作方的独立模型
        self.meta_model = None    # 元学习模型
        self.feature_processor = FeatureProcessor()
        self.distribution_optimizer = DistributionOptimizer()
    
    def design_architecture(self):
        """
        模型架构设计
        """
        return {
            "level1_partner_models": "每个合作方的独立预测模型",
            "level2_meta_model": "元学习模型，学习合作方偏好变化",
            "level3_distribution": "分发优化算法",
            "adaptive_learning": "在线学习和模型更新机制"
        }


class FeatureProcessor:
    """
    异质性特征处理器
    处理数值、分类、文本、时间、地理等不同类型的特征
    """
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.text_processors = {}
    
    def process_features(self, loan_request: Dict) -> Dict:
        """
        处理贷款请求的异质性特征
        """
        processed_features = {}
        
        # 1. 数值特征处理
        numerical_features = self._process_numerical_features(loan_request)
        processed_features.update(numerical_features)
        
        # 2. 分类特征处理
        categorical_features = self._process_categorical_features(loan_request)
        processed_features.update(categorical_features)
        
        # 3. 文本特征处理
        text_features = self._process_text_features(loan_request)
        processed_features.update(text_features)
        
        # 4. 时间特征处理
        temporal_features = self._process_temporal_features(loan_request)
        processed_features.update(temporal_features)
        
        # 5. 地理特征处理
        geo_features = self._process_geo_features(loan_request)
        processed_features.update(geo_features)
        
        return processed_features
    
    def _process_numerical_features(self, loan_request: Dict) -> Dict:
        """处理数值特征"""
        features = {}
        
        # 金额相关
        amount = float(loan_request.get('amount', 0))
        features['amount_log'] = np.log1p(amount)  # 对数变换
        features['amount_normalized'] = amount / 1000000  # 归一化
        
        # 期数
        term = int(loan_request.get('term', 12))
        features['term'] = term
        features['monthly_payment'] = amount / term if term > 0 else 0
        
        # 设备信息数值化
        device_info = loan_request.get('deviceInfo', {})
        features['memory_gb'] = float(device_info.get('memory', 0)) / (1024**3)
        features['storage_gb'] = float(device_info.get('storage', 0)) / (1024**3)
        features['electricity'] = float(device_info.get('electricity', 0))
        
        return features
    
    def _process_categorical_features(self, loan_request: Dict) -> Dict:
        """处理分类特征"""
        features = {}
        
        # 教育程度编码
        degree_mapping = {
            'DOCTOR': 6, 'MASTER': 5, 'BACHELOR': 4, 
            'COLLEGE': 3, 'SENIOR': 2, 'JUNIOR': 1
        }
        features['degree_level'] = degree_mapping.get(loan_request.get('degree'), 0)
        
        # 收入等级编码
        income_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        features['income_level'] = income_mapping.get(loan_request.get('income'), 0)
        
        # 婚姻状况
        features['marital_status'] = int(loan_request.get('maritalStatus', 0))
        
        # 行业编码
        industry_mapping = {
            'A': 1, 'C': 2, 'E': 3, 'G': 4, 'H': 5, 'Z': 0  # 简化示例
        }
        company_info = loan_request.get('companyInfo', {})
        features['industry_code'] = industry_mapping.get(
            company_info.get('industry'), 0
        )
        
        return features
    
    def _process_text_features(self, loan_request: Dict) -> Dict:
        """处理文本特征"""
        features = {}
        
        # 地址文本特征
        address = loan_request.get('liveAddress', '')
        features['address_length'] = len(address)
        features['address_contains_apartment'] = 1 if '小区' in address or '公寓' in address else 0
        
        # 公司名称特征
        company_info = loan_request.get('companyInfo', {})
        company_name = company_info.get('companyName', '')
        features['company_name_length'] = len(company_name)
        features['is_government'] = 1 if any(keyword in company_name for keyword in ['政府', '公安', '法院']) else 0
        features['is_bank'] = 1 if '银行' in company_name else 0
        
        return features
    
    def _process_temporal_features(self, loan_request: Dict) -> Dict:
        """处理时间特征"""
        features = {}
        
        # 从身份证信息提取年龄
        id_info = loan_request.get('idInfo', {})
        birth_date = id_info.get('birthDate', '')
        if birth_date and len(birth_date) == 8:
            try:
                birth_year = int(birth_date[:4])
                current_year = datetime.now().year
                features['age'] = current_year - birth_year
                features['age_group'] = self._get_age_group(features['age'])
            except ValueError:
                features['age'] = 0
                features['age_group'] = 0
        
        # 证件有效期特征
        validity_date = id_info.get('validityDate', '')
        if '长期' in validity_date:
            features['id_validity_years'] = 99  # 长期
        else:
            # 解析有效期
            features['id_validity_years'] = 10  # 默认值
        
        return features
    
    def _process_geo_features(self, loan_request: Dict) -> Dict:
        """处理地理特征"""
        features = {}
        
        # GPS坐标
        device_info = loan_request.get('deviceInfo', {})
        try:
            lat = float(device_info.get('gpsLatitude', 0))
            lng = float(device_info.get('gpsLongitude', 0))
            
            features['gps_latitude'] = lat
            features['gps_longitude'] = lng
            features['gps_valid'] = 1 if lat != 0 and lng != 0 else 0
            
            # 计算到一线城市的距离（示例）
            beijing_coords = (39.9042, 116.4074)
            features['distance_to_beijing'] = self._calculate_distance(
                (lat, lng), beijing_coords
            )
            
        except (ValueError, TypeError):
            features['gps_latitude'] = 0
            features['gps_longitude'] = 0
            features['gps_valid'] = 0
            features['distance_to_beijing'] = 0
        
        # 地区等级
        province = loan_request.get('province', '')
        city = loan_request.get('city', '')
        features['is_tier1_city'] = 1 if city in ['北京市', '上海市', '广州市', '深圳市'] else 0
        features['is_municipality'] = 1 if province in ['北京市', '上海市', '天津市', '重庆市'] else 0
        
        return features
    
    def _get_age_group(self, age: int) -> int:
        """年龄分组"""
        if age < 25:
            return 1
        elif age < 35:
            return 2
        elif age < 45:
            return 3
        elif age < 55:
            return 4
        else:
            return 5
    
    def _calculate_distance(self, coord1: Tuple[float, float], 
                          coord2: Tuple[float, float]) -> float:
        """计算两点间距离（简化版）"""
        lat1, lng1 = coord1
        lat2, lng2 = coord2
        return np.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)


class PartnerSpecificModel:
    """
    合作方特定模型
    为每个合作方训练独立的预测模型
    """
    
    def __init__(self, partner_id: str):
        self.partner_id = partner_id
        self.model = None
        self.feature_importance = {}
        self.performance_history = []
        self.last_update_time = None
        
    def train(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        训练合作方特定模型
        
        Args:
            X: 特征矩阵
            y: 标签 (0/1)
            weights: 样本权重，用于处理不平衡数据
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils.class_weight import compute_class_weight
        
        # 处理类不平衡
        if weights is None:
            class_weights = compute_class_weight(
                'balanced', classes=np.unique(y), y=y
            )
            weight_dict = {0: class_weights[0], 1: class_weights[1]}
        else:
            weight_dict = None
            
        # 训练模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=weight_dict,
            random_state=42
        )
        
        if weights is not None:
            self.model.fit(X, y, sample_weight=weights)
        else:
            self.model.fit(X, y)
        
        # 记录特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(enumerate(self.model.feature_importances_))
        
        self.last_update_time = datetime.now()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测成功概率"""
        if self.model is None:
            return np.zeros((X.shape[0], 2))
        return self.model.predict_proba(X)
    
    def update_performance(self, accuracy: float, precision: float, recall: float):
        """更新性能指标"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        })


class OnlineLearningManager:
    """
    在线学习管理器
    处理概念漂移和模型更新
    """
    
    def __init__(self, drift_detection_window: int = 1000):
        self.drift_detection_window = drift_detection_window
        self.recent_predictions = defaultdict(list)
        self.recent_outcomes = defaultdict(list)
        
    def detect_concept_drift(self, partner_id: str, 
                           prediction: float, actual: int) -> bool:
        """
        检测概念漂移
        
        Args:
            partner_id: 合作方ID
            prediction: 预测概率
            actual: 实际结果
            
        Returns:
            是否检测到漂移
        """
        self.recent_predictions[partner_id].append(prediction)
        self.recent_outcomes[partner_id].append(actual)
        
        # 保持窗口大小
        if len(self.recent_predictions[partner_id]) > self.drift_detection_window:
            self.recent_predictions[partner_id].pop(0)
            self.recent_outcomes[partner_id].pop(0)
        
        # 简单的漂移检测：比较最近的性能和历史性能
        if len(self.recent_predictions[partner_id]) >= 100:
            recent_error = self._calculate_recent_error(partner_id)
            historical_error = self._calculate_historical_error(partner_id)
            
            # 如果最近误差显著高于历史误差，认为发生了漂移
            drift_threshold = 0.1  # 可调参数
            return recent_error > historical_error + drift_threshold
        
        return False
    
    def _calculate_recent_error(self, partner_id: str) -> float:
        """计算最近的预测误差"""
        predictions = self.recent_predictions[partner_id][-50:]  # 最近50个
        outcomes = self.recent_outcomes[partner_id][-50:]
        
        errors = [abs(pred - actual) for pred, actual in zip(predictions, outcomes)]
        return np.mean(errors)
    
    def _calculate_historical_error(self, partner_id: str) -> float:
        """计算历史预测误差"""
        predictions = self.recent_predictions[partner_id][:-50]  # 历史数据
        outcomes = self.recent_outcomes[partner_id][:-50]
        
        if not predictions:
            return 0.5  # 默认值
        
        errors = [abs(pred - actual) for pred, actual in zip(predictions, outcomes)]
        return np.mean(errors)


class DistributionOptimizer:
    """
    分发优化器
    在数量约束下优化导流策略
    """
    
    def __init__(self):
        self.partner_capacities = {}  # 合作方容量限制
        self.user_constraints = {}    # 用户分发限制
        
    def optimize_distribution(self, user_id: str, loan_request: Dict,
                            partner_predictions: Dict[str, float],
                            max_partners: int) -> List[str]:
        """
        优化分发策略
        
        Args:
            user_id: 用户ID
            loan_request: 贷款请求
            partner_predictions: 各合作方的成功概率预测
            max_partners: 最大分发合作方数量
            
        Returns:
            选择的合作方列表
        """
        
        # 方法1: 基于期望收益的贪心选择
        expected_revenues = self._calculate_expected_revenues(
            loan_request, partner_predictions
        )
        
        # 方法2: 考虑多样性的选择
        diversity_scores = self._calculate_diversity_scores(partner_predictions)
        
        # 方法3: 上置信界算法（UCB）用于探索-利用权衡
        ucb_scores = self._calculate_ucb_scores(partner_predictions)
        
        # 综合评分
        final_scores = {}
        for partner_id in partner_predictions:
            final_scores[partner_id] = (
                0.6 * expected_revenues.get(partner_id, 0) +
                0.2 * diversity_scores.get(partner_id, 0) +
                0.2 * ucb_scores.get(partner_id, 0)
            )
        
        # 选择Top-K合作方
        sorted_partners = sorted(
            final_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        selected_partners = [
            partner_id for partner_id, _ in sorted_partners[:max_partners]
        ]
        
        return selected_partners
    
    def _calculate_expected_revenues(self, loan_request: Dict,
                                   predictions: Dict[str, float]) -> Dict[str, float]:
        """计算期望收益"""
        amount = float(loan_request.get('amount', 0))
        commission_rate = 0.05  # 5%提成率
        
        revenues = {}
        for partner_id, success_prob in predictions.items():
            revenues[partner_id] = success_prob * amount * commission_rate
        
        return revenues
    
    def _calculate_diversity_scores(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """计算多样性评分，鼓励尝试不同类型的合作方"""
        # 简化实现：基于预测概率的方差
        probs = list(predictions.values())
        mean_prob = np.mean(probs)
        
        diversity_scores = {}
        for partner_id, prob in predictions.items():
            # 给予中等概率的合作方更高的多样性评分
            diversity_scores[partner_id] = 1 - abs(prob - mean_prob)
        
        return diversity_scores
    
    def _calculate_ucb_scores(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """计算上置信界评分"""
        # 简化实现，实际应该基于历史尝试次数
        ucb_scores = {}
        for partner_id, prob in predictions.items():
            # 给予较少尝试的合作方更高的探索奖励
            confidence_bonus = 0.1  # 简化的置信奖励
            ucb_scores[partner_id] = prob + confidence_bonus
        
        return ucb_scores


class LoanDistributionPipeline:
    """
    完整的贷款分发流水线
    """
    
    def __init__(self):
        self.feature_processor = FeatureProcessor()
        self.partner_models = {}
        self.online_learning = OnlineLearningManager()
        self.optimizer = DistributionOptimizer()
        
    def process_loan_request(self, user_id: str, loan_request: Dict,
                           max_partners: int = 3) -> Dict:
        """
        处理贷款请求
        
        Args:
            user_id: 用户ID
            loan_request: 贷款请求数据
            max_partners: 最大分发合作方数
            
        Returns:
            分发决策和预测结果
        """
        
        # 1. 特征处理
        processed_features = self.feature_processor.process_features(loan_request)
        feature_vector = np.array(list(processed_features.values())).reshape(1, -1)
        
        # 2. 为每个合作方进行预测
        partner_predictions = {}
        for partner_id, model in self.partner_models.items():
            if model.model is not None:
                prob = model.predict_proba(feature_vector)[0][1]  # 成功概率
                partner_predictions[partner_id] = prob
        
        # 3. 优化分发策略
        selected_partners = self.optimizer.optimize_distribution(
            user_id, loan_request, partner_predictions, max_partners
        )
        
        # 4. 返回结果
        return {
            'user_id': user_id,
            'selected_partners': selected_partners,
            'predictions': partner_predictions,
            'processed_features': processed_features,
            'timestamp': datetime.now().isoformat()
        }
    
    def update_with_feedback(self, user_id: str, partner_id: str,
                           prediction: float, actual_result: int):
        """
        使用反馈更新模型
        
        Args:
            user_id: 用户ID  
            partner_id: 合作方ID
            prediction: 之前的预测概率
            actual_result: 实际结果 (0/1)
        """
        
        # 检测概念漂移
        drift_detected = self.online_learning.detect_concept_drift(
            partner_id, prediction, actual_result
        )
        
        if drift_detected:
            print(f"检测到合作方 {partner_id} 的概念漂移，需要重新训练模型")
            # 触发模型重训练
            self._retrain_partner_model(partner_id)
    
    def _retrain_partner_model(self, partner_id: str):
        """重新训练合作方模型"""
        print(f"重新训练合作方 {partner_id} 的模型")
        # 实际实现中，这里会重新加载最近的数据并重训练模型
        pass


def create_modeling_recommendations():
    """
    针对贷款导流问题的建模建议
    """
    
    recommendations = {
        "数据不平衡解决方案": {
            "SMOTE过采样": "生成合成的少数类样本",
            "类权重调整": "在损失函数中给成功案例更高权重", 
            "集成方法": "使用BalancedRandomForest等专门算法",
            "阈值调整": "根据业务需求调整分类阈值",
            "代价敏感学习": "将误分类成本纳入模型训练"
        },
        
        "概念漂移应对": {
            "在线学习": "使用增量学习算法持续更新模型",
            "滑动窗口": "只使用最近的数据训练模型",
            "概念漂移检测": "监控模型性能，及时发现规则变化",
            "模型版本管理": "保持多个模型版本，快速回滚",
            "A/B测试": "逐步部署新模型，降低风险"
        },
        
        "异质性特征处理": {
            "特征工程": "针对不同类型特征设计专门的处理方法",
            "深度学习": "使用神经网络自动学习特征表示",
            "多模态融合": "分别处理不同模态再融合",
            "特征选择": "为每个合作方选择最相关的特征",
            "领域适应": "使用迁移学习技术"
        },
        
        "分发优化策略": {
            "多臂老虎机": "平衡探索和利用",
            "组合优化": "在约束条件下最优化期望收益",
            "强化学习": "学习长期最优策略",
            "动态规划": "考虑时间序列的最优决策",
            "博弈论": "考虑合作方之间的竞争关系"
        }
    }
    
    return recommendations


def main():
    """演示建模思路"""
    print("贷款导流分发建模方案")
    print("="*50)
    
    # 创建建模建议
    recommendations = create_modeling_recommendations()
    
    for category, methods in recommendations.items():
        print(f"\n{category}:")
        print("-" * 30)
        for method, description in methods.items():
            print(f"• {method}: {description}")
    
    print("\n" + "="*50)
    print("核心建模思路:")
    print("1. 多层次建模：合作方特定模型 + 元学习模型")
    print("2. 在线学习：应对概念漂移和规则变化")
    print("3. 多目标优化：平衡收益、风险和探索")
    print("4. 异质特征融合：设计专门的特征处理流水线")
    print("5. 约束优化：在分发数量限制下最大化期望收益")


if __name__ == "__main__":
    main()
