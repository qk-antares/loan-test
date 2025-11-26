import os
import pandas as pd
import joblib
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from typing import List, Dict
from data_process.daily_partner_processor import LoanDataProcessor
from .increment_model import LoanDistributionModel
load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL_ROOT = os.getenv("MODEL_ROOT", "task/ml_result")

feature_columns = [
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
    # 'term',
    'deviceInfo.gpsLatitude',
    'deviceInfo.gpsLongitude',
    'deviceInfo.osType',
    'deviceInfo.isCrossDomain',
    'deviceInfo.applyPos'
]

feature_list = [
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


def convert_to_float_str(value):
    """数字转成 'x.0' 字符串；非数字或空保持不变"""
    if value is None:
        return value
    value = str(value).strip()
    if value == "":
        return value

    # 判断是否为纯数字（例如 '1', '01', '2'）
    if value.isdigit():
        return f"{int(value)}.0"  # '01' → '1' → '1.0'

    return value


def preprocess_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """
    单条或批量数据预处理，返回模型可用特征
    Args:
        df: 原始输入 DataFrame
        feature_list: 模型训练时使用的全量特征列表

    Returns:
        处理后的 DataFrame
    """
    print("开始特征预处理...")
    processed_df = df.copy()

    # -------------------------------
    # 公司名称过滤规则
    company_filter_rules = {
        'AWJ': ['公安局', '警察', '法院', '军队', '检察院', '城市管理局', '律师', '记者', '贷款', '金融', '执行局',
                '监狱', '交通警察', '派出所', '刑事侦查部门', '交警', '刑侦'],
        'RONG': ['学校', '小学', '中学', '大学', '学院', '公检法'],
    }

    if 'companyInfo.companyName' in processed_df.columns:
        company_series = processed_df['companyInfo.companyName'].astype(str).fillna("")
        for rule_name, keywords in company_filter_rules.items():
            for keyword in keywords:
                col_name = f"company_{rule_name}_{keyword}"
                processed_df[col_name] = company_series.apply(lambda x: 1 if keyword in x else 0)

    # 学历编码
    if 'degree' in processed_df.columns:
        processed_df['degree_encoded'] = processed_df['degree'].fillna(0)
        processed_df = processed_df.drop('degree', axis=1)

    # 收入编码
    if 'income' in processed_df.columns:
        processed_df['income_encoded'] = processed_df['income'].fillna(0)
        processed_df = processed_df.drop('income', axis=1)

    # 城市编码
    if 'city' in processed_df.columns:
        processed_df['city'] = processed_df['city'].fillna('UNKNOWN').astype(str)
        first_tier = ['北京市', '上海市', '广州市', '深圳市']
        new_first_tier = ['杭州市', '南京市', '成都市', '武汉市', '重庆市', '苏州市', '天津市', '西安市', '长沙市',
                          '青岛市']
        def simplify_city(city):
            if city in first_tier:
                return 'first_tier'
            elif city in new_first_tier:
                return 'new_first_tier'
            else:
                return 'others'
        processed_df['city_group'] = processed_df['city'].apply(simplify_city)
        city_dummies = pd.get_dummies(processed_df['city_group'], prefix='city')
        processed_df = pd.concat([processed_df, city_dummies], axis=1)
        processed_df = processed_df.drop(['city', 'city_group'], axis=1)

    # 省份编码
    if 'province' in processed_df.columns:
        processed_df['province'] = processed_df['province'].fillna('UNKNOWN').astype(str)
        key_provinces = ['北京市', '上海市', '广东省', '浙江省', '江苏省']
        def simplify_province(province):
            if province in key_provinces:
                return 'key_province'
            else:
                return 'other_province'
        processed_df['province_group'] = processed_df['province'].apply(simplify_province)
        province_dummies = pd.get_dummies(processed_df['province_group'], prefix='province')
        processed_df = pd.concat([processed_df, province_dummies], axis=1)
        processed_df = processed_df.drop(['province', 'province_group'], axis=1)

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

    # 设备省份
    if 'deviceInfo.applyPos' in processed_df.columns:
        processed_df['deviceInfo.applyPos'] = processed_df['deviceInfo.applyPos'].fillna('UNKNOWN').astype(str)
        processed_df['device_province'] = processed_df['deviceInfo.applyPos'].apply(lambda x: x[:3] if x != 'UNKNOWN' else 'UNKNOWN')
        def simplify_device_province(prov):
            if prov in key_provinces:
                return 'key_province'
            else:
                return 'others'
        processed_df['device_province_group'] = processed_df['device_province'].apply(simplify_device_province)
        device_province_dummies = pd.get_dummies(processed_df['device_province_group'], prefix='device_province')
        processed_df = pd.concat([processed_df, device_province_dummies], axis=1)
        processed_df = processed_df.drop(['deviceInfo.applyPos', 'device_province', 'device_province_group'], axis=1)

    # jobFunctions
    if 'jobFunctions' in processed_df.columns:
        processed_df['jobFunctions'] = processed_df['jobFunctions'].fillna('UNKNOWN').astype(str)
        # 打印原始数值（确保你看到是 "1" 还是 "1.0"）
        print("原始 jobFunctions 值：")
        print(processed_df['jobFunctions'].unique())
        job_dummies = pd.get_dummies(processed_df['jobFunctions'], prefix='job')
        processed_df = pd.concat([processed_df, job_dummies], axis=1)
        processed_df = processed_df.drop('jobFunctions', axis=1)

    # resideFunctions
    if 'resideFunctions' in processed_df.columns:
        processed_df['resideFunctions'] = processed_df['resideFunctions'].fillna('UNKNOWN').astype(str)
        reside_dummies = pd.get_dummies(processed_df['resideFunctions'], prefix='reside')
        processed_df = pd.concat([processed_df, reside_dummies], axis=1)
        processed_df = processed_df.drop('resideFunctions', axis=1)

    # maritalStatus
    if 'maritalStatus' in processed_df.columns:
        processed_df['maritalStatus'] = processed_df['maritalStatus'].fillna('UNKNOWN').astype(str)
        marital_dummies = pd.get_dummies(processed_df['maritalStatus'], prefix='marital')
        # 打印新增的列名
        # print("新增的婚姻状态列:", marital_dummies.columns.tolist())
        processed_df = pd.concat([processed_df, marital_dummies], axis=1)
        processed_df = processed_df.drop('maritalStatus', axis=1)


    # purpose
    if 'purpose' in processed_df.columns:
        processed_df['purpose'] = processed_df['purpose'].fillna('UNKNOWN').astype(str)
        purpose_dummies = pd.get_dummies(processed_df['purpose'], prefix='purpose')
        processed_df = pd.concat([processed_df, purpose_dummies], axis=1)
        processed_df = processed_df.drop('purpose', axis=1)

    # customerSource
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

    # deviceInfo.osType
    if 'deviceInfo.osType' in processed_df.columns:
        processed_df['deviceInfo.osType'] = processed_df['deviceInfo.osType'].fillna('UNKNOWN').astype(str)
        ostype_dummies = pd.get_dummies(processed_df['deviceInfo.osType'], prefix='ostype')
        processed_df = pd.concat([processed_df, ostype_dummies], axis=1)
        processed_df = processed_df.drop('deviceInfo.osType', axis=1)

    # idInfo.gender
    if 'idInfo.gender' in processed_df.columns:
        processed_df['idInfo.gender'] = processed_df['idInfo.gender'].fillna('UNKNOWN').astype(str)
        gender_dummies = pd.get_dummies(processed_df['idInfo.gender'], prefix='gender')
        processed_df = pd.concat([processed_df, gender_dummies], axis=1)
        processed_df = processed_df.drop('idInfo.gender', axis=1)

    # linkmanList.0.relationship
    if 'linkmanList.0.relationship' in processed_df.columns:
        processed_df['linkmanList.0.relationship'] = processed_df['linkmanList.0.relationship'].fillna('UNKNOWN').astype(str)
        relationship_dummies = pd.get_dummies(processed_df['linkmanList.0.relationship'], prefix='relationship')
        processed_df = pd.concat([processed_df, relationship_dummies], axis=1)
        processed_df = processed_df.drop('linkmanList.0.relationship', axis=1)

    # linkmanList.1.relationship
    if 'linkmanList.1.relationship' in processed_df.columns:
        processed_df['linkmanList.1.relationship'] = processed_df['linkmanList.1.relationship'].fillna('UNKNOWN').astype(str)
        relationship1_dummies = pd.get_dummies(processed_df['linkmanList.1.relationship'], prefix='relationship1')
        processed_df = pd.concat([processed_df, relationship1_dummies], axis=1)
        processed_df = processed_df.drop('linkmanList.1.relationship', axis=1)

    # idInfo.nation
    if 'idInfo.nation' in processed_df.columns:
        processed_df['idInfo.nation'] = processed_df['idInfo.nation'].fillna('UNKNOWN').astype(str).str.strip()
        def simplify_nation(nation):
            if nation == '汉':
                return 'han'
            elif nation == 'UNKNOWN':
                return 'unknown'
            else:
                return 'minority'
        processed_df['nation_simple'] = processed_df['idInfo.nation'].apply(simplify_nation)
        nation_dummies = pd.get_dummies(processed_df['nation_simple'], prefix='nation')
        processed_df = pd.concat([processed_df, nation_dummies], axis=1)
        processed_df = processed_df.drop(['idInfo.nation', 'nation_simple'], axis=1)

    # deviceInfo.isCrossDomain
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

    # 数值列填充
    numerical_features = ['amount', 'pictureInfo.0.faceScore', 'term',
                          'companyInfo.industry', 'companyInfo.occupation']
    for feature in numerical_features:
        if feature in processed_df.columns:
            processed_df[feature] = pd.to_numeric(processed_df[feature], errors='coerce')
            processed_df[feature] = processed_df[feature].fillna(processed_df[feature].median())

    # 最终缺失列补0
    missing_cols = [col for col in feature_list if col not in processed_df.columns]
    for col in missing_cols:
        processed_df[col] = 0

    print(f"已补全 {len(missing_cols)} 列，缺失列为：{missing_cols}")
    print(f"预处理完成，特征维度: {processed_df.shape[1]}")

    # 对齐 feature_list 顺序
    processed_df = processed_df.reindex(columns=feature_list, fill_value=0)

    return processed_df


def load_latest_model_date():
    """自动获取 ml_result 最新日期文件夹"""
    dates = [d for d in os.listdir(MODEL_ROOT) if os.path.isdir(os.path.join(MODEL_ROOT, d))]
    if not dates:
        raise Exception("ml_result 下没有任何模型日期文件夹")
    return sorted(dates)[-1]


def preprocess_data(raw_A: dict, model_feature_columns: list):
    print("raw_A是",raw_A)

    fields_to_fix = ["maritalStatus", "jobFunctions", "resideFunctions", "companyInfo.occupation"]

    for field in fields_to_fix:
        if field in raw_A:
            raw_A[field] = convert_to_float_str(raw_A[field])

    processor = LoanDataProcessor()
    features = processor.process_single_record_for_predict(raw_A, feature_columns)
    print("\n📋 第一次处理后的特征列是：",features)
    df_ = pd.DataFrame([features])  # 单条记录
    df = preprocess_features(df_, feature_list)

    # 显示所有列
    pd.set_option('display.max_columns', None)

    # 可选：显示所有行
    pd.set_option('display.max_rows', None)

    # 显示完整 DataFrame（前几行）
    print("\n📋 第二次处理后的特征列是：")
    print(df)

    # 打印完整列名列表
    print("\n📋 列名列表：")
    print(df.columns.tolist())


    missing_cols = [col for col in feature_list if col not in df.columns]
    for col in missing_cols:
        df[col] = 0

    # 打印缺失列
    if missing_cols:
        print(f"\n❗ 缺失并补齐的列 ({len(missing_cols)} 列)：")
        print(missing_cols)
    else:
        print("\n✅ 无缺失列")


    final_cols = [c for c in model_feature_columns if c in df.columns]
    df = df.reindex(columns=final_cols, fill_value=0)

    return df


def predict_result(request: dict):
    """
    预测接口：返回固定格式
    成功：
    {
        "code": 200,
        "data": { "partnerA": 0.0523, "partnerB": 0.1123, ... },
        "msg": "成功"
    }
    失败：
    {
        "code": 400,
        "data": {},
        "msg": "错误信息"
    }
    """

    if request.get("key") != API_KEY:
        return {"code": 400, "data": {}, "msg": "Invalid API key"}

    raw_A = request.get("A")
    partners = request.get("B", [])

    if not isinstance(raw_A, dict):
        return {"code": 400, "data": {}, "msg": "A 参数必须为 dict"}

    if not partners:
        return {"code": 400, "data": {}, "msg": "B 参数为空"}

    model_date = load_latest_model_date()
    results = {}

    # --------- 先处理一次特征 ----------
    df_A = preprocess_data(raw_A, feature_list)

    # 打印为列表形式，更清晰
    print(list(df_A.columns))

    # --------- 循环调用不同合作方模型 ----------
    for partner in partners:
        model_path = Path(MODEL_ROOT) / model_date / partner / "best_model.pkl"

        if not model_path.exists():
            results[partner] = None
            continue

        try:
            model = joblib.load(model_path)
            # 预测
            prob = model.predict_proba(df_A)[0][1]
            results[partner] = round(float(prob), 4)

        except Exception as e:
            print(f"[Predict Error] {e}")
            results[partner] = None

    # 检查是否所有预测都失败
    if all(v is None for v in results.values()):
        return {"code": 500, "data": {}, "msg": "所有预测失败，请检查输入或模型"}

    return {"code": 200, "data": results, "msg": "成功"}



if __name__ == "__main__":
    # 测试请求示例
    test_request = {
        "A": {
            "amount": "100000",
            "bankCardInfo.bankCode": "",
            "city": "杭州市",
            "companyInfo.companyName": "浙江师范大学萧山校区",
            "companyInfo.industry": "Z",
            "companyInfo.occupation": "90",
            "customerSource": "XCX",
            "degree": "BACHELOR",
            "idInfo.birthDate": "19920319",
            "idInfo.gender": "F",
            "idInfo.nation": "汉",
            "idInfo.validityDate": "2020.10.15-2040.10.15",
            "income": "D",
            "jobFunctions": "01",
            "linkmanList.0.relationship": "FATHER",
            "linkmanList.1.relationship": "MOTHER",
            "maritalStatus": "2",
            "pictureInfo.0.faceScore": "78.31419",
            "pictureInfo.1.faceScore": "78.31419",
            "pictureInfo.2.faceScore": "78.31419",
            "province": "浙江省",
            "purpose": "CONSUME",
            "resideFunctions": "01",
            "deviceInfo.gpsLatitude": "31.040912185925714",
            "deviceInfo.gpsLongitude": "121.46704181988909",
            "deviceInfo.osType": "ANDROID",
            "deviceInfo.isCrossDomain": "",
            "deviceInfo.applyPos": "上海市闵行区吴泾镇金家塘路",
            "id": "20251118000010987596"
        },

        "B": ["LXJ_CODE", "HXQB_CODE"],
        "key": "xiaohua666"
    }


    # 调用预测
    result = predict_result(test_request)
    print("预测结果：")
    print(result)


