from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import os
import traceback
from dotenv import load_dotenv
from task.pull_data import pull_missing_files_temp, run_day_pipeline
from task.predict import predict_result

load_dotenv()

app = Flask(__name__)
CORS(app)

# 从环境变量读取配置
API_KEY = os.getenv('API_KEY')
SFTP_HOST = os.getenv('SFTP_HOST')
SFTP_PORT = int(os.getenv('SFTP_PORT', 38981))
SFTP_USER = os.getenv('SFTP_USER')
SFTP_PASS = os.getenv('SFTP_PASS')

def setup_logging():
    """配置日志"""
    log_dir = "task/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'res.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def validate_key(request_key):
    """验证API密钥"""
    return request_key == API_KEY

@app.route('/api/health_check/', methods=['GET'])
def health_check():
    """服务健康检查接口"""
    try:
        key = request.args.get('key', '')
        logging.info(f"健康检查请求 - 密钥: {key}")
        
        if not validate_key(key):
            logging.warning(f"健康检查 - 状态码: 401, 无效API密钥: {key}")
            return jsonify({
                "code": 401,
                "data": {},
                "msg": "无效的API密钥"
            })
        logging.info("健康检查 - 状态码: 200, 服务状态正常")
        return jsonify({
            "code": 200,
            "data": {
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            },
            "msg": "服务正常运行"
        })
        
    except Exception as e:
        logging.error(f"服务健康检查失败 - 状态码: 500, 错误：{str(e)}")
        return jsonify({
            "code": 500,
            "data": {},
            "msg": f"服务健康检查失败: {str(e)}"
        })

@app.route('/api/predict_candidates_proba', methods=['POST'])
def predict_candidates_proba():
    """通过率预测接口"""
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            logging.warning("预测请求 - 状态码: 400, 请求体为空或非JSON格式")
            return jsonify({
                "code": 400,
                "data": {},
                "msg": "请求体必须为JSON格式"
            })
        
        # 验证密钥
        request_key = data.get('key', '')
        if not validate_key(request_key):
            logging.warning(f"预测请求 - 状态码: 401, 无效API密钥: {request_key}")
            return jsonify({
                "code": 401,
                "data": {},
                "msg": "无效的API密钥"
            })
        
        # 数据合法性检查
        validation_result = validate_prediction_data(data)
        if not validation_result["valid"]:
            logging.warning(f"预测请求 - 状态码: 400, 数据校验失败: {validation_result['msg']}")
            return jsonify({
                "code": 400,
                "data": {},
                "msg": validation_result['msg']
            })
        
        # 记录请求日志
        logging.info(f"请求数据 - 预测数据: {data.get('A', {})}, 合作方列表: {data.get('B', [])}")
        
        # 调用预测函数
        result = predict_result(data)
        
        # 记录预测结果
        if result['code'] == 200:
            # 成功时记录详细的概率数据
            prob_data = result.get('data', {})
            logging.info(f"预测成功 - 状态码: 200, 概率结果: {prob_data}")
        else:
            # 失败时记录错误信息
            logging.warning(f"预测失败 - 状态码: {result['code']}, 信息: {result.get('msg', '未知错误')}")
        return jsonify(result)
            
    except Exception as e:
        logging.error(f"预测接口异常 - 状态码: 500, 错误: {str(e)}")
        return jsonify({
            "code": 500,
            "data": {},
            "msg": f"服务器内部错误: {str(e)}"
        })

def validate_prediction_data(data):
    """验证预测数据的合法性"""
    # 检查必需字段
    if 'A' not in data:
        return {"valid": False, "msg": "缺少必需字段: A"}
    
    if 'B' not in data:
        return {"valid": False, "msg": "缺少必需字段: B"}
    
    applicant_data = data.get('A', {})
    candidate_partners = data.get('B', [])
    
    # 检查申请人数据是否为字典
    if not isinstance(applicant_data, dict):
        return {"valid": False, "msg": "字段A必须是对象类型"}
    
    # 检查候选合作方是否为列表
    if not isinstance(candidate_partners, list):
        return {"valid": False, "msg": "字段B必须是数组类型"}
    
    # 检查id字段不能为空
    if not applicant_data.get('id'):
        return {"valid": False, "msg": "id字段不能为空"}
    
    # 检查其他字段的空值情况并记录到日志
    check_empty_fields(applicant_data)
    
    return {"valid": True, "msg": "数据校验通过"}

def check_empty_fields(applicant_data):
    """检查字段空值并记录到日志"""
    # 定义需要检查的字段列表
    fields_to_check = [
        'id', 'amount', 'idInfo.birthDate', 'idInfo.gender', 
        'idInfo.nation', 'idInfo.validityDate', 'degree', 'maritalStatus', 
        'income', 'companyInfo.companyName', 'companyInfo.industry', 
        'companyInfo.occupation', 'jobFunctions', 'province', 'city', 
        'resideFunctions', 'linkmanList.0.relationship', 
        'linkmanList.1.relationship', 'purpose', 'customerSource', 
        'pictureInfo.0.faceScore', 'pictureInfo.1.faceScore', 
        'pictureInfo.2.faceScore', 'deviceInfo.osType',
        'deviceInfo.gpsLatitude', 'deviceInfo.gpsLongitude', 
        'deviceInfo.applyPos', 'deviceInfo.isCrossDomain', 
        'bankCardInfo.bankCode'
    ]
    
    empty_fields = []
    
    for field in fields_to_check:
        value = applicant_data.get(field)
        
        # 检查字段是否为空（None、空字符串、空列表、空字典）
        if value is None or value == "" or value == [] or value == {}:
            empty_fields.append(field)
    
    # 记录空字段到日志（过滤掉id字段，因为id已经单独检查过了）
    if empty_fields:
        empty_fields_without_id = [field for field in empty_fields if field != 'id']
        if empty_fields_without_id:
            logging.warning(f"检测到空字段: {', '.join(empty_fields_without_id)}")

@app.route('/api/fit_with_new_data', methods=['POST'])
def fit_with_new_data():
    """手动触发SFTP数据拉取和增量训练"""
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            logging.warning("手动训练请求 - 状态码: 400, 请求体为空")
            return jsonify({
                "code": 400,
                "data": {},
                "msg": "请求体必须为JSON格式"
            })
        
        # 验证密钥
        request_key = data.get('key', '')
        if not validate_key(request_key):
            logging.warning(f"手动训练请求 - 状态码: 401, 无效API密钥: {request_key}")
            return jsonify({
                "code": 401,
                "data": {},
                "msg": "无效的API密钥"
            })
        
        # 1. 手动调用SFTP拉取函数
        logging.info("开始手动SFTP数据拉取...")
        has_new_files = pull_missing_files_temp()
        
        if not has_new_files:
            logging.info("手动训练 - 状态码: 200, 无新数据，跳过训练")
            return jsonify({
                "code": 200,
                "data": {
                    "has_new_data": False,
                    "timestamp": datetime.now().isoformat()
                },
                "msg": "无新数据可用，跳过训练"
            })
        
        # 2. 有新数据时执行增量训练
        logging.info("检测到新数据，开始增量训练...")
        run_day_pipeline()
        
        logging.info("手动训练成功 - 状态码: 200, 数据拉取和训练完成")
        return jsonify({
            "code": 200,
            "data": {
                "has_new_data": True,
                "timestamp": datetime.now().isoformat(),
                "status": "data_pulled_and_trained"
            },
            "msg": "数据拉取和模型训练完成"
        })
            
    except Exception as e:
        logging.error(f"手动训练接口异常 - 状态码: 500, 错误: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "code": 500,
            "data": {},
            "msg": f"手动训练失败: {str(e)}"
        })
    

@app.before_request
def log_request():
    """记录请求日志"""
    logging.info(f"Request: {request.method} {request.path}")

@app.route('/')
def hello():
    return 'Loan Prediction API Service'

if __name__ == '__main__':
    # 初始化日志
    setup_logging()
    
    # 启动应用
    logging.info("Flask应用启动...")
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=False  # 生产环境设为False
    )