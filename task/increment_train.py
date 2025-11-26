import os
from .increment_model import LoanDistributionModel
from .data_loader import DataLoader


def run_day_pipeline():
    """
    统一入口：
    自动加载 → 预处理 → 特征对齐 → 训练 → 输出结果 → 保存最佳模型
    """

    # 1️⃣ 确定 processed 路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(project_root, "processed")
    loader = DataLoader(processed_root=processed_path)

    # 2️⃣ 自动划分训练 / 测试日期
    train_start, train_end, test_start, test_end = loader.get_train_test_dates()

    print(f"训练集日期: {train_start} → {train_end}")
    print(f"测试集日期: {test_start} → {test_end}")

    # 3️⃣ 加载训练集
    print("\n=== 📌 加载训练数据 ===")
    train_data = loader.load_data_range(train_start, train_end)

    # 4️⃣ 加载测试集
    print("\n=== 📌 加载测试数据 ===")
    test_data = loader.load_data_range(test_start, test_end)

    # 5️⃣ 初始化模型
    model = LoanDistributionModel()
    model.partners = train_data["partner_code"].dropna().unique().tolist()

    # 6️⃣ 数据预处理
    print("\n=== 🔧 数据预处理 ===")
    processed_train_data = model.preprocess_features(train_data)
    processed_test_data = model.preprocess_features(test_data)

    # 7️⃣ 测试集特征对齐（补齐缺失列）
    processed_test_data = processed_test_data.reindex(columns=processed_train_data.columns, fill_value=0)

    # 8️⃣ 准备训练特征和标签
    print("\n=== 🧩 特征准备 ===")
    X_train, Y_train_dict = model.prepare_training_data(processed_train_data, fit_scaler=True)
    X_test, Y_test_dict = model.prepare_training_data(processed_test_data, fit_scaler=False)

    # 9️⃣ 比较策略、训练所有模型、保存 best_model
    print("\n=== 🎯 开始训练并比较策略 ===")
    comparison_results = model.compare_imbalance_strategies(
        X_train, Y_train_dict,
        X_test, Y_test_dict
    )


    return comparison_results
