import paramiko
import os  
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from data_process.daily_partner_processor import LoanDataProcessor
from .increment_train import run_day_pipeline
import traceback
from datetime import datetime, timedelta
import tempfile
from dotenv import load_dotenv

# ====== SFTP 配置 ======
load_dotenv()
REMOTE_DIR = "/upload"
LOCAL_DIR = "../data"
LOG_PATH = "./logs/sftp_pull.log"
SFTP_HOST = os.getenv('SFTP_HOST')
SFTP_PORT = int(os.getenv('SFTP_PORT', 38981))
SFTP_USER = os.getenv('SFTP_USER')
SFTP_PASS = os.getenv('SFTP_PASS')

def write_log(message: str):
    """写日志"""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - {message}\n\n")

def pull_missing_files_temp():
    """
    从 SFTP 下载缺失文件到临时文件处理，处理完自动删除。
    返回:
        has_new_files (bool): 是否有新文件被拉取并处理
    """
    has_new_files = False

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # 获取 processed 文件夹已有日期
    existing_dates = set(
        name for name in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, name))
    )
    print(f"已存在日期: {sorted(existing_dates)}")

    # 连接 SFTP
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))
    transport.connect(username=SFTP_USER, password=SFTP_PASS)
    sftp = paramiko.SFTPClient.from_transport(transport)

    # 获取远程文件列表
    remote_files = [f.filename for f in sftp.listdir_attr(REMOTE_DIR)
                    if f.filename.endswith(".txt")]
    remote_files = [f for f in remote_files if len(f) == 14 and f[:4].isdigit()]

    # 找出缺失文件
    missing_files = [f for f in remote_files if f[:10] not in existing_dates]
    if not missing_files:
        print(" processed 文件夹已经是最新，无需补拉")
        sftp.close()
        transport.close()
        return has_new_files

    print(f"发现缺失文件 {len(missing_files)} 个：")
    for f in missing_files:
        print("  -", f)

    processor = LoanDataProcessor()
    temp_dir = tempfile.gettempdir()  # 系统临时目录

    for filename in missing_files:
        tmp_path = os.path.join(temp_dir, filename)  # 保留原始文件名
        try:
            remote_path = f"{REMOTE_DIR}/{filename}"
            sftp.get(remote_path, tmp_path)
            print(f"\n➡ 下载完成（临时）：{filename}")

            # 处理临时文件
            processor.process_file(tmp_path)
            print(f"✅ 文件 {filename} 处理完成！")
            has_new_files = True

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                print(f"🗑 已删除临时文件: {tmp_path}")

    sftp.close()
    transport.close()
    print("\n全部缺失文件处理完成（临时下载模式）")
    return has_new_files

def daily_task():
    """每天定时拉取 SFTP 数据并处理增量训练"""
    print("开始尝试手动拉取 SFTP 数据文件...")
    try:
        result = pull_missing_files_temp()
    except Exception as e:
        print(f"任务执行出错: {e}")
        write_log(f"任务执行出错: {e}\n{traceback.format_exc()}")
        return

    print("执行结束，请检查 data 文件夹和 logs 日志。")
    if result:
        print("有新文件，开始增量训练...")
        try:
            run_day_pipeline()
        except Exception as e:
            print(f"增量训练出错: {e}")
            write_log(f"增量训练出错: {e}\n{traceback.format_exc()}")
    else:
        print("没有新文件，跳过训练")


# if __name__ == "__main__":
#     write_log("SFTP 定时任务已启动")
#     scheduler = BlockingScheduler()
#     # 每天 09:00 执行
#     scheduler.add_job(daily_task, "cron", hour=9, minute=0)
#     scheduler.start()

if __name__ == "__main__":
    write_log("SFTP 定时任务已启动")
    scheduler = BlockingScheduler()
    # 每分钟执行一次（测试用）
    scheduler.add_job(daily_task, "cron", minute="*")
    scheduler.start()





