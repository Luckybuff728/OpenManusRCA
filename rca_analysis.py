
import pandas as pd
import os
from datetime import datetime

# --- 配置 ---

# 故障时间窗口 (UTC)
FAULT_START_TIME = "2025-06-11T15:10:55Z"
FAULT_END_TIME = "2025-06-11T15:31:55Z"

# 数据集的根目录和日期
# 根据时区差异，故障数据位于 2025-06-12 的文件中。
DATA_DATE = "2025-06-12"
BASE_PATH = os.path.join("dataset", "phaseone")

# Hipstershop 的微服务列表
SERVICES = [
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "shippingservice",
    "redis-cart"
]

# --- 辅助函数 ---

def load_parquet_data(file_path: str) -> pd.DataFrame | None:
    """安全地加载 Parquet 文件，如果文件不存在则返回 None。"""
    if not os.path.exists(file_path):
        print(f"⚠️  警告: 文件未找到, 跳过: {file_path}")
        return None
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"❌ 错误: 加载文件失败 {file_path}. 原因: {e}")
        return None

def filter_by_time(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """根据时间戳过滤 DataFrame。"""
    if df is None or "timestamp" not in df.columns:
        return pd.DataFrame()
    # 转换时间字符串为 datetime 对象
    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
    # 确保 timestamp 列是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]

# --- 分析模块 ---

def analyze_service_metrics(start_time: str, end_time: str) -> str | None:
    """
    分析所有服务的指标数据，找出异常服务。
    主要分析 P99 延迟和错误计数。
    """
    print("\n--- 1. 开始分析服务指标 ---")
    anomalous_services = {}

    for service in SERVICES:
        file_path = os.path.join(BASE_PATH, DATA_DATE, "metric-parquet", "apm", "service", f"service_{service}_{DATA_DATE}.parquet")
        df = load_parquet_data(file_path)
        if df is None:
            continue

        fault_df = filter_by_time(df, start_time, end_time)
        if fault_df.empty:
            continue

        # 分析延迟 (rpc.server.duration)
        latency_metrics = fault_df[fault_df["name"] == "rpc.server.duration"]
        if not latency_metrics.empty:
            p99_latency = latency_metrics["value"].quantile(0.99)
            anomalous_services[service] = {"p99_latency": p99_latency}
            print(f"  - 服务 [{service}]: 故障期间 P99 延迟 = {p99_latency:.2f} ms")

        # 分析错误 (假设 status.code = 'ERROR' 标记了一个错误)
        error_metrics = fault_df[fault_df.get("otel.status_code", pd.Series(dtype=str)) == "ERROR"]
        if not error_metrics.empty:
            error_count = error_metrics.shape[0]
            if service in anomalous_services:
                anomalous_services[service]["error_count"] = error_count
            print(f"  - 服务 [{service}]: 发现 {error_count} 个错误。")

    if not anomalous_services:
        print("在指定时间窗口内未发现显著的指标异常。")
        return None

    # 基于 P99 延迟找出最可疑的服务
    most_anomalous_service = max(anomalous_services, key=lambda s: anomalous_services[s].get("p99_latency", 0))
    print(f"\n✅ 指标分析完成: [{most_anomalous_service}] 是最可疑的服务，其 P99 延迟最高。")
    return most_anomalous_service

def analyze_logs(service: str, start_time: str, end_time: str):
    """
    分析指定服务的日志，查找错误或异常信息。
    """
    print(f"\n--- 2. 开始分析 [{service}] 的日志 ---")
    # 注意：日志文件的命名和结构是基于通用模式的推断
    log_file_path = os.path.join(BASE_PATH, DATA_DATE, "log-parquet", f"{service}_{DATA_DATE}.log.parquet")
    log_df = load_parquet_data(log_file_path)

    if log_df is None:
        print(f"无法直接加载服务 [{service}] 的独立日志文件。")
        print("请手动检查 'dataset/phaseone/2025-06-11/log-parquet/' 目录下的文件，确认日志的存储方式。")
        print("分析中止。")
        return

    fault_logs = filter_by_time(log_df, start_time, end_time)
    if fault_logs.empty:
        print("在故障时间窗口内未找到相关日志。")
        return

    # 兼容多种可能的日志内容字段，如 'body' 或 'message'
    log_content_column = None
    if "body" in fault_logs.columns:
        log_content_column = "body"
    elif "message" in fault_logs.columns:
        log_content_column = "message"
    
    if not log_content_column:
        print("日志中未找到 'body' 或 'message' 列，无法分析日志内容。")
        return

    # 查找包含关键词的日志
    keywords = ["error", "exception", "failed", "timeout", "oom", "out of memory"]
    found_logs = fault_logs[fault_logs[log_content_column].str.contains("|".join(keywords), case=False, na=False)]

    if found_logs.empty:
        print("日志中未发现明确的错误关键词 (error, OOM, etc.)。")
    else:
        print("✅ 在日志中发现以下可疑记录:")
        for _, row in found_logs.head().iterrows():
            print(f"  - [{row['timestamp']}] {row[log_content_column]}")

def main():
    """主执行函数"""
    print("==============================================")
    print("= 微服务故障根因分析脚本 (Hipstershop) =")
    print("==============================================")
    print(f"分析时间窗口: {FAULT_START_TIME} to {FAULT_END_TIME}")

    # 步骤 1: 分析指标，找出嫌疑服务
    culprit_service = analyze_service_metrics(FAULT_START_TIME, FAULT_END_TIME)

    if culprit_service:
        # 步骤 2: 分析嫌疑服务的日志
        analyze_logs(culprit_service, FAULT_START_TIME, FAULT_END_TIME)

        # 步骤 3: 最终结论
        print("\n--- 3. 分析结论 ---")
        print(f"综合分析，本次故障的根本原因很可能源于 [{culprit_service}] 服务。")
        print("主要表现为该服务的 P99 延迟在故障期间显著升高。")
        print("请检查上方打印的日志详情以获取更具体的错误信息，例如内存溢出 (OOM) 或其他运行时异常。")
    else:
        print("\n--- 3. 分析结论 ---")
        print("在所有服务中均未检测到明确的异常指标或日志，可能需要更深入地检查基础设施层面或数据完整性。")

if __name__ == "__main__":
    main() 