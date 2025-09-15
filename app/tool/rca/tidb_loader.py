import glob
import os
import re
from datetime import datetime, timedelta
import json
from typing import Any, Dict, List, Optional, ClassVar

import numpy as np
import pandas as pd

from app.tool.base import ToolResult
from app.tool.rca.base_loader import BaseLoader


def format_bytes(byte_value: float) -> str:
    """将字节数转换为更易读的格式 (KB, MB, GB)"""
    if pd.isna(byte_value):
        return "N/A"
    if abs(byte_value) < 1024:
        return f"{byte_value:.2f} B"
    elif abs(byte_value) < 1024**2:
        return f"{byte_value/1024:.2f} KB"
    elif abs(byte_value) < 1024**3:
        return f"{byte_value/(1024**2):.2f} MB"
    else:
        return f"{byte_value/(1024**3):.2f} GB"


# V30: 引入专家分类的指标库，以取代硬编码的if/else规则
TIDB_METRIC_CATEGORIES = {
    "tidb": {
        "latency": ["duration_99th", "duration_95th", "duration_avg", "slow_query"],
        "throughput": ["qps", "failed_query_ops"],
        "resource": ["cpu_usage", "memory_usage"],
        "internal": ["connection_count", "block_cache_size", "transaction_retry", "top_sql_cpu"],
        "health": ["server_is_up", "uptime"],
    },
    "tikv": {
        "latency": ["raft_propose_wait", "raft_apply_wait", "grpc_qps"],
        "storage_io": ["read_mbps", "write_wal_mbps", "io_util"],
        "resource": ["cpu_usage", "memory_usage", "threadpool_readpool_cpu"],
        "raft_state": ["region_pending", "snapshot_apply_count"],
        "storage_engine": ["rocksdb_write_stall"],
        "storage_capacity": ["store_size", "available_size", "capacity_size"],
        "health": ["server_is_up"],
    },
    "pd": {
        "cluster_state": ["store_up_count", "store_down_count", "store_unhealth_count", "store_low_space_count", "store_slow_count", "witness_count"],
        "region_state": ["abnormal_region_count", "region_health", "region_count", "leader_count", "learner_count"],
        "storage": ["storage_capacity", "storage_size", "storage_used_ratio"],
        "resource": ["cpu_usage", "memory_usage"],
        "health": ["leader_primary"],
    }
}

# V30: 定义零容忍告警指标集合
ZERO_TOLERANCE_ALERTS = {
    'rocksdb_write_stall', 'store_down_count', 'store_unhealth_count', 
    'store_slow_count', 'transaction_retry', 'server_is_up'
}

class TiDBMetricsLoader(BaseLoader):
    """加载并分析TiDB、TiKV、PD相关的监控指标，以支持根因定位。"""

    name: str = "tidb_loader"
    description: str = "加载并分析TiDB、TiKV、PD的性能和状态指标，报告所有发生变化的指标。"
    
    # V8: 引入统计学异常检测的Sigma阈值
    SIGMA_THRESHOLD: ClassVar[float] = 3.0

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "description": "故障案例的唯一标识符。"},
            "start_time": {"type": "string", "description": "开始时间，UTC格式。"},
            "end_time": {"type": "string", "description": "结束时间，UTC格式。"},
            "components": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要分析的TiDB组件列表 (例如, 'tidb', 'tikv', 'pd')。如果为空，则分析所有支持的组件。"
            },
        },
        "required": ["uuid", "start_time", "end_time"],
    }
    
    def _find_tidb_metric_files(self, start_dt: datetime, end_dt: datetime, components: List[str]) -> List[str]:
        """V14: 决定性修复 - 使用glob和绝对路径构建，放弃有问题的辅助函数，确保路径正确性。"""
        all_files = set()
        cst_tz = timedelta(hours=8)
        start_dt_cst = start_dt + cst_tz
        end_dt_cst = end_dt + cst_tz
        dates_to_check = pd.date_range(start_dt_cst.date(), end_dt_cst.date())
        
        # 获取数据集的根目录
        # FIXME: This is a bit of a hack, assuming a fixed structure.
        # A more robust solution might involve a config or environment variable.
        base_path = os.path.join(os.getcwd(), 'dataset', 'phaseone')

        for date in dates_to_check:
            date_str = date.strftime("%Y-%m-%d")
            # 完整日期目录
            day_path = os.path.join(base_path, date_str)

            for component in components:
                search_paths = []
                if component == 'tidb':
                    search_paths.append(os.path.join(day_path, "metric-parquet", "infra", "infra_tidb"))
                elif component in ['pd', 'tikv']:
                    search_paths.append(os.path.join(day_path, "metric-parquet", "other"))
                
                for path in search_paths:
                    if os.path.exists(path):
                        pattern = os.path.join(path, f"infra_{component}_*.parquet")
                        all_files.update(glob.glob(pattern))
                        
        return list(all_files)

    def _load_and_process_tidb_data(self, files: List[str], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        加载并处理TiDB指标数据，同时从文件名中提取并保留组件类型信息。
        """
        df = self._load_data(files, "tidb", start_dt, end_dt)
        if df.empty:
            return pd.DataFrame()

        all_dfs = []
        for file_name, group_df in df.groupby('source_file'):
            try:
                base_name = os.path.basename(file_name)
                
                inferred_type = 'unknown'
                if 'infra_tidb' in base_name: inferred_type = 'tidb'
                elif 'infra_pd' in base_name: inferred_type = 'pd'
                elif 'infra_tikv' in base_name: inferred_type = 'tikv'

                # V10: 使用正则表达式替换脆弱的rsplit，以健壮地处理带或不带下划线的日期后缀
                stem = os.path.splitext(base_name)[0]
                key_part_with_prefix = re.sub(r'_?\d{4}-\d{2}-\d{2}$', '', stem)
                
                prefixes = ["infra_tidb_", "infra_pd_", "infra_tikv_"]
                kpi_key = key_part_with_prefix
                for prefix in prefixes:
                    if kpi_key.startswith(prefix):
                        kpi_key = kpi_key[len(prefix):]
                        break
                
                temp_df = group_df.copy()
                if not temp_df.empty:
                    # V11: 决定性修复 - 直接使用kpi_key作为列名读取数据，然后统一放入'value'列
                    if kpi_key in temp_df.columns:
                        temp_df['value'] = temp_df[kpi_key]
                    else:
                        # 如果文件中连与kpi_key同名的列都没有，则此文件无效
                            continue
                    
                    temp_df['kpi_key'] = kpi_key
                    temp_df['inferred_component_type'] = inferred_type
                    all_dfs.append(temp_df)
            except Exception:
                continue

        if not all_dfs:
            return pd.DataFrame()
        
        df = pd.concat(all_dfs, ignore_index=True)
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
        # 保留 instance 字段，因为它用于标识具体的组件实例
        # df.rename(columns={'instance': 'unified_component_id'}, inplace=True)
        return df


    async def execute(self, uuid: str, start_time: str, end_time: str, components: Optional[List[str]] = None) -> ToolResult:
        try:
            start_dt, end_dt = self._parse_time_range(start_time, end_time)

            # --- 最佳实践 (V3): 带有隔离带的窗口设置 ---
            # 1. 扩展异常窗口以保证数据完整性
            MIN_ANOMALY_DURATION_SECONDS = 120
            if (end_dt - start_dt).total_seconds() < MIN_ANOMALY_DURATION_SECONDS:
                end_dt = start_dt + timedelta(seconds=MIN_ANOMALY_DURATION_SECONDS)

            # 2. 定义包含隔离带的基线窗口
            QUARANTINE_MINUTES = 5
            BASELINE_WINDOW_MINUTES = 10
            baseline_end_dt = start_dt - timedelta(minutes=QUARANTINE_MINUTES)
            baseline_start_dt = baseline_end_dt - timedelta(minutes=BASELINE_WINDOW_MINUTES)

            # 3. 定义完整的数据加载窗口
            full_load_start_dt = baseline_start_dt
            full_load_end_dt = end_dt
            
            if not components:
                # 移除对 IMPORTANT_METRICS 的依赖, 硬编码支持的组件类型
                components = ['tidb', 'tikv', 'pd']

            all_files = self._find_tidb_metric_files(full_load_start_dt, full_load_end_dt, components)
            if not all_files:
                return ToolResult(output=json.dumps({
                    "summary": "未找到任何TiDB相关组件的指标文件。",
                    "components_with_changes": []
                }))

            # 使用专门的TiDB数据处理方法
            df = self._load_and_process_tidb_data(all_files, full_load_start_dt, full_load_end_dt)
            if df.empty:
                return ToolResult(output=json.dumps({
                    "summary": "加载的TiDB指标数据为空。",
                    "components_with_changes": []
                }))

            # V22: 关键修正 - 将字符串'null'替换为实际的NaN，以确保后续的fillna能正常工作
            for col in ['instance', 'pod', 'kubernetes_node']:
                if col in df.columns:
                    df[col] = df[col].replace('null', np.nan)
            
            # V18: 修正 - 处理在 'kubernetes_node' 中包含真实节点名的特殊指标
            # 对于 io_util 等指标, 'instance' 可能是 exporter 地址, 而 'kubernetes_node' 才是物理节点
            if 'kubernetes_node' in df.columns and 'kpi_key' in df.columns:
                # 定义需要此特殊处理的指标列表
                node_level_kpis = ['io_util'] 
                
                # 创建一个布尔掩码来定位需要修正的行
                condition = (df['kpi_key'].isin(node_level_kpis)) & (df['kubernetes_node'].notna()) & (df['kubernetes_node'] != '')
                
                # 对于匹配的行，用 'kubernetes_node' 的值覆盖 'instance'
                df.loc[condition, 'instance'] = df.loc[condition, 'kubernetes_node']

            # V17: 重构节点与组件实例的识别逻辑，根除 related_node 为 null 的问题
            # V21: 再次重构，确立 kubernetes_node 为物理节点的唯一真实来源
            placeholder = 'N/A_cluster_level'
            
            # 步骤 0: 如果 'kubernetes_node' 存在，则它就是最可信的物理节点来源。
            # 我们将它的值直接赋给 'related_node' 列，后续所有操作都围绕此列。
            if 'kubernetes_node' in df.columns and df['kubernetes_node'].notna().any():
                df['related_node'] = df['kubernetes_node']
            else:
                # 如果 'kubernetes_node' 不存在，则创建一个空列作为占位符
                df['related_node'] = pd.NA

            # 步骤 1: 统一 'pod' 列作为组件实例的唯一标识
            # TiDB/TiKV/PD 指标中，'instance' 列通常包含组件实例名 (如 'tidb-pd-0', '10.233.79.158:10080')
            if 'pod' not in df.columns and 'instance' in df.columns:
                df['pod'] = df['instance']
            
            if 'pod' in df.columns:
                df['pod'] = df['pod'].fillna(placeholder)
            else:
                df['pod'] = placeholder

            # 步骤 2: 填充 'related_node' 的缺失值
            # 如果 'related_node' 在某些行上是缺失的 (例如，来自没有 kubernetes_node 列的文件)，
            # 我们使用 'instance' 列（如果存在）或 'pod' 列作为备用填充。
            # 这确保了 related_node 最终总有一个有意义的值。
            if 'instance' in df.columns:
                df['related_node'] = df['related_node'].fillna(df['instance'])
            df['related_node'] = df['related_node'].fillna(df['pod'])


            # V20: 为处理子类型指标，添加 'type' 列
            # 许多指标（如QPS、abnormal_region_count）根据'type'列细分为不同子项
            if 'type' not in df.columns:
                df['type'] = 'N/A'
            else:
                df['type'] = df['type'].fillna('N/A')


            # --- 存在性分析 ---
            # 检查请求的组件是否在数据中存在
            # 优先使用数据中的 object_type 字段，如果没有则使用推断的类型
            if 'object_type' in df.columns:
                found_components = df['object_type'].unique().tolist()
            else:
                found_components = df['inferred_component_type'].unique().tolist()
            
            # 如果指定了特定组件，检查哪些缺失
            if components:
                pass
            else:
                # 全局查询：不检查缺失
                pass

            # 4. 分割基线和异常数据
            time_col = self._get_time_column("metrics")
            baseline_df = df[(df[time_col] >= baseline_start_dt) & (df[time_col] < baseline_end_dt)].copy()
            anomaly_df = df[(df[time_col] >= start_dt) & (df[time_col] <= end_dt)].copy()

            if baseline_df.empty:
                baseline_df = pd.DataFrame(columns=df.columns)

            if anomaly_df.empty:
                return ToolResult(output=json.dumps({
                    "summary": "在指定故障时间窗口内没有找到TiDB指标数据。",
                    "components_with_changes": []
                }))

            # 5. 执行基线对比分析
            analysis_result = self._compare_with_baseline(baseline_df, anomaly_df)
            
            # 6. 生成最终报告
            report = self._create_comparison_report(analysis_result, anomaly_df)
            report = self.convert_numpy_types(report)

            return ToolResult(output=json.dumps(report, indent=2, ensure_ascii=False))

        except Exception as e:
            import traceback
            return ToolResult(error=f"TiDB指标加载器执行出错: {str(e)}\n{traceback.format_exc()}")

    def _compare_with_baseline(self, baseline_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> pd.DataFrame:
        # V31: 增加对计数器指标的速率转换
        COUNTER_METRICS = [
            'failed_query_ops', 'transaction_retry', 'slow_query', 
            'snapshot_apply_count', 'store_down_count', 'store_unhealth_count', 
            'store_slow_count'
        ]

        def convert_counters_to_rates(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            
            is_counter = df['kpi_key'].isin(COUNTER_METRICS)
            gauges_df = df[~is_counter].copy()
            counters_df = df[is_counter].copy()

            if counters_df.empty:
                return gauges_df

            group_keys_for_rate = ['pod', 'related_node', 'kpi_key', 'inferred_component_type', 'type']
            counters_df.sort_values(group_keys_for_rate + ['time'], inplace=True)
            
            time_col = self._get_time_column("metrics")
            value_col = 'value'
            
            counters_df['value_diff'] = counters_df.groupby(group_keys_for_rate)[value_col].diff()
            counters_df['time_diff'] = counters_df.groupby(group_keys_for_rate)[time_col].diff().dt.total_seconds()

            counters_df[value_col] = counters_df.apply(
                lambda row: row['value_diff'] / row['time_diff'] if row['time_diff'] > 0 else 0,
                axis=1
            )
            
            counters_df['kpi_key'] = counters_df['kpi_key'] + '_per_second'
            
            counters_df.drop(columns=['value_diff', 'time_diff'], inplace=True)
            counters_df.dropna(subset=[value_col], inplace=True)

            return pd.concat([gauges_df, counters_df], ignore_index=True)

        baseline_df = convert_counters_to_rates(baseline_df)
        anomaly_df = convert_counters_to_rates(anomaly_df)

        # V20: 将 'type' 加入group_keys, 以分别分析每个指标的子类型
        # V24: 关键修正 - 必须将 related_node 加入 group_keys, 否则该信息会在聚合过程中丢失
        group_keys = ['pod', 'related_node', 'kpi_key', 'inferred_component_type', 'type']
        
        # V9 & V12: 预处理已上移到execute方法，此处不再需要填充
        # placeholder = 'N/A_cluster_level'
        # if 'instance' in baseline_df.columns:
        #     baseline_df['instance'] = baseline_df['instance'].fillna(placeholder)
        # if 'instance' in anomaly_df.columns:
        #     anomaly_df['instance'] = anomaly_df['instance'].fillna(placeholder)

        # 确保group_keys在DataFrame中存在
        for key in group_keys:
            if key not in baseline_df.columns:
                baseline_df[key] = 'N/A' # 保持一个默认值以防万一
            if key not in anomaly_df.columns:
                anomaly_df[key] = 'N/A'


        if not baseline_df.empty:
            # V37: 计算更丰富的基线统计数据
            baseline_metrics = baseline_df.groupby(group_keys)['value'].agg(['mean', 'std', lambda x: x.quantile(0.99)]).reset_index()
            baseline_metrics.rename(columns={'mean': 'baseline_value', 'std': 'baseline_std', '<lambda_0>': 'baseline_p99'}, inplace=True)
        else:
            baseline_metrics = pd.DataFrame(columns=group_keys + ['baseline_value', 'baseline_std', 'baseline_p99'])

        # V3 核心改造: 异常窗口聚合更多统计维度
        anomaly_metrics = anomaly_df.groupby(group_keys)['value'].agg(['mean', 'max', 'min', 'std']).reset_index()
        anomaly_metrics.rename(columns={
            'mean': 'anomaly_value',
            'max': 'anomaly_max',
            'min': 'anomaly_min',
            'std': 'anomaly_std'
        }, inplace=True)

        comparison_df = pd.merge(anomaly_metrics, baseline_metrics, on=group_keys, how='left')
        
        # V37: 统一填充NA值
        comparison_df.fillna(0, inplace=True)
        
        comparison_df['change_percentage'] = comparison_df.apply(
            lambda row: 99999.0 if np.isclose(row['baseline_value'], 0) and row['anomaly_value'] > 0.001 else (
                ((row['anomaly_value'] - row['baseline_value']) / row['baseline_value'] * 100) if not np.isclose(row['baseline_value'], 0) else 0.0
            ),
            axis=1
        )

        return comparison_df

    def _is_statistically_significant(self, current_val: float, baseline_val: float, baseline_std: float) -> bool:
        """
        V8: 使用3-Sigma原则判断当前值相对于基线是否为统计学显著异常。
        V31: 关键修复 - 避免报告从0到0的无意义增长，并为稳定基线增加健壮性检查。
        """
        if np.isclose(current_val, 0) and np.isclose(baseline_val, 0):
            return False

        if baseline_std < 1e-6:
            if np.isclose(baseline_val, 0):
                return not np.isclose(current_val, 0)
            
            relative_change = abs(current_val - baseline_val) / baseline_val
            return relative_change > 0.5

        return (current_val > baseline_val + self.SIGMA_THRESHOLD * baseline_std) or \
               (current_val < baseline_val - self.SIGMA_THRESHOLD * baseline_std)

    # V30: 引入新的、基于类别的动态分析函数
    def _analyze_metric_category(
        self,
        category_name: str,
        category_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        分析单一组件在特定指标类别下的所有指标。
        如果发现任何显著异常，则返回一个包含详细信息的字典列表。
        """
        anomalies = []

        # V37: 定义容易出现尖峰的TiDB相关指标
        SPIKY_METRICS = [
            'qps', 'failed_query_ops_per_second', 'cpu_usage', 'connection_count', 
            'transaction_retry_per_second', 'top_sql_cpu', 'slow_query_per_second',
            'raft_propose_wait', 'raft_apply_wait', 'grpc_qps',
            'read_mbps', 'write_wal_mbps', 'io_util',
            'threadpool_readpool_cpu', 'region_pending', 
            'snapshot_apply_count_per_second'
        ]

        for _, row in category_df.iterrows():
            metric = row['kpi_key']
            metric_type = row.get('type', 'N/A')
            current_val = row['anomaly_value']
            baseline_val = row['baseline_value']
            baseline_std = row['baseline_std']
            max_val = row['anomaly_max']
            min_val = row['anomaly_min']

            report_str = None
            severity = "minor" # V40: 默认严重等级为次要
            
            clean_metric_name = metric.replace('_per_second', '')
            metric_name_for_report = f"「{clean_metric_name}" + (f" (type: {metric_type})」" if metric_type != 'N/A' else "」")

            # V38: uptime 只应在发生重置（值下降）时报告
            if metric == 'uptime':
                if min_val < baseline_val and not np.isclose(baseline_val, 0):
                    report_str = f"{metric_name_for_report}可能发生重置, 从基线 ~{baseline_val/3600:.1f}h 降至最低 ~{min_val/3600:.1f}h"
                    severity = "major" # Uptime重置是主要事件
            
            # 规则1: 零容忍告警
            elif clean_metric_name in ZERO_TOLERANCE_ALERTS:
                 # server_is_up 从 1->0 是严重问题, 其他零容忍指标 > 0 是严重问题
                is_critical_alert = (clean_metric_name == 'server_is_up' and max_val < 1 and baseline_val >= 1) or \
                                    (clean_metric_name != 'server_is_up' and max_val > 0)
                if is_critical_alert:
                    report_str = f"{metric_name_for_report}出现严重告警, 值为 {max_val:.2f}"
                    severity = "critical"
            
            # 规则2: 专家知识 - abnormal_region_count
            elif metric == 'abnormal_region_count' and max_val > 0:
                 benign_types = ['miss-peer-region-count', 'pending-peer-region-count', 'learner-peer-region-count']
                 is_benign = metric_type in benign_types
                 report_str = f"{metric_name_for_report}出现{'低优先级观察项' if is_benign else '严重告警'}, 值为 {max_val:.0f}"
                 severity = "minor" if is_benign else "major"

            # V37 & V38: 规则3 - 尖峰检测 (增加相对变化阈值)
            elif metric in SPIKY_METRICS:
                anomaly_max = row['anomaly_max']
                baseline_p99 = row['baseline_p99']
                dynamic_threshold = baseline_p99 + (2 * baseline_std)
                
                is_spike = False
                if anomaly_max > dynamic_threshold and anomaly_max > 0.01:
                    # 如果基线不为零，则要求有显著的相对增幅 (例如 > 20%)
                    if baseline_p99 > 1e-6:
                        relative_increase = (anomaly_max - baseline_p99) / baseline_p99
                        if relative_increase > 0.2:
                            is_spike = True
                    else: # 基线为零，任何显著值都是尖峰
                        is_spike = True

                if is_spike:
                    severity = "major" # 尖峰总是至少是主要问题
                    if ("usage" in metric or "util" in metric) and anomaly_max > 0.9:
                        severity = "critical" # 超过90%的利用率是严重问题

                    if "bytes" in metric or "size" in metric or "mbps" in metric:
                        report_str = f"{metric_name_for_report}出现异常尖峰，达到 {format_bytes(anomaly_max)} (正常上限: {format_bytes(dynamic_threshold)})"
                    elif "rate" in metric or "usage" in metric or "ratio" in metric or "util" in metric:
                        report_str = f"{metric_name_for_report}出现异常尖峰，达到 {anomaly_max*100:.1f}% (正常上限: {dynamic_threshold*100:.1f}%)"
                    elif "duration" in metric or "wait" in metric:
                        report_str = f"{metric_name_for_report}出现异常尖峰，达到 {anomaly_max*1000:.1f}ms (正常上限: {dynamic_threshold*1000:.1f}ms)"
                    elif "_per_second" in metric:
                        report_str = f"{metric_name_for_report}速率出现异常尖峰，达到 {anomaly_max:.2f}/s (正常上限: {dynamic_threshold:.2f}/s)"
                    else:
                        report_str = f"{metric_name_for_report}出现异常尖峰，达到 {anomaly_max:.2f} (正常上限: {dynamic_threshold:.2f})"

            # 规则4: 通用统计显著性分析 (均值对比)
            if report_str is None: # V38: 确保在没有生成特定报告时才执行通用检查
                is_significant = self._is_statistically_significant(current_val, baseline_val, baseline_std)
                if is_significant:
                    severity = "major" # 统计显著变化视为主要问题
                    # 只有在之前的规则没有生成报告的情况下，才使用通用报告
                    if report_str is None:
                        if "_per_second" in metric:
                            report_str = f"{metric_name_for_report}速率从 {baseline_val:.2f}/s 变化至 {current_val:.2f}/s"
                        elif "bytes" in metric or "size" in metric:
                            report_str = f"{metric_name_for_report}从 {format_bytes(baseline_val)} 变化至 {format_bytes(current_val)}"
                        elif "rate" in metric or "usage" in metric or "ratio" in metric:
                             report_str = f"{metric_name_for_report}从 {baseline_val*100:.1f}% 变化至 {current_val*100:.1f}%"
                        elif "duration" in metric or "wait" in metric:
                             report_str = f"{metric_name_for_report}延迟从 {baseline_val*1000:.1f}ms 变化至 {current_val*1000:.1f}ms"
                        else:
                             report_str = f"{metric_name_for_report}从 {baseline_val:.1f} 变化至 {current_val:.1f}"

            if report_str:
                anomalies.append({
                    "description": report_str,
                    "severity": severity,
                    "category": category_name,
                    "metric": clean_metric_name,
                    "value": max_val
                })
        
        return anomalies

    # V30: 全面重构报告生成逻辑，采用按类别动态分析范式
    def _create_comparison_report(self, comparison_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> Dict[str, Any]:
        reports = []
        
        for (pod_id, component_type, node_identifier), group in comparison_df.groupby(['pod', 'inferred_component_type', 'related_node']):
            component_anomalies = []
            
            # 遍历该组件类型的所有专家类别
            categories_to_check = TIDB_METRIC_CATEGORIES.get(component_type, {})
            for category_name, metric_keys in categories_to_check.items():
                # V40: 将 kpi_key 列中的 "_per_second" 后缀去掉，以匹配原始 metric_keys
                group_copy = group.copy()
                group_copy['kpi_key_base'] = group_copy['kpi_key'].str.replace('_per_second', '')
                category_df = group_copy[group_copy['kpi_key_base'].isin(metric_keys)]
                
                if not category_df.empty:
                    category_analysis = self._analyze_metric_category(category_name, category_df)
                    if category_analysis:
                        component_anomalies.extend(category_analysis)

            # V30 & V40: 跨类别关联分析 (专家系统核心)
            if component_anomalies:
                # 规则: TiKV 高IO利用率 + 高延迟 = IO瓶颈 (严重)
                if component_type == 'tikv':
                    # 从已检测到的异常中查找相关指标
                    anomalies_map = {a['metric']: a for a in component_anomalies}
                    io_util = anomalies_map.get('io_util', {}).get('value', 0)
                    apply_wait = anomalies_map.get('raft_apply_wait', {}).get('value', 0)
                
                    if io_util > 0.8 and apply_wait > 0.1: # 80% util, 100ms wait
                        bottleneck_report = (
                            f"检测到高IO利用率(峰值 {io_util*100:.0f}%) "
                            f"同时伴随着高Raft Apply延迟(峰值 {apply_wait*1000:.0f}ms)。"
                        )
                        # 添加一个更高优先级的诊断结论
                        component_anomalies.append({
                            "description": bottleneck_report,
                            "severity": "critical",
                            "category": "io_bottleneck",
                            "metric": "io_bottleneck_hypothesis",
                            "value": 1
                        })
                
                # 按严重程度排序
                severity_order = {"critical": 0, "major": 1, "minor": 2}
                component_anomalies.sort(key=lambda x: severity_order.get(x["severity"], 99))

                reports.append({
                    "component": pod_id if pod_id != 'N/A_cluster_level' else f"tidb-{component_type}",
                    "type": component_type,
                    "related_node": node_identifier,
                    "anomalies": component_anomalies
                })

        # V40: 基于严重等级的智能摘要
        summary_parts = []
        if reports:
            # 按组件的最高严重等级排序
            severity_order = {"critical": 0, "major": 1, "minor": 2}
            reports.sort(key=lambda r: severity_order.get(r['anomalies'][0]['severity'], 99))

            critical_reports = [r for r in reports if r['anomalies'][0]['severity'] == 'critical']
            major_reports = [r for r in reports if r['anomalies'][0]['severity'] == 'major']
            
            if critical_reports:
                top_critical = critical_reports[0]
                summary_parts.append(
                    f"**检测到严重异常**: TiDB组件「{top_critical['component']}」({top_critical['type']}) "
                    f"出现严重问题: {top_critical['anomalies'][0]['description']}。"
                )
            elif major_reports:
                top_major = major_reports[0]
                summary_parts.append(
                    f"**检测到主要异常**: TiDB组件「{top_major['component']}」({top_major['type']}) "
                    f"出现显著性能问题: {top_major['anomalies'][0]['description']}。"
                )

            total_anomalous_components = len(reports)
            if total_anomalous_components > 1:
                summary_parts.append(f"总共在 {total_anomalous_components} 个TiDB组件上检测到不同程度的指标异常。")
        
        summary = " ".join(summary_parts) if summary_parts else "无显著的TiDB相关指标异常。"
        return {"summary": summary, "anomalous_components": reports}

    def _get_time_column(self, data_type: str) -> Optional[str]:
        return "time"
    
    def _get_relevant_columns(self, data_type: str, available_columns: List[str]) -> List[str]:
        return available_columns 