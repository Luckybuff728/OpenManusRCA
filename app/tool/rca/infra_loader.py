import glob
import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from app.tool.base import ToolResult
from app.tool.rca.base_loader import BaseLoader
from collections import defaultdict

# V20: 用带有专家类别的结构替换扁平化的指标列表
METRIC_CATEGORIES = {
    "node": {
        "cpu": ["node_cpu_usage_rate"],
        "memory": ["node_memory_usage_rate", "node_memory_MemAvailable_bytes", "node_memory_MemTotal_bytes"],
        "disk": [
            "node_disk_read_bytes_total", "node_disk_written_bytes_total",
            "node_disk_read_time_seconds_total", "node_disk_write_time_seconds_total",
            "node_filesystem_free_bytes", "node_filesystem_size_bytes", "node_filesystem_usage_rate"
        ],
        "network": [
            "node_network_receive_bytes_total", "node_network_transmit_bytes_total",
            "node_network_receive_packets_total", "node_network_transmit_packets_total",
            "node_sockstat_TCP_inuse"
        ]
    },
    "pod": {
        "cpu": ["pod_cpu_usage"],
        "memory": ["pod_memory_working_set_bytes"],
        "disk": ["pod_fs_reads_bytes_total", "pod_fs_writes_bytes_total"],
        "network": ["pod_network_receive_bytes_total", "pod_network_transmit_bytes_total", "pod_network_receive_packets_total", "pod_network_transmit_packets_total"],
        "process": ["pod_processes"]
    }
}


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


class InfraMetricsLoader(BaseLoader):
    """加载并分析基础设施（节点和Pod）的监控指标，以支持根因定位。"""

    name: str = "infra_loader"
    description: str = "加载并分析节点(Node)和容器(Pod)的基础设施指标，报告所有显著的异常。"

    SIGMA_THRESHOLD: ClassVar[float] = 3.0

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "description": "故障案例的唯一标识符"},
            "start_time": {
                "type": "string",
                "description": "开始时间，UTC格式 (如 2025-06-05T16:00:00Z)",
            },
            "end_time": {
                "type": "string",
                "description": "结束时间，UTC格式 (如 2025-06-05T18:00:00Z)",
            },
            "components": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选，指定要加载的组件名称列表。如果未提供，将分析所有组件。",
            },
        },
        "required": ["uuid", "start_time", "end_time"],
    }

    _instance_to_node_map: Dict[str, str] = {}
    _node_to_instance_map: Dict[str, str] = {} 

    FORBIDDEN_COMPONENTS: ClassVar[Set[str]] = {"tidb", "tikv", "pd"}

    def _add_unified_component_id(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        for col in ['pod', 'kubernetes_node', 'instance']:
            if col not in df.columns:
                df[col] = None

        conditions = [
            df['pod'].notna() & (df['pod'] != 'null') & (df['pod'] != ''),
            df['kubernetes_node'].notna() & (df['kubernetes_node'] != 'null') & (df['kubernetes_node'] != '')
        ]
        choices = [
            df['pod'],
            df['kubernetes_node']
        ]
        
        df['unified_component_id'] = np.select(conditions, choices, default=df['instance'])

        if 'pod_ip' in df.columns:
            df['ip_address'] = df['pod_ip']
        elif 'instance' in df.columns:
            df['ip_address'] = df['instance'].str.split(':').str[0]
        else:
            df['ip_address'] = None
        
        if 'instance' in df.columns:
            if 'kubernetes_node' not in df.columns:
                df['kubernetes_node'] = None
            df['kubernetes_node'] = df.apply(
                lambda row: row['instance'] if pd.isna(row['kubernetes_node']) or row['kubernetes_node'] == 'null' else row['kubernetes_node'],
                axis=1
            )

        return df
    
    def _load_and_process_infra_data(self, files: List[str], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        df = self._load_data(files, "metrics", start_dt, end_dt)
        if df.empty:
            return pd.DataFrame()

        all_dfs = []
        for file_name, group_df in df.groupby('source_file'):
            try:
                base_name = os.path.basename(file_name)
                
                inferred_type = 'unknown'
                if 'infra_node' in base_name: inferred_type = 'node'
                elif 'infra_pod' in base_name: inferred_type = 'pod'
                
                key_part_with_prefix = base_name.rsplit('.', 1)[0].rsplit('_', 1)[0]
                prefixes = ["infra_pod_", "infra_node_"]
                kpi_key = key_part_with_prefix
                for prefix in prefixes:
                    if kpi_key.startswith(prefix):
                        kpi_key = kpi_key[len(prefix):]
                        break
                
                temp_df = group_df.copy()
                if not temp_df.empty:
                    if kpi_key in temp_df.columns:
                        temp_df['value'] = temp_df[kpi_key]
                    elif 'value' in temp_df.columns:
                        pass
                    else:
                        continue
                    
                    temp_df['kpi_key'] = kpi_key
                    temp_df['inferred_component_type'] = inferred_type
                    all_dfs.append(temp_df)
            except Exception:
                continue

        if not all_dfs:
            return pd.DataFrame()
        
        return pd.concat(all_dfs, ignore_index=True)


    async def execute(
        self,
        uuid: str,
        start_time: str,
        end_time: str,
        components: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        try:
            if "resource_names" in kwargs and kwargs["resource_names"]:
                if not components:
                    components = kwargs["resource_names"]
                    self.logger.warning(
                        f"参数警告: 'resource_names' 是一个非预期的参数，但它包含了有效值。"
                        f"已将其值 '{components}' 用于 'components' 参数以继续执行。"
                        "请考虑修正调用方代码，直接使用 'components' 参数。"
                    )

            if components:
                forbidden_requested = self.FORBIDDEN_COMPONENTS.intersection(set(components))
                if forbidden_requested:
                    return ToolResult(
                        error=f"逻辑错误: infra_loader 被用于查询被禁止的组件: {list(forbidden_requested)}。请改用 tidb_loader 来查询所有TiDB相关的指标。"
                    )

            start_dt, end_dt = self._parse_time_range(start_time, end_time)

            MIN_ANOMALY_DURATION_SECONDS = 120
            if (end_dt - start_dt).total_seconds() < MIN_ANOMALY_DURATION_SECONDS:
                end_dt = start_dt + timedelta(seconds=MIN_ANOMALY_DURATION_SECONDS)

            QUARANTINE_MINUTES = 5
            BASELINE_WINDOW_MINUTES = 10
            baseline_end_dt = start_dt - timedelta(minutes=QUARANTINE_MINUTES)
            baseline_start_dt = baseline_end_dt - timedelta(minutes=BASELINE_WINDOW_MINUTES)

            full_load_start_dt = baseline_start_dt
            full_load_end_dt = end_dt

            all_files = self._find_infra_metric_files(full_load_start_dt, full_load_end_dt)
            if not all_files:
                return ToolResult(output=json.dumps({"summary": "未找到指定时间范围内的基础设施指标文件。", "anomalous_components": []}))

            df = self._load_and_process_infra_data(all_files, full_load_start_dt, full_load_end_dt)

            if df.empty:
                return ToolResult(output=json.dumps({"summary": "加载的基础设施指标数据为空。", "anomalous_components": []}))

            df = self._add_unified_component_id(df)
            
            df['value'] = pd.to_numeric(df.get('value'), errors='coerce')
            df.dropna(subset=['time', 'value', 'unified_component_id'], inplace=True)

            initial_components = []
            if components:
                initial_components = list(components)
            
            found_components = df['unified_component_id'].unique().tolist()
            
            missing_components = []
            if initial_components:
                missing_components = [c for c in initial_components if c not in found_components]

            if components:
                regex_pattern = '|'.join(map(re.escape, components))
                df = df[df['unified_component_id'].str.contains(regex_pattern, na=False, regex=True)]
            
            found_components_df = pd.DataFrame()
            missing_components = []
            
            if components:
                all_found_ids = df['unified_component_id'].unique()
                found_components_list = [comp for comp in components if comp in all_found_ids]
                missing_components = [comp for comp in components if comp not in all_found_ids]
                
                if not found_components_list:
                    summary_msg = f"按组件 {components} 筛选后无数据。这可能意味着这些服务已离线或未上报指标。"
                    return ToolResult(output=json.dumps({
                        "summary": summary_msg, 
                        "has_anomaly": True,
                        "anomalous_components": [],
                    }))
                
                found_components_df = df[df['unified_component_id'].isin(found_components_list)]
            else:
                found_components_df = df

            time_col = self._get_time_column("metrics")
            baseline_df = found_components_df[(found_components_df[time_col] >= baseline_start_dt) & (found_components_df[time_col] < baseline_end_dt)].copy()
            anomaly_df = found_components_df[(found_components_df[time_col] >= start_dt) & (found_components_df[time_col] <= end_dt)].copy()

            if anomaly_df.empty:
                summary_msg = "在指定的故障时间窗口内没有找到基础设施指标数据。"
                if missing_components:
                    summary_msg += f" 此外，组件 {missing_components} 在整个查询期间均无数据。"
                return ToolResult(output=json.dumps({
                    "summary": summary_msg, 
                    "has_anomaly": bool(missing_components),
                    "anomalous_components": [],
                }))

            analysis_result = self._compare_with_baseline(baseline_df, anomaly_df)

            report = self._create_comparison_report(analysis_result, anomaly_df, missing_components)
            report = self.convert_numpy_types(report)

            return ToolResult(output=json.dumps(report, indent=2, ensure_ascii=False))

        except Exception as e:
            import traceback
            return ToolResult(
                error=f"基础设施指标加载与分析错误: {str(e)}\n{traceback.format_exc()}"
            )
            
    def _compare_with_baseline(self, baseline_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> pd.DataFrame:
        COUNTER_METRICS = [
            "node_disk_read_bytes_total", "node_disk_written_bytes_total",
            "node_disk_read_time_seconds_total", "node_disk_write_time_seconds_total",
            "node_network_receive_bytes_total", "node_network_transmit_bytes_total",
            "node_network_receive_packets_total", "node_network_transmit_packets_total",
            "pod_fs_reads_bytes_total", "pod_fs_writes_bytes_total",
            "pod_network_receive_bytes_total", "pod_network_transmit_bytes_total",
            "pod_network_receive_packets_total", "pod_network_transmit_packets_total"
        ]

        def convert_counters_to_rates(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            
            is_counter = df['kpi_key'].isin(COUNTER_METRICS)
            gauges_df = df[~is_counter].copy()
            counters_df = df[is_counter].copy()

            if counters_df.empty:
                return gauges_df

            counters_df.sort_values(['unified_component_id', 'kpi_key', 'time'], inplace=True)
            
            time_col = self._get_time_column("metrics")
            value_col = 'value'
            
            # Calculate difference in value and time
            counters_df['value_diff'] = counters_df.groupby(['unified_component_id', 'kpi_key'])[value_col].diff()
            counters_df['time_diff'] = counters_df.groupby(['unified_component_id', 'kpi_key'])[time_col].diff().dt.total_seconds()

            # Calculate rate, handle division by zero or NaN time_diff
            counters_df[value_col] = counters_df.apply(
                lambda row: row['value_diff'] / row['time_diff'] if row['time_diff'] > 0 else 0,
                axis=1
            )
            
            # Update KPI name to reflect it's a rate now
            counters_df['kpi_key'] = counters_df['kpi_key'].str.replace('_total', '_per_second')
            
            # Drop intermediate columns and rows with NaN rates (the first measurement for each group)
            counters_df.drop(columns=['value_diff', 'time_diff'], inplace=True)
            counters_df.dropna(subset=[value_col], inplace=True)

            return pd.concat([gauges_df, counters_df], ignore_index=True)

        baseline_df = convert_counters_to_rates(baseline_df)
        anomaly_df = convert_counters_to_rates(anomaly_df)
        group_keys = ['unified_component_id', 'kpi_key', 'inferred_component_type']

        if not baseline_df.empty:
            # V37: 计算更丰富的基线统计数据，包括p99和std
            baseline_metrics = baseline_df.groupby(group_keys)['value'].agg(['mean', 'std', lambda x: x.quantile(0.99)]).reset_index()
            baseline_metrics.rename(columns={'mean': 'baseline_value', 'std': 'baseline_std', '<lambda_0>': 'baseline_p99'}, inplace=True)
        else:
            baseline_metrics = pd.DataFrame(columns=group_keys + ['baseline_value', 'baseline_std', 'baseline_p99'])

        anomaly_metrics = anomaly_df.groupby(group_keys)['value'].agg(['mean', 'max', 'min', 'std']).reset_index()
        anomaly_metrics.rename(columns={
            'mean': 'anomaly_value',
            'max': 'anomaly_max',
            'min': 'anomaly_min',
            'std': 'anomaly_std'
        }, inplace=True)

        comparison_df = pd.merge(anomaly_metrics, baseline_metrics, on=group_keys, how='left')
        
        # V37: 对所有新列填充NA值
        comparison_df.fillna(0, inplace=True)


        comparison_df['change_percentage'] = comparison_df.apply(
            lambda row: 99999.0 if np.isclose(row['baseline_value'], 0) and row['anomaly_value'] > 0.001 else (
                ((row['anomaly_value'] - row['baseline_value']) / row['baseline_value'] * 100) if not np.isclose(row['baseline_value'], 0) else 0.0
            ),
            axis=1
        )
        
        if not anomaly_df.empty:
            component_info_cols = ['unified_component_id', 'ip_address', 'kubernetes_node']
            for col in component_info_cols:
                if col not in anomaly_df.columns:
                    anomaly_df[col] = None
            
            component_info = anomaly_df[component_info_cols].drop_duplicates('unified_component_id')
            comparison_df = pd.merge(comparison_df, component_info, on='unified_component_id', how='left')

        return comparison_df

    def _is_statistically_significant_increase(self, current_val: float, baseline_val: float, baseline_std: float) -> bool:
        # V21: 关键修复 - 避免报告从0到0的无意义增长
        if np.isclose(current_val, 0) and np.isclose(baseline_val, 0):
            return False

        # 如果标准差很小，说明基线非常稳定，此时使用相对变化率和绝对变化量来判断
        if baseline_std < 1e-6:
            # 对于从零开始的增长，只要当前值不为零即视为显著
            if np.isclose(baseline_val, 0):
                return not np.isclose(current_val, 0)
            
            # 对于非零基线，要求相对增幅超过50%且绝对增量也足够大，避免噪音
            relative_change = (current_val - baseline_val) / baseline_val
            absolute_change = current_val - baseline_val
            # 这里的0.01是一个例子，可能需要根据指标的量级调整
            return relative_change > 0.5 and absolute_change > 0.01 

        # 对于有显著波动的基线，使用经典的3-Sigma原则
        return current_val > baseline_val + self.SIGMA_THRESHOLD * baseline_std

    def _is_statistically_significant_decrease(self, current_val: float, baseline_val: float, baseline_std: float) -> bool:
        if baseline_std < 1e-6:
            if np.isclose(baseline_val, 0):
                return False
            if current_val < baseline_val:
                return (baseline_val - current_val) / baseline_val > 0.5
            return False

        return current_val < baseline_val - self.SIGMA_THRESHOLD * baseline_std

    # V20: 实现新的、基于类别的动态分析函数
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

        # V37: 定义容易出现尖峰的指标
        SPIKY_METRICS = [
            'node_cpu_usage_rate', 'pod_cpu_usage',
            'node_disk_read_bytes_per_second', 'node_disk_written_bytes_per_second',
            'node_disk_read_time_seconds_per_second', 'node_disk_write_time_seconds_per_second',
            'node_network_receive_bytes_per_second', 'node_network_transmit_bytes_per_second',
            'node_network_receive_packets_per_second', 'node_network_transmit_packets_per_second',
            'pod_fs_reads_bytes_per_second', 'pod_fs_writes_bytes_per_second',
            'pod_network_receive_bytes_per_second', 'pod_network_transmit_bytes_per_second',
            'pod_network_receive_packets_per_second', 'pod_network_transmit_packets_per_second'
        ]

        for _, row in category_df.iterrows():
            metric = row['kpi_key']
            current_val = row['anomaly_value']
            baseline_val = row['baseline_value']
            baseline_std = row['baseline_std']
            change_percentage = row['change_percentage']
            report_str = None
            severity = "minor"  # 默认严重等级为次要

            # V37 & V38: 对 spiky 指标使用带相对阈值的峰值检测逻辑
            if metric in SPIKY_METRICS:
                anomaly_max = row['anomaly_max']
                baseline_p99 = row['baseline_p99']
                dynamic_threshold = baseline_p99 + (2 * baseline_std)
                
                is_spike = False
                # 规则1: 峰值必须超过动态阈值
                if anomaly_max > dynamic_threshold:
                    # 规则2: 峰值必须超过一个合理的绝对值
                    # V43: 对于pod_cpu_usage，绝对值很小，所以放宽此限制
                    if anomaly_max > 0.01 or metric == 'pod_cpu_usage':
                        # 规则3: 如果基线不为零，则要求有显著的相对增幅 (例如 > 20%)
                        if baseline_p99 > 1e-6:
                            relative_increase = (anomaly_max - baseline_p99) / baseline_p99
                            if relative_increase > 0.20:
                                is_spike = True
                        else: # 基线为零，则满足前两个条件即为尖峰
                            is_spike = True

                if is_spike:
                    severity = "major"  # 尖峰至少是主要异常
                    
                    # V44: 对所有 usage_rate 指标应用动态阈值
                    if "usage_rate" in metric:
                        is_large_scale = anomaly_max > 1.1
                        critical_threshold = 90.0 if is_large_scale else 0.90
                        major_threshold = 80.0 if is_large_scale else 0.80
                        if anomaly_max > critical_threshold:
                            severity = "critical"
                        elif anomaly_max > major_threshold:
                            severity = "major"
                    # pod_cpu_usage 是核心数，不是比率，单独处理
                    elif metric == "pod_cpu_usage":
                        if anomaly_max > 1.0: # 超过1个核心是严重问题
                            severity = "critical"
                    
                    if "bytes_per_second" in metric:
                        report_str = f"「{metric}」出现异常尖峰，达到 {format_bytes(anomaly_max)}/s (正常上限: {format_bytes(dynamic_threshold)}/s)"
                    # V44: 对所有 usage_rate 指标应用动态格式化
                    elif "usage_rate" in metric:
                        is_large_scale = anomaly_max > 1.1 or dynamic_threshold > 1.1
                        if is_large_scale:
                            report_str = f"「{metric}」出现异常尖峰，达到 {anomaly_max:.1f}% (正常上限: {dynamic_threshold:.1f}%)"
                        else:
                            report_str = f"「{metric}」出现异常尖峰，达到 {anomaly_max*100:.1f}% (正常上限: {dynamic_threshold*100:.1f}%)"
                    elif metric == "pod_cpu_usage":
                        report_str = f"「{metric}」出现异常尖峰，达到 {anomaly_max:.2f} cores (正常上限: {dynamic_threshold:.2f} cores)"
                    elif "rate" in metric or "usage" in metric: # 其他rate, V43保留
                        report_str = f"「{metric}」出现异常尖峰，达到 {anomaly_max*100:.1f}% (正常上限: {dynamic_threshold*100:.1f}%)"
                    elif "time_seconds_per_second" in metric:
                        report_str = f"「{metric}」出现异常尖峰，达到 {anomaly_max*1000:.1f}ms/s (正常上限: {dynamic_threshold*1000:.1f}ms/s)"
                    else:
                        report_str = f"「{metric}」出现异常尖峰，达到 {anomaly_max:.2f} (正常上限: {dynamic_threshold:.2f})"
            else:
                # 对非 spiky 指标使用原有的均值比较逻辑
                is_increase = self._is_statistically_significant_increase(current_val, baseline_val, baseline_std)
                is_decrease = self._is_statistically_significant_decrease(current_val, baseline_val, baseline_std)

                if is_increase:
                    # V44: 移除不可能指标检查，代之以动态阈值
                    if "usage_rate" in metric:
                        is_large_scale = current_val > 1.1
                        critical_threshold = 90.0 if is_large_scale else 0.90
                        major_threshold = 80.0 if is_large_scale else 0.80
                        if current_val > critical_threshold:
                            severity = "critical"
                        elif current_val > major_threshold:
                            severity = "major"
                    elif change_percentage > 200: # 超过200%的增长视为主要问题
                        severity = "major"

                    if "bytes_per_second" in metric:
                        report_str = f"「{metric}」从 {format_bytes(baseline_val)}/s 增长至 {format_bytes(current_val)}/s"
                    elif "bytes" in metric: # gauges
                        report_str = f"「{metric}」从 {format_bytes(baseline_val)} 增长至 {format_bytes(current_val)}"
                    # V44: 对所有 usage_rate 指标应用动态格式化
                    elif "usage_rate" in metric:
                        is_large_scale = current_val > 1.1 or baseline_val > 1.1
                        if is_large_scale:
                            report_str = f"「{metric}」从 {baseline_val:.1f}% 增长至 {current_val:.1f}%"
                        else:
                            report_str = f"「{metric}」从 {baseline_val*100:.1f}% 增长至 {current_val*100:.1f}%"
                    elif "rate" in metric or "usage" in metric: # 其他rate
                         report_str = f"「{metric}」从 {baseline_val*100:.1f}% 增长至 {current_val*100:.1f}%"
                    elif "time_seconds_per_second" in metric:
                         report_str = f"「{metric}」从 {baseline_val*1000:.1f}ms/s 增长至 {current_val*1000:.1f}ms/s"
                    elif "time_seconds" in metric: # gauges
                         report_str = f"「{metric}」从 {baseline_val*1000:.1f}ms 增长至 {current_val*1000:.1f}ms"
                elif is_decrease:
                    severity = "major" # 显著下降通常是主要问题
                    if "bytes_per_second" in metric:
                        report_str = f"「{metric}」从 {format_bytes(baseline_val)}/s 下降至 {format_bytes(current_val)}/s"
                    elif "bytes" in metric: # gauges
                         report_str = f"「{metric}」从 {format_bytes(baseline_val)} 下降至 {format_bytes(current_val)}"
                    else:
                         report_str = f"「{metric}」从 {baseline_val:.1f} 下降至 {current_val:.1f}"

            if report_str:
                anomalies.append({
                    "description": report_str,
                    "severity": severity,
                    "category": category_name
                })

        return anomalies
    # V20: 全面重构报告生成逻辑
    def _create_comparison_report(self, comparison_df: pd.DataFrame, anomaly_df: pd.DataFrame, missing_components: List[str]) -> Dict[str, Any]:
        reports = []
        
        for component_id, group in comparison_df.groupby('unified_component_id'):
            component_anomalies = []
            component_type = group['inferred_component_type'].iloc[0]
            related_node = group['kubernetes_node'].iloc[0] if 'kubernetes_node' in group.columns else "N/A"

            # 遍历该组件类型的所有专家类别
            categories_to_check = METRIC_CATEGORIES.get(component_type, {})
            for category_name, metric_keys in categories_to_check.items():
                category_df = group[group['kpi_key'].isin(metric_keys)]
                if not category_df.empty:
                    category_anomalies = self._analyze_metric_category(category_name, category_df)
                    if category_anomalies:
                        component_anomalies.extend(category_anomalies)

            if component_anomalies:
                # 按严重程度排序
                severity_order = {"critical": 0, "major": 1, "minor": 2}
                component_anomalies.sort(key=lambda x: severity_order.get(x["severity"], 99))
                
                reports.append({
                    "component": component_id,
                    "type": component_type,
                    "related_node": related_node,
                    "anomalies": component_anomalies
                })

        summary_parts = []
        if missing_components:
            summary_parts.append(f"**严重警告**: {len(missing_components)}个组件的数据缺失({', '.join(missing_components)})，可能已离线，这本身就是一个关键的故障信号。")

        # V40: 基于严重等级的智能摘要
        if reports:
            # 按组件的最高严重等级排序
            severity_order = {"critical": 0, "major": 1, "minor": 2}
            reports.sort(key=lambda r: severity_order.get(r['anomalies'][0]['severity'], 99))

            critical_reports = [r for r in reports if r['anomalies'][0]['severity'] == 'critical']
            major_reports = [r for r in reports if r['anomalies'][0]['severity'] == 'major']
            
            if critical_reports:
                # 报告最严重的组件及其最严重的问题
                top_critical = critical_reports[0]
                summary_parts.append(
                    f"**检测到严重异常**: 组件「{top_critical['component']}」({top_critical['type']}) "
                    f"出现严重问题: {top_critical['anomalies'][0]['description']}。"
                )
            elif major_reports:
                top_major = major_reports[0]
                summary_parts.append(
                    f"**检测到主要异常**: 组件「{top_major['component']}」({top_major['type']}) "
                    f"出现显著性能问题: {top_major['anomalies'][0]['description']}。"
                )

            total_anomalous_components = len(reports)
            if total_anomalous_components > 1:
                summary_parts.append(f"总共在 {total_anomalous_components} 个组件上检测到不同程度的指标异常。")
        
        if not summary_parts:
            summary = "无显著的基础设施指标异常。"
        else:
            summary = " ".join(summary_parts)

        # 整合高级诊断：吵闹的邻居分析在所有报告生成后进行
        all_pod_reports = [r for r in reports if r.get("type") == "pod"]
        if all_pod_reports:
            noisy_neighbor_hypothesis = self._analyze_for_noisy_neighbor(all_pod_reports, comparison_df)
            if noisy_neighbor_hypothesis:
                summary += f" {noisy_neighbor_hypothesis}"
        
        return {
            "summary": summary,
            "anomalous_components": reports,
        }

    # V20: 调整“吵闹的邻居”分析以适应新的报告格式
    def _analyze_for_noisy_neighbor(self, reports: List[Dict], all_components_df: pd.DataFrame) -> Optional[str]:
        """
        分析是否存在“吵闹的邻居”问题，即节点级故障。
        """
        # 1. 找出所有报告了异常的节点
        nodes_with_anomalies = defaultdict(list)
        for report in reports:
            # 我们只关心有底层资源问题的pod
            if report.get("type") == "pod" and report.get("related_node") != "N/A":
                # V41: 适配新的anomalies结构，检查是否存在cpu或memory类别的异常
                has_resource_issue = any(
                    anomaly['category'] in ['cpu', 'memory']
                    for anomaly in report.get('anomalies', [])
                )
                if has_resource_issue:
                    nodes_with_anomalies[report["related_node"]].append(report["component"])

        # 2. 检查这些节点本身是否饱和
        for node, pods in nodes_with_anomalies.items():
            # 如果一个节点上有超过1个pod出现资源问题
            if len(set(pods)) > 1:
                # 检查该节点自身的饱和度
                node_metrics = all_components_df[
                    (all_components_df['unified_component_id'] == node) &
                    (all_components_df['kpi_key'].isin(['node_cpu_usage_rate', 'node_memory_usage_rate']))
                ]
                if not node_metrics.empty:
                    # V42: 将饱和度阈值调整回更保守的90%
                    is_saturated = (node_metrics['anomaly_max'] > 0.90).any()
                    if is_saturated:
                        # 构建一个更有信息的pod列表，包含服务名
                        pod_service_map = {
                            row['unified_component_id']: row['unified_component_id'].rsplit('-', 1)[0]
                            for _, row in all_components_df[all_components_df['inferred_component_type'] == 'pod'].iterrows()
                        }
                        # V41: 修正pod列表获取逻辑，避免重复
                        unique_pods = sorted(list(set(pods)))
                        pod_details = [f"{p}({pod_service_map.get(p, 'unknown_service')})" for p in unique_pods]

                        return (
                            f"**Hypothesis: Node-Level Issue Detected on '{node}'.** "
                            f"该节点上的多个Pod ({', '.join(pod_details)}) 同时出现资源竞争，"
                            f"并且节点本身已达到饱和状态。根因很可能在节点层面，而不是单个Pod。"
                        )
        return None
        
    # V20: _check_io_saturation 现在主要作为辅助函数，由主流程调用
    def _check_io_saturation(self, node_reports: List[Dict], all_components_df: pd.DataFrame) -> Optional[str]:
        """
        Analyzes node components for I/O saturation, which is a strong indicator of a bottleneck.
        """
        IO_TIME_METRICS = ["node_disk_write_time_seconds_total", "node_disk_read_time_seconds_total"]
        # A threshold close to 1.0 means the I/O is busy for almost the entire second.
        SATURATION_THRESHOLD = 0.8 

        saturated_nodes = []

        for report in node_reports:
            node_name = report["name"]
            node_df = all_components_df[all_components_df['unified_component_id'] == node_name]
            
            for metric in IO_TIME_METRICS:
                metric_df = node_df[node_df['kpi_key'] == metric]
                if not metric_df.empty:
                    max_io_time = metric_df['anomaly_value'].max()
                    if max_io_time >= SATURATION_THRESHOLD:
                        saturated_nodes.append(f"{node_name} (metric: {metric}, peak: {max_io_time:.2f}s/s)")
                        break # Move to the next node once one saturated metric is found

        if saturated_nodes:
            return (
                f"**Hypothesis: Node IO Saturation Detected on {', '.join(saturated_nodes)}.** "
                f"This indicates a severe disk I/O bottleneck, which is a strong candidate for the root cause."
            )
        
        return None

    def _create_node_maps(self, df: pd.DataFrame):
        if df.empty or 'instance' not in df.columns or 'kubernetes_node' not in df.columns:
            return

        node_info_df = df[['instance', 'kubernetes_node']].dropna()
        node_info_df = node_info_df[node_info_df['kubernetes_node'] != 'null']
        node_info_df = node_info_df[node_info_df['instance'] != 'null']
        
        self._instance_to_node_map = node_info_df.drop_duplicates(
            subset=['instance']
        ).set_index('instance')['kubernetes_node'].to_dict()
        
        self._node_to_instance_map = node_info_df.drop_duplicates(
            subset=['kubernetes_node']
        ).set_index('kubernetes_node')['instance'].to_dict()

    def _find_infra_metric_files(
        self, start_dt: datetime, end_dt: datetime
    ) -> List[str]:
        
        base_patterns = [
            "infra/infra_node/*.parquet",
            "infra/infra_pod/*.parquet",
        ]
        
        all_files = set()
        cst_tz = timedelta(hours=8)
        
        start_dt_cst = start_dt + cst_tz
        end_dt_cst = end_dt + cst_tz
        
        dates_to_check = pd.date_range(start_dt_cst.date(), end_dt_cst.date())
        
        for date in dates_to_check:
            cst_date_str = date.strftime("%Y-%m-%d")
            for pattern in base_patterns:
                all_files.update(
                    self._find_files_by_time_range(
                        start_dt, end_dt, "metric-parquet", pattern, force_date=cst_date_str
                    )
                )

        return list(all_files)

    def _get_relevant_columns(
        self, data_type: str, available_columns: List[str]
    ) -> List[str]:
        return list(available_columns)

    def _get_time_column(self, data_type: str) -> Optional[str]:
        return "time"

    def _get_component_columns(self, data_type: str) -> List[str]:
        return [
            "unified_component_id",
            "object_type",
            "pod",
            "kubernetes_node",
            "instance",
        ]

    def _determine_component_column(self, df: pd.DataFrame) -> Optional[str]:
        if "unified_component_id" in df.columns:
            return "unified_component_id"
        if "pod" in df.columns and df["pod"].notna().any():
            return "pod"
        if "instance" in df.columns and df["instance"].notna().any():
            return "instance"
        if "kubernetes_node" in df.columns and df["kubernetes_node"].notna().any():
            return "kubernetes_node"
        if "object_id" in df.columns and df["object_id"].notna().any():
            return "object_id"
        if "node" in df.columns and df["node"].notna().any():
            return "node"
        return None

    def _determine_component_type(self, component_id: str, object_type_val: str) -> str:
        component_id = str(component_id).lower()
        object_type_val = str(object_type_val).lower()
        
        if object_type_val in ["pod", "node"]:
            return object_type_val

        if any(node in component_id for node in ["aiops-k8s", "k8s-master"]):
            return "node"
        
        return "pod" 