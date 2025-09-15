import glob
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from collections import defaultdict

from app.prompt.v1_rca_context import TOPOLOGY
from app.tool.base import ToolResult
from app.tool.rca.base_loader import BaseLoader


class APMMetricsLoader(BaseLoader):
    """加载应用性能监控(APM)指标数据"""

    name: str = "apm_loader"
    description: str = "加载APM指标数据，分析业务层面的异常情况，识别异常服务和Pod"

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
        },
        "required": ["uuid", "start_time", "end_time"],
    }

    # 定义要从分析中排除的组件名称列表
    EXCLUDED_COMPONENTS: List[str] = ["hipstershop"]

    # --- V8 新增：定义用户入口服务 ---
    USER_FACING_SERVICES: List[str] = ["frontend"]

    METRICS_TO_ANALYZE: List[str] = [
        'error_ratio', 'rrt', 'server_error_ratio', 'request',
        'client_error_ratio', 'timeout', 'rrt_max',
        'error', 'client_error', 'server_error'
    ]

    async def execute(
        self,
        uuid: str,
        start_time: str,
        end_time: str,
        **kwargs: Any,
    ) -> Any:
        try:
            # 1. 解析时间范围
            start_dt, end_dt = self._parse_time_range(start_time, end_time)

            # --- 最佳实践 (V3): 带有隔离带的窗口设置 ---
            # 1. 扩展异常窗口以保证数据完整性
            MIN_ANOMALY_DURATION_SECONDS = 120
            if (end_dt - start_dt).total_seconds() < MIN_ANOMALY_DURATION_SECONDS:
                start_dt = start_dt - timedelta(seconds=120)
                end_dt = start_dt + timedelta(seconds=MIN_ANOMALY_DURATION_SECONDS)

            # 2. 定义包含隔离带的基线窗口
            QUARANTINE_MINUTES = 5
            BASELINE_WINDOW_MINUTES = 10 # 基线窗口本身持续10分钟
            baseline_end_dt = start_dt - timedelta(minutes=QUARANTINE_MINUTES)
            baseline_start_dt = baseline_end_dt - timedelta(minutes=BASELINE_WINDOW_MINUTES)
            
            # 3. 定义完整的数据加载窗口
            full_load_start_dt = baseline_start_dt
            full_load_end_dt = end_dt

            # 2. 查找并加载故障窗口和基线窗口所需的所有文件
            all_files = self._find_apm_metric_files(full_load_start_dt, full_load_end_dt)
            if not all_files:
                return ToolResult(output=json.dumps({
                    "summary": "未找到指定时间范围内的APM指标数据文件。",
                    "components_with_changes": []
                }))

            # 使用谓词下推进行精准加载
            df = self._load_and_process_apm_data(all_files, full_load_start_dt, full_load_end_dt)
            if df.empty:
                 return ToolResult(output=json.dumps({
                    "summary": "加载的APM指标数据为空。",
                    "components_with_changes": []
                }))
            
            # --- 新增：过滤掉已知的非组件系统名 ---
            df = df[~df['object_id'].isin(self.EXCLUDED_COMPONENTS)]

            # --- 存在性分析 ---
            # 由于工具现在总是全局的，我们只关心数据是否完全缺失

            # 4. 分割基线和异常数据 (基于已加载的精准数据)
            time_col = self._get_time_column("metrics")
            baseline_df = df[(df[time_col] >= baseline_start_dt) & (df[time_col] < baseline_end_dt)]
            anomaly_df = df[(df[time_col] >= start_dt) & (df[time_col] <= end_dt)]
            
            if anomaly_df.empty:
                return ToolResult(output=json.dumps({
                    "summary": "在指定的故障时间窗口内没有找到APM指标数据。",
                    "components_with_changes": []
                }))

            # 5. 执行基线对比分析
            analysis_result = self._compare_with_baseline(baseline_df, anomaly_df)
            
            # 6. 生成最终报告
            report = self._create_report(analysis_result, anomaly_df)
            report = self.convert_numpy_types(report)

            return ToolResult(output=json.dumps(report, indent=2, ensure_ascii=False))

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return ToolResult(error=f"APM指标加载及分析错误: {str(e)}\n调用栈: {error_trace}")

    def _compare_with_baseline(self, baseline_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> pd.DataFrame:
        """
        V36: 重构以使用标准化的长表格式数据。
        按组件和指标类型进行分组，以正确计算基线。
        """
        if baseline_df.empty:
            # 返回一个空的、但结构正确的DataFrame，以避免下游错误
            return pd.DataFrame(columns=['object_id', 'object_type', 'metric', 'anomaly_value', 'baseline_value'])

        group_keys = ['object_id', 'object_type', 'metric']

        # V37: 计算更丰富的基线统计数据，包括p99和std
        baseline_metrics = baseline_df.groupby(group_keys)['value'].agg(['mean', 'std', lambda x: x.quantile(0.99)]).reset_index()
        baseline_metrics.rename(columns={'mean': 'baseline_value', 'std': 'baseline_std', '<lambda_0>': 'baseline_p99'}, inplace=True)
        baseline_metrics.fillna(0, inplace=True)

        # V37: 计算异常窗口的峰值
        # V-Fix (20250730): 增加 'sum' 聚合，用于正确报告事件总数，避免均值被截断。
        anomaly_metrics = anomaly_df.groupby(group_keys)['value'].agg(['mean', 'max', 'sum']).reset_index()
        anomaly_metrics.rename(columns={'mean': 'anomaly_value', 'max': 'anomaly_max', 'sum': 'anomaly_sum'}, inplace=True)

        comparison_df = pd.merge(anomaly_metrics, baseline_metrics, on=group_keys, how='left')
        comparison_df.fillna(0, inplace=True)
        return comparison_df

    def _create_report(self, comparison_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> Dict[str, Any]:
        
        reports = []
        all_anomalies = []

        for _, row in comparison_df.iterrows():
            component_name = row['object_id']
            metric = row['metric'] # V36: 从正确的 'metric' 列获取指标名称
            current_val = row['anomaly_value']
            baseline_val = row['baseline_value']
            
            report_str = None
            is_significant = False
            anomaly_details = None
            severity = "minor" # V42: 默认严重等级

            if metric in ['error', 'timeout']:
                # V-Fix (20250730): 使用 anomaly_sum 保证事件总数准确
                total_events = row['anomaly_sum']
                if total_events > baseline_val: # 只要异常期间事件数高于基线均值就认为显著
                    is_significant = True
                    
                    # V42: 引入严重等级判断
                    if total_events > 100:
                        severity = "critical"
                    elif total_events > 10:
                        severity = "major"
                    
                    # 查找首次出现时间
                    first_occurrence = "N/A"
                    first_occurrence_row = anomaly_df[
                        (anomaly_df['object_id'] == component_name) & 
                        (anomaly_df['metric'] == metric) &
                        (anomaly_df['value'] > 0)
                    ]
                    if not first_occurrence_row.empty:
                        first_occurrence = first_occurrence_row['time'].min().strftime('%H:%M:%SZ')

                    report_str = f"出现「{metric}」: {int(total_events)} 次 (基线: {baseline_val:.1f} 次) (首次于 {first_occurrence})"
                    anomaly_details = {"metric": metric, "value": int(total_events), "baseline": int(baseline_val), "first_occurrence": first_occurrence}


            elif metric == 'rrt':
                # V37: 使用新的尖峰检测逻辑
                # V38: 增加相对变化阈值以提高准确性
                baseline_p99 = row['baseline_p99']
                baseline_std = row['baseline_std']
                anomaly_max = row['anomaly_max']

                # 定义动态阈值: 基线P99 + 2个标准差
                dynamic_threshold = baseline_p99 + (2 * baseline_std)
                
                is_spike = False
                # 规则1: 峰值必须超过动态阈值
                if anomaly_max > dynamic_threshold:
                    # 规则2: 峰值必须超过一个合理的绝对值（例如100ms）
                    if anomaly_max * 1000 > 100:
                        # 规则3: 如果基线不为零，则要求有显著的相对增幅 (例如 > 20%)
                        if baseline_p99 > 1e-6: # 避免除以零
                            relative_increase = (anomaly_max - baseline_p99) / baseline_p99
                            if relative_increase > 0.2:
                                is_spike = True
                        else: # 基线为零，则满足前两个条件即为尖峰
                            is_spike = True
                
                if is_spike:
                    is_significant = True
                    
                    # V42: 引入严重等级判断
                    if anomaly_max > 10: # 超过10秒的RRT是严重问题
                        severity = "critical"
                    elif anomaly_max > 2 * dynamic_threshold and anomaly_max > 1: # 超过动态阈值2倍且绝对值大于1秒
                        severity = "major"

                    # 查找rrt尖峰首次出现时间
                    first_occurrence_rrt = "N/A"
                    # 在异常窗口内查找首次超过动态阈值的点
                    first_occurrence_rrt_row = anomaly_df[
                        (anomaly_df['object_id'] == component_name) & 
                        (anomaly_df['metric'] == 'rrt') &
                        (anomaly_df['value'] > dynamic_threshold)
                    ]
                    if not first_occurrence_rrt_row.empty:
                        first_occurrence_rrt = first_occurrence_rrt_row['time'].min().strftime('%H:%M:%SZ')

                    report_str = f"「rrt」出现异常尖峰，达到 {anomaly_max*1000:.0f}ms (正常波动上限: {dynamic_threshold*1000:.0f}ms, 首次于 {first_occurrence_rrt})"
                    anomaly_details = {
                        "metric": "rrt_spike", 
                        "value": f"{anomaly_max*1000:.0f}ms", 
                        "baseline_upper_bound": f"{dynamic_threshold*1000:.0f}ms",
                        "first_occurrence": first_occurrence_rrt
                    }
            
            if is_significant:
                all_anomalies.append({
                    "name": component_name,
                    "description": report_str,
                    "severity": severity,
                    "details": anomaly_details
                })

        if not all_anomalies:
            return {"summary": "全局APM扫描完成，未发现显著的性能异常。", "components_with_changes": []}

        # 按组件聚合报告
        component_reports = defaultdict(list)
        for anomaly in all_anomalies:
            component_reports[anomaly['name']].append({
                "description": anomaly['description'],
                "severity": anomaly['severity'],
                "details": anomaly['details']
            })

        # V39 & V42: 重构报告生成和摘要逻辑
        for name, anomalies in component_reports.items():
            component_type = "service"
            is_pod = '-' in name and name.rsplit('-', 1)[-1].isdigit()
            if is_pod:
                component_type = "pod"
            
            # 按严重性排序
            severity_order = {"critical": 0, "major": 1, "minor": 2}
            anomalies.sort(key=lambda x: severity_order.get(x["severity"], 99))

            reports.append({
                "name": name,
                "type": component_type,
                "anomalies": anomalies
            })
        
        # V39 & V42: 移除基于分数的排序和总结，改为基于洞察和严重等级的总结
        if reports:
            # 按组件最高严重等级排序
            severity_order = {"critical": 0, "major": 1, "minor": 2}
            reports.sort(key=lambda r: severity_order.get(r['anomalies'][0]['severity'], 99))

            summary_parts = [f"全局APM扫描完成。在系统中检测到 {len(reports)} 个组件存在性能异常。"]
            
            # 高亮最严重的问题
            top_report = reports[0]
            top_anomaly = top_report['anomalies'][0]
            summary_prefix = f"**检测到{top_anomaly['severity'].capitalize()}异常**: " if top_anomaly['severity'] in ["critical", "major"] else ""
            summary_parts.append(
                f"{summary_prefix}组件「{top_report['name']}」({top_report['type']}) "
                f"出现显著问题: {top_anomaly['description']}。"
            )

            # 识别广泛的服务故障
            service_to_pods = defaultdict(set)
            all_pods_in_report = {r['name'] for r in reports if r['type'] == 'pod'}
            for pod_name in all_pods_in_report:
                service_name = pod_name.rsplit('-', 1)[0]
                service_to_pods[service_name].add(pod_name)

            widespread_service_failures = []
            for service, pods in service_to_pods.items():
                if len(pods) > 1: # 超过一个实例出现问题
                     widespread_service_failures.append(f"`{service}` (影响了 {len(pods)} 个实例)")
            
            if widespread_service_failures:
                summary_parts.append(
                    "**关键洞察**: 服务 " + ", ".join(widespread_service_failures) + 
                    " 的多个实例均出现异常，表明可能存在服务级别的系统性问题。"
                )
            
            summary = " ".join(summary_parts)
        else:
            summary = "全局APM扫描完成，未发现显著的性能异常。"


        return {"summary": summary, "components_with_changes": reports}

    def _load_and_process_apm_data(self, files: List[str], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        V36: 重构此方法以处理宽表到长表的转换。
        使用 pandas.melt 将不同的指标列（rrt, error, timeout等）统一到 'metric' 和 'value' 列中。
        """
        df = self._load_data(files, "apm", start_dt, end_dt)
        if df.empty:
            return pd.DataFrame()

        # 确定ID列和要转换的指标列
        id_vars = ['time', 'object_id', 'object_type']
        
        # 找出所有可以被当作指标的列
        # README中定义的APM指标 + 任何其他可能的数字列
        known_metrics = [
            'request', 'response', 'rrt', 'rrt_max', 'error', 
            'client_error', 'server_error', 'timeout', 'error_ratio',
            'client_error_ratio', 'server_error_ratio'
        ]
        
        # 仅保留数据帧中实际存在的指标列
        value_vars = [col for col in known_metrics if col in df.columns]

        # 如果没有有效的指标列，则返回空DataFrame
        if not value_vars:
            self.logger.warning("在APM数据中未找到任何已知的指标列。")
            return pd.DataFrame()

        # 确保ID列存在
        for col in id_vars:
            if col not in df.columns:
                self.logger.error(f"APM数据中缺失关键ID列: {col}")
                return pd.DataFrame()
        
        # 使用melt函数进行转换
        melted_df = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='metric',
            value_name='value'
        )

        # 转换value列为数字，处理无效值
        melted_df['value'] = pd.to_numeric(melted_df['value'], errors='coerce').fillna(0)
        
        return melted_df

    def _find_apm_metric_files(
        self, start_dt: datetime, end_dt: datetime
    ) -> List[str]:
        """
        使用通用的、时区正确的文件查找方法来定位APM指标文件。
        此版本经过重构，可以正确处理跨时区的日期问题。
        """
        base_patterns = [
            "apm/pod/pod_*.parquet",
            "apm/service/service_*.parquet",
            "apm/pod_ns_hipstershop_*.parquet"
        ]

        # --- 核心重构：基于CST时间计算日期范围 ---
        all_files = set()
        cst_tz = timedelta(hours=8)
        
        start_dt_cst = start_dt + cst_tz
        end_dt_cst = end_dt + cst_tz
        
        # 获取查询时间范围在CST时区所覆盖的所有日期
        dates_to_check = pd.date_range(start_dt_cst.date(), end_dt_cst.date())
        
        for date in dates_to_check:
            cst_date_str = date.strftime("%Y-%m-%d")
            for pattern in base_patterns:
                # 恢复正确的 data_type 参数，确保在 'metric-parquet' 目录下查找
                all_files.update(
                    self._find_files_by_time_range(
                        start_dt, end_dt, "metric-parquet", pattern, force_date=cst_date_str
                    )
                )

        return list(all_files)

    def _get_relevant_columns(
        self, data_type: str, available_columns: List[str]
    ) -> List[str]:
        """确定需要加载的APM指标列"""
        # APM指标中的关键列
        apm_columns = [
            "time",  # 时间 (UTC时区)
            "error",  # 错误数
            "error_ratio",  # 错误率
            "object_id",  # 对象ID
            "object_type",  # 对象类型
            "server_error",  # 服务器错误
            "server_error_ratio",  # 服务器错误率
            "client_error",  # 客户端错误
            "client_error_ratio",  # 客户端错误率
            "request",  # 请求数
            "response",  # 响应数
            "rrt",  # 响应时间
            "rrt_max",  # 最大响应时间
            "timeout",  # 超时数
        ]
        return [col for col in apm_columns if col in available_columns]

    def _get_time_column(self, data_type: str) -> Optional[str]:
        """获取APM指标时间列名"""
        return "time"

    def _get_component_columns(self, data_type: str) -> List[str]:
        """获取APM指标组件列名"""
        return ["object_id", "object_type"]

    def _detect_anomalies(self, df: pd.DataFrame, baseline_df: pd.DataFrame) -> Dict[str, Any]:
        """
        使用基线数据和固定阈值检测APM指标中的异常。
        增强了变化检测和异常聚类。
        """
        anomalies = {"services": {}, "pods": {}}
        if df.empty:
            return anomalies

        # 增加一步过滤，只处理 object_type 为 'service' 或 'pod' 的数据
        # 这样可以确保像 'hipstershop' 这样的命名空间级别指标不会被错误地归类为 pod。
        df = df[df['object_type'].isin(['service', 'pod'])].copy()

        # 1. 固定阈值检测 (针对错误和超时)
        static_thresholds = {
            "error_ratio": 0.01,  # 容忍1%以下的瞬时错误
            "server_error_ratio": 0.01,
            "timeout": 1,  # 任何超时都值得关注
        }

        # 2. 动态阈值针对 rrt
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors='coerce', utc=True)
        if "time" in baseline_df.columns:
            baseline_df["time"] = pd.to_datetime(baseline_df["time"], errors='coerce', utc=True)
        
        # 按组件和服务分组
        for (obj_type, obj_id), group_df in df.groupby(["object_type", "object_id"]):
            anomaly_metrics = {}
            
            # --- 固定阈值检测 ---
            for metric, threshold in static_thresholds.items():
                if metric not in group_df.columns: continue
                
                anomaly_points = group_df[group_df[metric] > threshold]
                if not anomaly_points.empty:
                    # 异常聚类：如果连续异常，视为一个事件
                    first_occurrence = anomaly_points['time'].min()
                    max_value = anomaly_points[metric].max()
                    
                    anomaly_metrics[metric] = {
                        "count": len(anomaly_points),
                        "max_value": float(max_value),
                        "threshold": float(threshold),
                        "threshold_type": "fixed",
                        "examples": [{"time": str(first_occurrence), "value": float(max_value)}]
                    }

            # --- 动态阈值检测 (rrt) ---
            if not baseline_df.empty and 'rrt' in baseline_df.columns and 'rrt' in group_df.columns:
                # 找到该组件对应的基线数据
                comp_baseline_df = baseline_df[baseline_df['object_id'] == obj_id]
                if not comp_baseline_df.empty:
                    # 计算P95和标准差作为动态阈值
                    p95 = comp_baseline_df['rrt'].quantile(0.95)
                    std_dev = comp_baseline_df['rrt'].std()
                    dynamic_threshold = p95 + (2 * std_dev)  # P95 + 2个标准差
                    
                    anomaly_points = group_df[group_df['rrt'] > dynamic_threshold]
                    if not anomaly_points.empty:
                        first_occurrence = anomaly_points['time'].min()
                        max_value = anomaly_points['rrt'].max()
                        
                        anomaly_metrics['rrt'] = {
                            "count": len(anomaly_points),
                            "max_value": float(max_value),
                            "threshold": float(dynamic_threshold),
                            "threshold_type": "dynamic_baseline(P95+2sigma)",
                            "examples": [{"time": str(first_occurrence), "value": float(max_value)}]
                        }
                        
                        # 如果是服务，找到异常时刻对应的pod
                        if obj_type == 'service':
                            service_name = obj_id
                            # 找到所有属于该服务的pod
                            # 假设pod命名规范为 a-0, a-1
                            pod_prefix = service_name
                            
                            related_pods_df = df[
                                df['object_type'].eq('pod') & 
                                df['object_id'].str.startswith(pod_prefix) &
                                df['time'].isin(anomaly_points['time'])
                            ]
                            
                            if not related_pods_df.empty:
                                pod_summary = related_pods_df.groupby('object_id')['rrt'].max().sort_values(ascending=False)
                                anomaly_metrics['rrt']['related_pods'] = [
                                    {"name": pod, "value": val} for pod, val in pod_summary.items()
                                ]


            if anomaly_metrics:
                target_dict = anomalies["services"] if obj_type == "service" else anomalies["pods"]
                target_dict[obj_id] = anomaly_metrics
        
        return anomalies
