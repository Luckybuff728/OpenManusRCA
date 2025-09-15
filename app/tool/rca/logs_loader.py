import glob
import json
import os
import re
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, ClassVar

import pandas as pd

from app.tool.base import ToolResult
from app.tool.rca.base_loader import BaseLoader


class LogsLoader(BaseLoader):
    """加载并分析容器日志数据，以支持根因定位。"""

    name: str = "logs_loader"
    description: str = "加载并分析指定组件的容器日志，自动提取异常模式。"
    
    # V5: Introduce Severity Levels for nuanced analysis
    LOG_SEVERITY_PATTERNS: ClassVar[Dict[str, List[str]]] = {
        "system_critical": [
            "oomkilled", "out of memory",
            "no space left on device",
            "read-only file system",
            "crashloopbackoff",  # k8s state for repeated crashes
        ],
        "application_fatal": [
            "panic",
            "fatal",
            "agent failed to start",  # Fatal for the app, but might be a symptom
        ],
        "business_impact_error": [
            "failed to charge card",
            "could not retrieve products",
            "failed to get cart",
            "failed to get recommendations",
            "failed to get ads",
            "payment failed",
            "shipping failed",
        ],
        "connectivity_error": [
            "unable to connect to",
            "connection refused",
            "unavailable",  # Often network related
            "grpc",  # gRPC errors often hint at connectivity issues
            "timeout",
            "pd server timeout",
            "region is unavailable",
            "failed to connect to all addresses",
        ],
        "application_warning": [
            "exception",  # General exceptions
            "liveness probe failed",
            "readiness probe failed",
            "error",  # General error keyword
        ],
    }

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
                "description": "可选，指定要加载日志的特定组件名称列表（可以是 pod 名称或 node 名称）",
            },
            "error_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选，要搜索的错误关键字列表 (注意: 不支持error_level参数，请使用error_keywords)",
            },
        },
        "required": ["uuid", "start_time", "end_time"],
    }

    async def execute(
        self,
        uuid: str,
        start_time: str,
        end_time: str,
        components: Optional[List[str]] = None,
        error_keywords: Optional[List[str]] = None,
        **kwargs,  # 添加kwargs以捕获所有额外参数
    ) -> Any:
        """执行日志数据加载操作

        Args:
            uuid: 故障案例唯一标识
            start_time: 开始时间（UTC）
            end_time: 结束时间（UTC）
            components: 可选的资源名称列表 (pod或node)
            error_keywords: 可选的错误关键字列表 (注意: 不支持error_level参数)
            **kwargs: 捕获额外的参数，用于检查不支持的参数

        Returns:
            加载并处理后的日志数据摘要
        """
        try:
            # 检查是否有不支持的参数
            if "error_level" in kwargs:
                return ToolResult(
                    error=f"LogsLoader不支持'error_level'参数，请使用'error_keywords'代替"
                )
            
            # --- 新增：向后兼容和参数统一 ---
            if "resource_names" in kwargs and kwargs["resource_names"]:
                if not components:
                    components = kwargs["resource_names"]
                    self.logger.warning(
                        "参数警告: 'resource_names' 已被弃用，请改用 'components'。"
                        f"已将 resource_names 的值 '{components}' 用于 components 继续执行。"
                    )


            # 解析时间范围
            start_dt, end_dt = self._parse_time_range(start_time, end_time)

            # --- 最佳实践 (V3): 带有隔离带的窗口设置 ---
            # 1. 扩展异常窗口以保证数据完整性
            MIN_ANOMALY_DURATION_SECONDS = 120
            if (end_dt - start_dt).total_seconds() < MIN_ANOMALY_DURATION_SECONDS:
                end_dt = start_dt + timedelta(seconds=MIN_ANOMALY_DURATION_SECONDS)

            # 2. 为日志加载定义一个包含故障前上下文的窗口
            CONTEXT_MINUTES = 5
            load_start_dt = start_dt - timedelta(minutes=CONTEXT_MINUTES)
            load_end_dt = end_dt

            # --- 修复：无论是否有components，都先查找所有相关的日志文件 ---
            all_log_files = self._find_log_files(load_start_dt, load_end_dt)
            
            # 统一检查是否找到了日志文件，如果没有则提前返回
            if not all_log_files:
                summary_msg = "在指定时间范围内未找到任何日志文件。"
                if components:
                    summary_msg = f"在指定时间范围内未找到资源 {components} 的任何日志文件。"
                
                return ToolResult(output=json.dumps({
                    "summary": summary_msg,
                    "anomalous_components": [],
                }))

            if not error_keywords:
                error_keywords = [
                    "error", "exception", "failed", "timeout", "refused",
                    "unavailable", "critical", "panic", "fatal", "traceback",
                    "deadlock", "corrupt", "denied"
                ]

            # V8 核心重构: 先加载所有相关日志，再进行分析
            # 1. 加载所有相关组件的全部日志
            all_component_logs_df = self._load_all_logs_for_components(all_log_files, components, load_start_dt, load_end_dt)

            # 2. 在故障时间窗口内，根据关键字筛选错误日志
            error_df = pd.DataFrame()
            if not all_component_logs_df.empty:
                logs_in_anomaly_window = all_component_logs_df[
                    (pd.to_datetime(all_component_logs_df['@timestamp']) >= start_dt) &
                    (pd.to_datetime(all_component_logs_df['@timestamp']) <= end_dt)
                ].copy()
                
                if not logs_in_anomaly_window.empty:
                    keyword_regex = "|".join(error_keywords)
                    if "message" in logs_in_anomaly_window.columns:
                        match_condition = logs_in_anomaly_window["message"].str.contains(keyword_regex, case=False, na=False, regex=True)
                        error_df = logs_in_anomaly_window[match_condition]

            # 3. 根据是否找到错误日志，决定生成报告的类型
            if error_df.empty:
                # 未找到错误，启动上下文探查
                if all_component_logs_df.empty:
                    summary = f"对于资源 {components if components else '系统全局'}，在时间范围内没有任何日志条目。这些组件可能已离线。"
                    report = {"summary": summary, "anomalous_components": []}
                else:
                    # V10: 新增活动量分析
                    activity_reports, silent_failure_detected = self._analyze_log_activity(all_component_logs_df, start_dt, end_dt)
                    
                    # V16: 重大修改 - 分析窗口内的活动模式，而不是返回原始日志
                    logs_in_anomaly_window = all_component_logs_df[
                        (pd.to_datetime(all_component_logs_df['@timestamp']) >= start_dt) &
                        (pd.to_datetime(all_component_logs_df['@timestamp']) <= end_dt)
                    ].copy()

                    report = self._create_contextual_report(logs_in_anomaly_window, components if components else ["系统全局"], error_keywords, activity_reports, silent_failure_detected)
            else:
                # 找到了错误日志，生成标准报告
                report = self._create_report(error_df, error_keywords)

                # V-Fix: 当提供了组件列表时，明确报告哪些组件没有发现异常
                if components and "summary" in report and report.get("anomalous_components"):
                    components_with_anomalies = {
                        comp['component'] for comp in report.get('anomalous_components', [])
                    }
                    
                    all_requested_components = set(components)
                    components_without_anomalies = all_requested_components - components_with_anomalies

                    if components_without_anomalies:
                        summary_addition = (
                            f"同时，对于指定的其他组件「{', '.join(sorted(list(components_without_anomalies)))}」"
                            f"未检测到异常日志。"
                        )
                        report['summary'] += f" {summary_addition}"
                        report['components_without_anomalies'] = sorted(list(components_without_anomalies))
            
            report = self.convert_numpy_types(report)

            return ToolResult(output=json.dumps(report, indent=2, ensure_ascii=False))

        except Exception as e:
            error_trace = traceback.format_exc()
            return ToolResult(error=f"日志加载错误: {str(e)}\n{error_trace}")

    def _analyze_log_activity(self, df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> Tuple[Dict[str, str], bool]:
        """
        V10 新增: 分析并比较基线和异常窗口期间的日志活动量。
        V15: 返回一个额外的布尔值来明确标识静默故障。
        """
        if df.empty:
            return {}, False

        df = df.copy()
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])

        baseline_end_dt = start_dt
        baseline_start_dt = baseline_end_dt - timedelta(minutes=5)

        baseline_df = df[
            (df['@timestamp'] >= baseline_start_dt) &
            (df['@timestamp'] < baseline_end_dt)
        ].copy()
        anomaly_df = df[
            (df['@timestamp'] >= start_dt) &
            (df['@timestamp'] <= end_dt)
        ].copy()

        if baseline_df.empty:
            print("基线窗口内无日志数据，无法进行活动量对比。")
            return {}, False

        group_by_col = None
        if 'k8_pod' in df.columns and df['k8_pod'].notna().any():
            group_by_col = 'k8_pod'
        elif 'k8_node_name' in df.columns and df['k8_node_name'].notna().any():
            group_by_col = 'k8_node_name'
        else:
            return {}, False

        activity_reports = {}

        # 1. 计算每个组件在基线期每分钟的日志数，以建立统计基准
        baseline_df['minute'] = baseline_df['@timestamp'].dt.floor('T')
        baseline_rates = baseline_df.groupby([group_by_col, 'minute']).size().reset_index(name='count')
        
        # 2. 计算每个组件基线速率的均值和标准差
        baseline_stats = baseline_rates.groupby(group_by_col)['count'].agg(['mean', 'std']).reset_index()
        baseline_stats['std'] = baseline_stats['std'].fillna(0) # 如果std为NaN，则填充为0

        # 3. 计算每个组件在异常窗口期的平均日志速率
        anomaly_duration_min = (end_dt - start_dt).total_seconds() / 60.0
        if anomaly_duration_min <= 0: anomaly_duration_min = 1
        anomaly_counts = anomaly_df.groupby(group_by_col).size().reset_index(name='total_count')
        anomaly_counts['rate'] = anomaly_counts['total_count'] / anomaly_duration_min

        # 4. 合并统计数据并进行比较
        merged_stats = pd.merge(anomaly_counts, baseline_stats, on=group_by_col, how='left').fillna(0)
        
        silent_failure_detected = False

        for _, row in merged_stats.iterrows():
            component = row[group_by_col]
            anomaly_rate = row['rate']
            baseline_mean = row['mean']
            baseline_std = row['std']

            # V12: 引入基于统计的动态阈值 (均值 + 3倍标准差)
            # 最小标准差确保即使在非常稳定的日志中也能检测到异常
            # 最小速率阈值避免在低活动量组件上产生误报
            dynamic_threshold = baseline_mean + 3 * max(baseline_std, 0.1) 
            MIN_RATE_THRESHOLD = 5 # 每分钟至少要有5条日志才被认为是“激增”

            if anomaly_rate > dynamic_threshold and anomaly_rate > MIN_RATE_THRESHOLD:
                increase_percent = ((anomaly_rate - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 99999
                activity_reports[component] = (
                    f"在「{component}」上检测到活动量激增: "
                    f"从基线平均 {baseline_mean:.1f}条/分钟 "
                    f"飙升至 {anomaly_rate:.1f}条/分钟 (高于正常范围 {increase_percent:.0f}%)。"
                )
            # V12: 新增 - 专门用于检测从静默到活跃的组件
            elif baseline_mean < 1 and anomaly_rate > 10: # 基线几乎无日志，但异常期间日志很多
                 activity_reports[component] = (
                    f"在「{component}」上检测到异常活动: "
                    f"组件从几乎静默 (基线平均 < 1条/分钟) "
                    f"变为高度活跃 ({anomaly_rate:.1f}条/分钟)。"
                )
            # V13: 新增 - 检测活动突然中断 (潜在的死锁/挂起信号)
            elif baseline_mean > 5 and anomaly_rate < 1: # 基线是活跃的，但异常期间突然静默
                 silent_failure_detected = True
                 activity_reports[component] = (
                    f"在「{component}」上检测到活动突然中断: "
                    f"组件从一个活跃状态 (基线平均 {baseline_mean:.1f}条/分钟) "
                    f"陷入几乎完全静默 ({anomaly_rate:.1f}条/分钟)。这可能是服务挂起或死锁的强烈信号，可能是导致故障的根因。"
                )

        return activity_reports, silent_failure_detected

    def _load_all_logs_for_components(self, files: List[str], components: Optional[List[str]], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        V8 新增: 加载指定组件或全部组件在时间范围内的所有日志。
        """
        df = self._load_data(files, "logs", start_dt, end_dt)
        if df.empty or not components:
            return df

        pod_match = df["k8_pod"].isin(components) if 'k8_pod' in df.columns else pd.Series(False, index=df.index)
        node_match = df["k8_node_name"].isin(components) if "k8_node_name" in df.columns else pd.Series(False, index=df.index)
        
        return df[pod_match | node_match]

    def _create_contextual_report(self, df: pd.DataFrame, components: List[str], searched_keywords: List[str], activity_reports: Optional[Dict[str, str]] = None, silent_failure_detected: bool = False) -> Dict[str, Any]:
        """
        V16: 完全重新设计，在未找到明确错误时，提供日志模式的分析摘要，而不是返回原始日志。
        这有助于理解组件在故障期间的实际活动。
        V18: 按组件和节点细分报告，提供更精确的上下文。
        """
        summary_parts = []
        if silent_failure_detected:
            summary_parts.append("**假设: 检测到静默故障.**")
            
        requested_components_str = f"组件 {components}" if components and components != ["系统全局"] else "系统全局"
        summary_parts.append(f"对于 {requested_components_str}，在异常窗口内未找到匹配关键字 {searched_keywords} 的日志。")

        if df.empty:
            summary_parts.append(" 在异常窗口内，这些组件没有任何日志活动记录。")
            return {"summary": " ".join(summary_parts), "anomalous_components": []}

        # --- V18: 核心修改 - 按组件和节点分组进行分析 ---
        if 'k8_pod' not in df.columns or df['k8_pod'].isna().all():
            summary_parts.append("日志数据中缺少有效的'k8_pod'组件标识，无法按组件细分活动模式。")
            return {"summary": " ".join(summary_parts), "anomalous_components": []}

        df_copy = df.copy()
        df_copy['template'] = df_copy['message'].apply(lambda x: self._get_smart_template(x)[0])

        contextual_reports = []
        group_by_col = 'k8_pod'
        node_col = 'k8_node_name' if 'k8_node_name' in df_copy.columns else None

        for component_name, group_df in df_copy.groupby(group_by_col):
            total_logs = len(group_df)
            
            pattern_counts = group_df.groupby('template').agg(
                count=('@timestamp', 'size'),
                first_occurrence=('@timestamp', 'min'),
                last_occurrence=('@timestamp', 'max')
            ).reset_index()
            
            pattern_counts['percentage'] = (pattern_counts['count'] / total_logs) * 100
            top_patterns = pattern_counts.sort_values(by='count', ascending=False).head(3)

            activity_patterns = []
            for _, row in top_patterns.iterrows():
                activity_patterns.append({
                    "template": row['template'],
                    "count": int(row['count']),
                    "percentage": round(row['percentage'], 2),
                    "first_occurrence": pd.to_datetime(row['first_occurrence']).isoformat().replace('+00:00', 'Z'),
                    "last_occurrence": pd.to_datetime(row['last_occurrence']).isoformat().replace('+00:00', 'Z'),
                })
            
            related_node = "N/A"
            if node_col and not group_df[node_col].dropna().empty:
                related_node = group_df[node_col].dropna().mode().iloc[0]

            comp_report = {
                "component": component_name,
                "related_node": related_node,
                "total_logs_in_window": total_logs,
                "activity_patterns": activity_patterns
            }
            
            if activity_reports and component_name in activity_reports:
                comp_report["activity_volume_analysis_summary"] = activity_reports[component_name]

            contextual_reports.append(comp_report)

        if not contextual_reports:
            summary_parts.append("已加载日志，但未能成功聚合出任何组件的活动模式。")
            return {"summary": " ".join(summary_parts), "anomalous_components": []}

        # V19: Limit the number of contextual components for conciseness
        sorted_contextual_reports = sorted(contextual_reports, key=lambda x: x['total_logs_in_window'], reverse=True)
        CONTEXTUAL_REPORT_LIMIT = 5
        reports_to_show = sorted_contextual_reports[:CONTEXTUAL_REPORT_LIMIT]
        other_contextual_components_count = len(sorted_contextual_reports) - len(reports_to_show)

        summary_parts.append("已生成各组件的常规活动模式分析。")

        if activity_reports:
            activity_summary = " ".join(activity_reports.values())
            if activity_summary:
                summary_parts.append(f"日志活动量分析显示: {activity_summary}")

        most_active_component = reports_to_show[0]
        top_pattern_info = ""
        if most_active_component.get('activity_patterns'):
            top_pattern = most_active_component['activity_patterns'][0]
            top_pattern_info = f"最主要的活动模式是 '{top_pattern['template']}' ({top_pattern['percentage']:.0f}%)。"

        summary_parts.append(
            f"其中，「{most_active_component['component']}」(于节点 {most_active_component['related_node']}) "
            f"活动最频繁 (共 {most_active_component['total_logs_in_window']} 条日志)。{top_pattern_info}"
        )

        report = {
            "summary": " ".join(summary_parts),
            "anomalous_components": reports_to_show
        }

        if other_contextual_components_count > 0:
            report["other_contextual_components_summary"] = f"另外 {other_contextual_components_count} 个活跃组件的模式未在此处详述以保持简洁。"

        if activity_reports:
            report["activity_volume_analysis"] = {
                "spike_or_silence_detected": True,
                "details": activity_reports
            }

        return report

    def _create_report(self, df: pd.DataFrame, keywords: List[str]) -> Dict[str, Any]:
        """
        V3: 基于找到的错误日志，进行聚类分析并生成报告。
        此版本增强了摘要，以反映异常日志的数量和严重性。
        """
        if df.empty:
            return {
                "summary": f"在日志中未发现匹配关键字 {keywords} 的条目。",
                "anomalies_found": False,
            }

        df = df.copy()
        # V9: 使用新的智能模板化函数
        df[['template', 'details']] = df['message'].apply(self._get_smart_template).apply(pd.Series)
        
        if 'k8_pod' not in df.columns or df['k8_pod'].isna().all():
            return {
                "summary": "日志数据中缺少有效的'k8_pod'组件标识，无法进行模式分析。",
                "anomalies_found": False
            }

        agg_functions = {
            '@timestamp': ['count', 'min'], # 同时计算count和min
            'k8_node_name': lambda x: next((val for val in x.dropna() if val != 'null'), 'N/A'),
            'details': 'first'  # V9: 保留第一条详细日志
        }
        log_summary = df.groupby(['k8_pod', 'template']).agg(agg_functions).reset_index()
        # 修正列名解包
        log_summary.columns = ['component_name', 'template', 'count', 'first_occurrence', 'related_node', 'details']


        if log_summary.empty:
            return {"summary": "已加载日志，但未能成功聚合出任何有效的错误日志模式。", "anomalies_found": False}

        component_agg = {}
        for _, row in log_summary.iterrows():
            comp_name = row['component_name']
            if comp_name not in component_agg:
                component_agg[comp_name] = {
                    "total_anomalies": 0,
                    "patterns": [],
                    "related_node": row['related_node']
                }
            
            component_agg[comp_name]["total_anomalies"] += row['count']
            component_agg[comp_name]["patterns"].append({
                "template": row['template'],
                "count": row['count'],
                "first_occurrence": row['first_occurrence'], # 存储首次出现时间
                "details": row['details'] # V9: 将details加入pattern
            })

        anomalous_components = []
        sorted_components = sorted(component_agg.items(), key=lambda item: item[1]['total_anomalies'], reverse=True)
        
        # V19: Limit the number of anomalous components in the report for clarity
        ANOMALOUS_COMPONENT_LIMIT = 10
        components_to_report = sorted_components[:ANOMALOUS_COMPONENT_LIMIT]
        other_anomalous_components_count = len(sorted_components) - len(components_to_report)
        
        for name, data in components_to_report:
            data["patterns"].sort(key=lambda x: x['count'], reverse=True)
            
            # V6: Promote the earliest timestamp to the top level
            # FIX: Calculate earliest occurrence BEFORE formatting timestamps to strings
            all_timestamps = [pd.to_datetime(p['first_occurrence']) for p in data["patterns"] if pd.notna(p.get('first_occurrence'))]
            earliest_occurrence = min(all_timestamps) if all_timestamps else None
            
            # Now format timestamps for output
            for pattern in data["patterns"]:
                first_occurrence_ts = pattern.get('first_occurrence')
                if pd.notna(first_occurrence_ts):
                    pattern['first_occurrence'] = pd.to_datetime(first_occurrence_ts).isoformat().replace('+00:00', 'Z')
                else:
                    pattern['first_occurrence'] = 'N/A'

            top_pattern = data["patterns"][0]
            
            anomalous_components.append({
                "component": name,
                "related_node": data['related_node'],
                "total_anomalies": data['total_anomalies'],
                "top_pattern": top_pattern,
                "first_occurrence": earliest_occurrence.isoformat().replace('+00:00', 'Z') if earliest_occurrence else 'N/A'
            })

        # V3: 增强报告，使其更具可读性和信息量
        if not anomalous_components:
            return {"summary": "日志分析完成，未发现任何显著的异常模式。", "anomalous_components": []}

        # --- V4 & V5: Inject domain knowledge with Severity Levels ---
        categorized_anomalies: Dict[str, List[Dict]] = {
            "system_critical": [],
            "application_fatal": [],
            "business_impact_error": [],
            "connectivity_error": [],
            "application_warning": [],
            "unknown": []
        }
        
        for comp in anomalous_components:
            template_lower = comp['top_pattern']['template'].lower()
            severity = "unknown" # Default severity
            for level, patterns in self.LOG_SEVERITY_PATTERNS.items():
                if any(pattern in template_lower for pattern in patterns):
                    severity = level
                    break
            comp["severity"] = severity
            categorized_anomalies[severity].append(comp)

        # --- Generate a more intelligent summary based on severity ---
        summary_parts = []
        # Report most critical issues first
        if categorized_anomalies["system_critical"]:
            comp_names = [c['component'] for c in categorized_anomalies["system_critical"]]
            summary_parts.append(f"在组件「{', '.join(comp_names)}」中检测到系统级致命错误(如OOM、磁盘空间不足)。这很可能是根因。")
        
        if categorized_anomalies["application_fatal"]:
            comp_names = [c['component'] for c in categorized_anomalies["application_fatal"]]
            summary_parts.append(f"在组件「{', '.join(comp_names)}」中检测到应用级错误(如panic、启动失败)。这可能是更深层次问题的症状。")

        if categorized_anomalies["business_impact_error"]:
            comp_names = [c['component'] for c in categorized_anomalies["business_impact_error"]]
            summary_parts.append(f"在组件「{', '.join(comp_names)}」中检测到关键业务流程失败的日志。这直接反映了终端用户受到的影响，是调查的关键起点。")

        if categorized_anomalies["connectivity_error"]:
            comp_names = [c['component'] for c in categorized_anomalies["connectivity_error"]]
            summary_parts.append(f"在组件「{', '.join(comp_names)}」中检测到连接性错误(如超时、拒绝连接)。这强烈表明其下游依赖或相关网络出现问题，是追溯故障链的重要线索。")

        # Only report warnings if no higher severity issues were found
        if not summary_parts and categorized_anomalies["application_warning"]:
            comp_names = [c['component'] for c in categorized_anomalies["application_warning"]]
            summary_parts.append(f"在组件「{', '.join(comp_names)}」中检测到应用级告警日志。")
        
        if not summary_parts:
            summary_parts.append("检测到异常日志，但未匹配到已知的严重错误模式。")

        summary = " ".join(summary_parts)
        # Add details about the most impactful component
        most_affected_component = anomalous_components[0]
        
        # V15: 突出日志风暴
        log_storm_threshold = 100
        total_anomalies = most_affected_component['total_anomalies']
        if total_anomalies > log_storm_threshold:
             summary += f" 其中「{most_affected_component['component']}」发生了日志风暴，产生了 {total_anomalies} 条相关异常记录。"
        
        summary += f" 其中「{most_affected_component['component']}」受影响最严重 (严重等级: {most_affected_component['severity']})，主要错误模式为: '{most_affected_component['top_pattern']['template']}'。"
        
        report = {
            "summary": summary,
            "anomalous_components": anomalous_components,
        }

        if other_anomalous_components_count > 0:
            report["other_anomalous_components_summary"] = f"另外 {other_anomalous_components_count} 个组件也检测到异常，但未在此处详述以保持简洁。"

        return report

    def _extract_structured_info(self, log_message: str) -> Dict[str, Any]:
        """从单条日志消息中提取结构化的信息（IP、错误码等）"""
        if not isinstance(log_message, str):
            return {}

        details = {}
        
        # 尝试将日志消息作为JSON解析
        try:
            log_json = json.loads(log_message)
            if isinstance(log_json, dict):
                # 如果是字典，提取关键字段
                for key, value in log_json.items():
                    if "error" in key.lower():
                        details['error_message'] = str(value)
                    if "trace" in key.lower() or "span" in key.lower():
                        details[key] = str(value)
        except json.JSONDecodeError:
            # 不是JSON，继续正则匹配
            pass

        # 使用正则表达式提取通用信息
        # 提取IP和端口
        ip_port_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)', log_message)
        if ip_port_match:
            details['target_ip'] = ip_port_match.group(1)
            details['target_port'] = ip_port_match.group(2)
        
        # 提取gRPC错误码和描述
        grpc_match = re.search(r'code = (\w+)\s+desc = (.*?)(?:\s*,|"}|\"$)', log_message, re.IGNORECASE)
        if grpc_match:
            details['grpc_code'] = grpc_match.group(1)
            # 清理描述中的转义字符
            desc = grpc_match.group(2).strip('"')
            try:
                # 解码可能因无效的转义序列而失败
                details['grpc_description'] = bytes(desc, "utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                # 如果解码失败，则使用原始字符串
                details['grpc_description'] = desc


        # 提取traceID和spanID
        trace_id_match = re.search(r'traceID=(\w+)', log_message)
        if trace_id_match:
            details['trace_id'] = trace_id_match.group(1)
            
        span_id_match = re.search(r'spanID=(\w+)', log_message)
        if span_id_match:
            details['span_id'] = span_id_match.group(1)

        return details

    def _get_smart_template(self, log_message: str) -> Tuple[str, Dict[str, Any]]:
        """
        V9 新增: 智能日志模板化函数。
        - 尝试将日志解析为JSON。
        - 如果成功，模板是排序后的键的组合，细节是完整的JSON。
        - 如果失败，回退到基于正则表达式的模板化。
        """
        if not isinstance(log_message, str):
            return "non-string log message", {}

        try:
            # 尝试解析JSON
            parsed_json = json.loads(log_message)
            if isinstance(parsed_json, dict):
                # V17: 数据驱动的模板化优化
                # 如果JSON中包含'message'字段，我们认为其内容比顶层键结构更具信息量
                if 'message' in parsed_json and isinstance(parsed_json['message'], str):
                    # 递归调用，但只取模板部分，细节部分依然是完整的原始JSON
                    template, _ = self._get_smart_template(parsed_json['message'])
                    return template, parsed_json

                # 如果没有'message'字段，回退到基于键的模板化
                template = "_".join(sorted(parsed_json.keys()))
                return template, parsed_json
        except (json.JSONDecodeError, TypeError):
            # 如果不是有效的JSON，则回退到基于正则表达式的模板化方法
            pass
            
        # 旧的基于正则表达式的模板化方法
        template = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.,]\d+Z?', '', log_message)
        template = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' <IP> ', template)
        template = re.sub(r'\b[0-9a-fA-F]{8,}\b', ' <HASH> ', template)
        template = re.sub(r'\b\d+\b', ' <NUM> ', template)
        template = re.sub(r'https?://\S+', ' <URL> ', template)
        template = re.sub(r'\s+', ' ', template).strip()
        
        return template[:120], {"original_message": log_message}

    def _get_time_column(self, data_type: str) -> Optional[str]:
        """获取日志数据的时间列名"""
        if data_type == "logs":
            return "@timestamp"
        return None

    def _get_component_columns(self, data_type: str) -> List[str]:
        """获取日志数据的组件列名"""
        if data_type == "logs":
            # 主要通过 k8_pod 来关联组件
            return ["k8_pod"]
        return []

    def _find_log_files(self, start_dt: datetime, end_dt: datetime) -> List[str]:
        """使用通用的、时区正确的文件查找方法来定位日志文件。"""
        all_files = set()
        cst_tz = timedelta(hours=8)

        start_dt_cst = start_dt + cst_tz
        end_dt_cst = end_dt + cst_tz

        # 遍历CST时区下的每一个小时
        current_hour_cst = start_dt_cst.replace(minute=0, second=0, microsecond=0)
        
        while current_hour_cst <= end_dt_cst:
            date_str = current_hour_cst.strftime("%Y-%m-%d")
            hour_str = current_hour_cst.strftime("%H")
            
            # log_filebeat-server_YYYY-MM-DD_HH-*.parquet
            # 根据最新的测试，文件名中的小时部分是固定的，我们使用通配符匹配所有小时
            pattern = f"log-parquet/log_filebeat-server_{date_str}_*.parquet"
            
            # 恢复正确的 data_type 参数
            all_files.update(
                self._find_files_by_time_range(
                    start_dt, end_dt, "", pattern, force_date=date_str
                )
            )
            # 前进到下一个小时
            current_hour_cst += timedelta(hours=1)
            
        return list(all_files)


    def _get_relevant_columns(
        self, data_type: str, available_columns: List[str]
    ) -> List[str]:
        """确定需要加载的日志列"""
        # 日志数据中的关键列
        log_columns = [
            "k8_namespace",  # 命名空间
            "@timestamp",  # 时间戳 (UTC时区)
            "k8_pod",  # Pod名称
            "message",  # 日志消息
            "k8_node_name",  # 节点名称
            "agent_name",  # 代理名称
            "log_level",  # 日志级别(如果有)
        ]
        return [col for col in log_columns if col in available_columns]

