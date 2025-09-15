import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.tool.base import ToolResult
from app.tool.rca.base_loader import BaseLoader


def _normalize_component_name(name: str) -> str:
    """将组件名转换为统一的可比较格式（小写，仅字母数字）。"""
    if not isinstance(name, str):
        return ""
    return re.sub(r'[^a-z0-9]', '', name.lower())

# --- Helper functions for nested data processing ---
def extract_parent_span_id(references):
    if isinstance(references, (list, np.ndarray)) and len(references) > 0:
        ref = references[0]
        if isinstance(ref, dict) and ref.get("refType") == "CHILD_OF":
            return ref.get("spanID")
    return "root"

def extract_tag_value(tags, key, fallback_key=None):
    if not isinstance(tags, (list, np.ndarray)): return None
    for tag in tags:
        if isinstance(tag, dict) and tag.get("key") == key:
            return tag.get("value")
    if fallback_key:
        for tag in tags:
            if isinstance(tag, dict) and tag.get("key") == fallback_key:
                return tag.get("value")
    return None # 返回None而不是0，以区分“未找到”和“值为0”

def extract_process_tag_value(process, key):
    if isinstance(process, dict) and 'tags' in process:
        tags = process.get("tags", [])
        if isinstance(tags, (list, np.ndarray)):
            for tag in tags:
                if isinstance(tag, dict) and tag.get("key") == key: return tag.get("value")
    return None

def extract_service_name(process):
    if isinstance(process, dict) and "serviceName" in process:
        return process["serviceName"]
    return None


class TracesLoader(BaseLoader):
    """加载并分析微服务调用链数据，以支持根因定位。"""

    name: str = "traces_loader"
    description: str = "加载并分析服务调用链数据，自动检测延迟和错误异常。"

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "description": "故障案例的唯一标识符。"},
            "start_time": {"type": "string", "description": "开始时间，UTC格式。"},
            "end_time": {"type": "string", "description": "结束时间，UTC格式。"},
            "component": {"type": "string", "description": "要重点分析的组件名。"},
        },
        "required": ["uuid", "start_time", "end_time", "component"],
    }

    def _process_nested_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "references" in df.columns:
            df["ParentspanID"] = df["references"].apply(extract_parent_span_id)
        if "process" in df.columns:
            # V3 修复: 根据调试发现的实际数据格式，使用正确的键 'name' 和 'node_name'
            df["pod_name"] = df["process"].apply(lambda x: extract_process_tag_value(x, 'name'))
            df["node_name"] = df["process"].apply(lambda x: extract_process_tag_value(x, 'node_name'))
            df["service"] = df["process"].apply(extract_service_name)
        if "tags" in df.columns:
            # 明确处理 status_code 可能为 None 的情况
            df['status_code'] = df['tags'].apply(
                lambda x: extract_tag_value(x, 'rpc.grpc.status_code', 'http.status_code')
            )
            df['error_tag'] = df['tags'].apply(lambda x: extract_tag_value(x, 'error'))
            df['status_message'] = df['tags'].apply(lambda x: extract_tag_value(x, 'rpc.grpc.status_message'))

        if "logs" in df.columns:
            df["log_error_reason"] = df["logs"].apply(self._extract_error_from_logs)

        return df

    def _extract_error_from_logs(self, logs: Any) -> Optional[str]:
        """
        从 span 的 logs 字段中提取错误信息。
        根据 OpenTelemetry 规范，错误事件通常有一个 'event'='error' 或 'exception' 的日志条目。
        """
        if not isinstance(logs, (list, np.ndarray)):
            return None
        for log_entry in logs:
            if isinstance(log_entry, dict) and 'fields' in log_entry:
                fields = log_entry.get('fields', [])
                if not isinstance(fields, (list, np.ndarray)):
                    continue
                
                log_as_dict = {field.get('key'): field.get('value') for field in fields if isinstance(field, dict)}

                # 检查 OpenTelemetry 异常日志记录约定
                if log_as_dict.get('event') == 'exception':
                    exception_message = log_as_dict.get('exception.message', 'Unknown exception')
                    return f"Span logged an exception: {exception_message}"
                
                # 检查简单的错误事件
                # README示例显示了 'message.event'，所以同时检查 'event' 和 'message.event'
                if log_as_dict.get('event') == 'error' or log_as_dict.get('message.event') == 'error':
                    error_description = log_as_dict.get('message', 'undescribed error')
                    return f"Span contains an error event: {error_description}"
        return None

    def _get_relevant_trace_ids(self, files: List[str], start_dt: datetime, end_dt: datetime, component: str) -> set:
        """
        以内存高效的方式扫描数据文件，找到与特定组件和时间范围相关的所有唯一trace ID。
        此版本经过简化，不再需要 component_type。
        """
        relevant_ids = set()
        
        # V-Fix: Handle redis-cart naming inconsistency where 'redis-cart' in logic corresponds to 'redis' in data.
        search_component = 'redis' if component == 'redis-cart' else component
        normalized_component = _normalize_component_name(search_component)

        filter_columns = ["traceID", "process", "operationName", "tags", "startTime"]
        
        start_ts = int(start_dt.timestamp() * 1_000_000)
        end_ts = int(end_dt.timestamp() * 1_000_000)
        time_filters = [('startTime', '>=', start_ts), ('startTime', '<=', end_ts)]
        
        def check_op_name(op_name):
            if not isinstance(op_name, str): return False
            # V2: 使用更宽松的包含关系检查，而不是严格的正则匹配
            return normalized_component in _normalize_component_name(op_name)

        def check_tags(tags):
            if not isinstance(tags, list): return False
            for tag in tags:
                if isinstance(tag, dict) and tag.get('key') in ['peer.service', 'net.peer.ip'] and 'value' in tag:
                    # V2: 使用包含关系，更灵活
                    if normalized_component in _normalize_component_name(str(tag.get('value', ''))):
                        return True
            return False

        for file_path in files:
            try:
                table = pq.read_table(file_path, columns=filter_columns, filters=time_filters, use_threads=True)
                if table.num_rows == 0:
                    continue
                
                chunk_df = table.to_pandas()
                
                # 在单个文件的小数据块上应用过滤逻辑
                if "process" in chunk_df.columns:
                    chunk_df["pod_name"] = chunk_df["process"].apply(lambda x: extract_process_tag_value(x, 'name'))
                    chunk_df["service"] = chunk_df["process"].apply(extract_service_name)

                # 检查 pod_name 和 service_name (V2: 使用包含关系)
                condition_pod = chunk_df['pod_name'].apply(lambda x: normalized_component in _normalize_component_name(x)) if 'pod_name' in chunk_df.columns and chunk_df['pod_name'].notna().any() else pd.Series(False, index=chunk_df.index)
                condition_service = chunk_df['service'].apply(lambda x: normalized_component in _normalize_component_name(x)) if 'service' in chunk_df.columns and chunk_df['service'].notna().any() else pd.Series(False, index=chunk_df.index)

                condition_op = chunk_df['operationName'].apply(check_op_name) if 'operationName' in chunk_df.columns else pd.Series(False, index=chunk_df.index)
                condition_tag = chunk_df['tags'].apply(check_tags) if 'tags' in chunk_df.columns else pd.Series(False, index=chunk_df.index)

                filtered_chunk = chunk_df[condition_pod | condition_service | condition_op | condition_tag]
                
                if not filtered_chunk.empty:
                    relevant_ids.update(filtered_chunk["traceID"].unique())
            except Exception:
                continue
                
        return relevant_ids
    
    def _load_all_traces(self, files: List[str], trace_ids: set) -> pd.DataFrame:
        """
        使用pyarrow谓词下推从Parquet文件中高效加载一组traceID的所有spans。
        """
        if not trace_ids:
            return pd.DataFrame()

        all_trace_dfs = []
        filters = [("traceID", "in", list(trace_ids))]
        
        for file in files:
            try:
                table = pq.read_table(file, filters=filters, use_threads=True)
                if table.num_rows > 0:
                    all_trace_dfs.append(table.to_pandas())
            except Exception:
                continue
        
        return pd.concat(all_trace_dfs, ignore_index=True) if all_trace_dfs else pd.DataFrame()

    async def execute(self, uuid: str, start_time: str, end_time: str, component: Optional[str] = None) -> Any:
        try:
            # 1. 设置
            start_dt, end_dt = self._parse_time_range(start_time, end_time)

            # --- 最佳实践 (V3): 带有隔离带的窗口设置 ---
            # 1. 扩展异常窗口以保证数据完整性
            MIN_ANOMALY_DURATION_SECONDS = 120
            if (end_dt - start_dt).total_seconds() < MIN_ANOMALY_DURATION_SECONDS:
                end_dt = start_dt + timedelta(seconds=MIN_ANOMALY_DURATION_SECONDS)

            # 2. 定义包含隔离带的基线窗口 (用于计算延迟阈值)
            QUARANTINE_MINUTES = 5
            BASELINE_WINDOW_MINUTES = 10
            threshold_end_dt = start_dt - timedelta(minutes=QUARANTINE_MINUTES)
            threshold_start_dt = threshold_end_dt - timedelta(minutes=BASELINE_WINDOW_MINUTES)
            
            threshold_files = self._find_trace_files(threshold_start_dt, threshold_end_dt)
            
            operation_thresholds = {}
            if threshold_files:
                op_durations = defaultdict(list)
                for file in threshold_files:
                    try:
                        # 仅加载计算P95所需的最小列
                        df = pd.read_parquet(file, columns=["process", "operationName", "duration", "references", "tags"])
                        if df.empty: continue
                        
                        processed_df = self._process_nested_data(df)
                        if "duration" not in processed_df.columns: continue

                        processed_df["duration"] = pd.to_numeric(processed_df["duration"], errors='coerce')
                        processed_df.dropna(subset=['pod_name', 'operationName', 'duration'], inplace=True)

                        for (pod, op_name), op_df in processed_df.groupby(["pod_name", "operationName"]):
                             op_durations[(pod, op_name)].extend(op_df["duration"].tolist())
                    except Exception:
                        continue
                
                for (pod, op_name), durations in op_durations.items():
                    if len(durations) >= 5:
                        p95 = int(pd.Series(durations).quantile(0.95))
                        operation_thresholds[(pod, op_name)] = max(p95, 1000)

            # 3. 核心分析逻辑: V7 - 聚合分析
            analysis_files = self._find_trace_files(start_dt, end_dt)
            if not analysis_files:
                return ToolResult(output=json.dumps({
                    "summary": "在指定的时间范围内未找到任何调用链数据文件。",
                    "has_error_or_latency": False,
                }))

            analysis_mode = f"Component-focused: {component}"
            relevant_trace_ids = self._get_relevant_trace_ids(analysis_files, start_dt, end_dt, component)

            if not relevant_trace_ids:
                return ToolResult(output=json.dumps({
                    "summary": f"未找到与组件 '{component}' 相关的任何调用链数据。",
                    "has_error_or_latency": False,
                }))

            all_anomaly_tuples = []
            total_traces_analyzed = 0

            # 分块加载和处理以优化内存
            for chunk_of_ids in self.chunk_list(list(relevant_trace_ids), 500):
                df = self._load_traces_by_ids(analysis_files, chunk_of_ids)
                if df.empty:
                    continue
                
                processed_df = self._process_nested_data(df)
                total_traces_analyzed += len(chunk_of_ids)

                for trace_id, trace_df in processed_df.groupby("traceID"):
                    trace_tree = self._build_trace_tree(trace_df)
                    anomaly_tuples = self._analyze_trace_tree_and_extract_anomalies(trace_tree, operation_thresholds)
                    if anomaly_tuples:
                        all_anomaly_tuples.extend(anomaly_tuples)
            
            # 4. 生成报告
            if not all_anomaly_tuples:
                return ToolResult(output=json.dumps({
                    "summary": f"已对组件 '{component}' 相关的 {total_traces_analyzed} 条调用链进行分析，未发现任何错误或显著延迟。",
                    "has_error_or_latency": False,
                }))
            
            report = self._create_aggregated_report(all_anomaly_tuples, total_traces_analyzed, component)
            report = self.convert_numpy_types(report)
            return ToolResult(output=json.dumps(report, indent=2, ensure_ascii=False))

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return ToolResult(error=f"调用链加载器执行出错: {str(e)}\n{error_trace}")
    
    def _get_relevant_trace_ids_from_df(self, df: pd.DataFrame, component: str) -> set:
        """
        从已加载的DataFrame中根据组件名筛选相关的trace ID。
        """
        if df.empty:
            return set()
            
        relevant_trace_ids = set()

        # 标准化组件名以进行匹配
        norm_component = _normalize_component_name(component)
        
        def check_op_name(op_name):
            if not isinstance(op_name, str): return False
            return norm_component in _normalize_component_name(op_name)

        def check_tags(tags):
            if not isinstance(tags, list): return False
            for tag in tags:
                if isinstance(tag, dict) and tag.get('key') == 'peer.service' and isinstance(tag.get('value'), str):
                    if norm_component in _normalize_component_name(tag['value']):
                        return True
            return False

        # 检查 'process' 列中的 'serviceName'
        service_name_matches = df['process'].apply(lambda p: isinstance(p, dict) and 'serviceName' in p and norm_component in _normalize_component_name(p['serviceName']))

        # 检查 'operationName'
        op_name_matches = df['operationName'].apply(check_op_name)

        # 检查 'tags'
        tags_matches = df['tags'].apply(check_tags)

        # 合并所有匹配条件
        final_mask = service_name_matches | op_name_matches | tags_matches
        
        if 'traceID' in df.columns:
            relevant_trace_ids.update(df.loc[final_mask, 'traceID'].unique())
            
        return relevant_trace_ids


    def _build_trace_tree(self, df: pd.DataFrame) -> Dict[str, Any]:
        """从扁平的span列表构建调用树，并进行分析。"""
        # 确保ParentspanID存在
        if "ParentspanID" not in df.columns:
            df["ParentspanID"] = df["references"].apply(extract_parent_span_id)

        span_map = {row['spanID']: row.to_dict() for _, row in df.iterrows()}
        root_spans = []
        
        for span_id, span in span_map.items():
            parent_id = span.get('ParentspanID', 'root')
            if parent_id in span_map:
                parent = span_map[parent_id]
                if 'children' not in parent:
                    parent['children'] = []
                parent['children'].append(span)
            else:
                root_spans.append(span)

        # 确保返回的spans是字典列表，而不是Pandas Series
        return {"traceID": df['traceID'].iloc[0], "spans": root_spans}
    
    def chunk_list(self, data: list, size: int):
        """将列表分块"""
        for i in range(0, len(data), size):
            yield data[i:i + size]


    def _analyze_trace_tree_and_extract_anomalies(self, trace_tree: Dict, thresholds: Dict) -> List[Tuple]:
        """
        V2: 分析单个调用链树，智能识别调用方、被调用方和根源组件。
        元组格式: (异常路径, 异常类型, 耗时, 根源组件)
        """
        all_anomalies = []
        for root_span in trace_tree.get("spans", []):
            all_anomalies.extend(self._find_anomalies_recursively(root_span, thresholds, trace_tree["traceID"]))

        if not all_anomalies:
            return []

        span_map = {span['spanID']: span for span in self._flatten_tree(trace_tree.get("spans", []))}

        extracted_tuples = []
        for anomaly in all_anomalies:
            span_id = anomaly['details']['span_id']
            span = span_map.get(span_id)
            if not span: continue

            span_service = anomaly['source_component']
            parent_id = span.get('ParentspanID')
            parent_span = span_map.get(parent_id) if parent_id else None
            caller_service = parent_span.get("service") if parent_span else "entrypoint"
            
            # 如果父子服务相同，说明是内部耗时，将调用方设为外部调用者
            if caller_service == span_service and parent_span:
                grandparent_id = parent_span.get('ParentspanID')
                grandparent_span = span_map.get(grandparent_id) if grandparent_id else None
                caller_service = grandparent_span.get("service") if grandparent_span else "entrypoint"

            # 确定被调用方 (callee)
            peer_service = extract_tag_value(span.get('tags', []), 'peer.service')
            callee_service = peer_service if peer_service else span_service

            # 确定责任组件 (culprit)
            # 如果是客户端span，责任在下游；否则责任在当前服务
            span_kind = extract_tag_value(span.get('tags', []), 'span.kind')
            culprit = callee_service if span_kind == 'client' else span_service
            
            # 修正：当 redis-cart 是被调用者时，明确责任归属
            if callee_service == 'redis':
                callee_service = 'redis-cart'
                culprit = 'redis-cart'


            anomaly_path = f"{caller_service} -> {callee_service}"
            
            extracted_tuples.append((
                anomaly_path,
                anomaly['type'],
                anomaly['details']['duration_ms'],
                culprit,
                span.get('startTimeMillis') # V10: Add timestamp
            ))
            
        return extracted_tuples

    def _flatten_tree(self, spans: List[Dict]) -> List[Dict]:
        """将树状结构的span列表扁平化。"""
        flat_list = []
        q = list(spans)
        while q:
            span = q.pop(0)
            flat_list.append(span)
            if 'children' in span and isinstance(span['children'], list):
                q.extend(span['children'])
        return flat_list

    def _find_parent_span(self, spans: List[Dict], child_span_id: str) -> Optional[Dict]:
        """在span列表中查找一个span的父span。"""
        for span in spans:
            for child in span.get('children', []):
                if child['spanID'] == child_span_id:
                    return span
            # 递归查找
            found = self._find_parent_span(span.get('children', []), child_span_id)
            if found:
                return found
        return None

    def _find_anomalies_recursively(self, span: Dict, thresholds: Dict, trace_id: str, depth: int = 0) -> List[Dict]:
        """
        与旧版类似，但现在只负责扁平地找出所有异常点，并附加一些上下文。
        V2: 增强了错误检测逻辑。
        """
        anomalies = []
        span_duration = span.get('duration', 0)
        operation_name = span.get('operationName', 'unknown_op')
        service_name = extract_service_name(span.get('process'))
        
        # 1. 错误分析 (V2: 增强逻辑)
        is_error = False
        error_reason = ""

        # 1a. 检查 'error' 标签
        error_tag_val = span.get('error_tag')
        if error_tag_val is True or str(error_tag_val).lower() == 'true':
            is_error = True
            error_reason = "操作在执行过程中报告了错误 (error=true)。"

        # 1b. 检查 'logs' 字段中的错误事件
        if not is_error:
            log_error_reason = span.get('log_error_reason')
            if log_error_reason:
                is_error = True
                error_reason = log_error_reason

        # 1c. 如果没有错误标签或日志，则检查状态码
        if not is_error:
            status_code = span.get('status_code')
            if status_code is not None:
                try:
                    numeric_status_code = int(float(status_code))
                    # gRPC 错误码 > 0, HTTP 错误码 >= 400
                    if 0 < numeric_status_code < 20:  # gRPC 错误
                        is_error = True
                        error_reason = f"操作返回了非成功的gRPC状态码: {numeric_status_code}。"
                    elif numeric_status_code >= 400:  # HTTP 错误
                        is_error = True
                        error_reason = f"操作返回了客户端或服务端错误HTTP状态码: {numeric_status_code}。"
                except (ValueError, TypeError):
                    pass # status_code 不是有效的数字，忽略

        if is_error:
            anomalies.append({
                "trace_id": trace_id,
                "type": "error",
                "source_component": service_name,
                "表现": error_reason,
                "details": {
                    "duration_ms": round(span_duration / 1000, 2),
                    "operation": operation_name,
                    "span_id": span.get('spanID'),
                    "depth": depth
                }
            })

        # 2. 延迟分析
        threshold_key = (service_name, operation_name)
        if threshold_key in thresholds and span_duration > thresholds[threshold_key]:
            children_duration = sum(child.get('duration', 0) for child in span.get('children', []))
            self_duration = span_duration - children_duration
            
            # 只有当自身耗时（非下游调用）也超过阈值的某个比例时，才认为是它自己的问题
            # 这避免了仅仅因为下游慢而将上游标记为延迟异常
            if self_duration > thresholds[threshold_key] * 0.5:
                anomalies.append({
                    "trace_id": trace_id,
                    "type": "latency",
                    "source_component": service_name,
                    "表现": f"自身执行耗时 {self_duration/1000:.2f}ms, 超过阈值。",
                    "details": {
                        "duration_ms": round(span_duration / 1000, 2),
                        "self_duration_ms": round(self_duration / 1000, 2),
                        "threshold_ms": round(thresholds.get(threshold_key, 0) / 1000, 2),
                        "operation": operation_name,
                        "span_id": span.get('spanID'),
                        "depth": depth
                    }
                })

        # 递归分析子span
        for child in span.get('children', []):
            anomalies.extend(self._find_anomalies_recursively(child, thresholds, trace_id, depth + 1))
            
        return anomalies

    def _create_aggregated_report(self, anomaly_tuples: List[Tuple], total_traces_analyzed: int, source_component: str) -> Dict[str, Any]:
        """
        V4: Downstream Hotspot Analysis with Timestamp.
        Focuses on aggregating outbound calls from the source_component to find the downstream component
        that is the most likely cause of latency or errors, and reports the first occurrence time.
        """
        from collections import Counter
        import pandas as pd

        if not anomaly_tuples:
            return {
                "summary": "已分析追踪，但未发现可聚合的错误或延迟模式。",
                "has_error_or_latency": False
            }

        # V10: anomaly_tuples format now includes timestamp
        df = pd.DataFrame(anomaly_tuples, columns=['path', 'type', 'duration', 'culprit', 'timestamp_ms'])
        
        df[['caller', 'callee']] = df['path'].str.split(' -> ', expand=True, n=1)

        # V-Final-Fix-Enhanced: Broaden the analysis. The goal is to find the hotspot,
        # which could be the source_component itself (internal issue) or a downstream
        # service (outbound issue). We analyze any anomaly where the source component
        # is either the caller or the culprit.
        relevant_df = df[(df['caller'] == source_component) | (df['culprit'] == source_component)]

        # After filtering, if the DataFrame is empty, it means the source_component
        # was not directly involved in any detected anomalies.
        if relevant_df.empty:
            return {
                "summary": f"对组件 '{source_component}' 的调用链分析完成，未发现其作为调用方或根本原因的任何错误或显著延迟。",
                "has_error_or_latency": False,
            }

        # 核心逻辑：我们关心的是哪个组件(culprit)最可疑，可能是source_component自己或下游
        # 我们按 "culprit" (责任方) 分组，因为它最能代表问题的根源
        hotspot_stats = relevant_df.groupby('culprit').agg(
            total_duration_ms=('duration', 'sum'),
            error_count=('type', lambda x: (x == 'error').sum()),
            call_count=('type', 'count'),
            first_occurrence_ms=('timestamp_ms', 'min')
        ).reset_index()

        if hotspot_stats.empty:
            return {
                "summary": f"分析了 {total_traces_analyzed} 条与 '{source_component}' 相关的追踪，但未发现明确的异常模式。",
                "has_error_or_latency": False,
            }
        
        # 评分逻辑: 错误优先，延迟其次
        # 将错误数归一化，延迟归一化
        max_errors = hotspot_stats['error_count'].max()
        max_duration = hotspot_stats['total_duration_ms'].max()
        
        hotspot_stats['score'] = 0.0
        if max_errors > 0:
            hotspot_stats['score'] += 0.7 * (hotspot_stats['error_count'] / max_errors)
        if max_duration > 0:
            hotspot_stats['score'] += 0.3 * (hotspot_stats['total_duration_ms'] / max_duration)

        # 找到分数最高的组件
        hotspot = hotspot_stats.sort_values(by='score', ascending=False).iloc[0]
        hotspot_component = hotspot['culprit']
            
        # 获取首次异常的时间戳
        first_occurrence_ts = pd.to_datetime(hotspot['first_occurrence_ms'], unit='ms', utc=True)
        time_str = first_occurrence_ts.strftime('%H:%M:%SZ')

        # 构建摘要
        summary_parts = []
        hotspot_error_count = int(hotspot['error_count'])
        hotspot_total_calls = int(hotspot['call_count'])
        
        if hotspot_error_count > 0:
            summary_parts.append(f"{hotspot_error_count}/{hotspot_total_calls} 次调用出错")
        
        # 如果主要问题是延迟
        if hotspot['score'] < 0.7 or hotspot_error_count == 0:
            avg_latency = hotspot['total_duration_ms'] / hotspot_total_calls if hotspot_total_calls > 0 else 0
            summary_parts.append(f"平均延迟 {avg_latency:.2f}ms")

        reason_summary = " 和 ".join(summary_parts)

        # 根据热点是自身还是下游，生成不同的摘要
        if hotspot_component == source_component:
            summary = (
                f"对组件 '{source_component}' 的调用链分析完成。在 {total_traces_analyzed} 条相关调用链中，"
                f"该组件「自身」被识别为主要问题点 (首次异常于 {time_str})，"
                f"其主要问题表现为: {reason_summary}。"
                f" **行动建议**: 深入调查组件「{source_component}」的内部状态（如日志、基础设施指标）。"
            )
        else:
            summary = (
                f"对组件 '{source_component}' 的出站调用分析完成。在 {total_traces_analyzed} 条相关调用链中，"
                f"下游组件「{hotspot_component}」被识别为主要问题点 (首次异常于 {time_str})，"
                f"其主要问题表现为: {reason_summary}。"
                f" **行动建议**: 立即对组件「{hotspot_component}」进行健康检查或进行下一轮追踪分析。"
            )

        # 为了调试和透明度，可以保留所有下游的统计数据
        all_downstream_stats = hotspot_stats.sort_values(by='score', ascending=False).to_dict('records')

        return {
            "summary": summary,
            "has_error_or_latency": True,
            "hotspot_analysis": {
                    "component": hotspot_component,
                    "reason": f"该组件在所有下游依赖中，其错误和延迟的加权分数最高。主要问题: {reason_summary}。",
                    "first_occurrence": time_str,
                    "all_downstream_stats": all_downstream_stats
                }
        }

    def _find_trace_files(self, start_dt: datetime, end_dt: datetime) -> List[str]:
        """
        使用通用的、时区正确的文件查找方法来定位调用链文件。
        """
        all_files = set()
        cst_tz = timedelta(hours=8)

        start_dt_cst = start_dt + cst_tz
        end_dt_cst = end_dt + cst_tz

        # 遍历CST时区下的每一个小时
        current_hour_cst = start_dt_cst.replace(minute=0, second=0, microsecond=0)
        
        while current_hour_cst <= end_dt_cst:
            date_str = current_hour_cst.strftime("%Y-%m-%d")
            hour_str = current_hour_cst.strftime("%H")
            
            # trace_jaeger-span_YYYY-MM-DD_HH-*.parquet
            pattern = f"trace_jaeger-span_{date_str}_{hour_str}-*.parquet"
            
            # 恢复正确的 data_type 参数
            all_files.update(
                self._find_files_by_time_range(
                    start_dt, end_dt, "trace-parquet", pattern, force_date=date_str
                )
            )
            # 前进到下一个小时
            current_hour_cst += timedelta(hours=1)
            
        return list(all_files)

    def _find_error_trace_ids(self, files: List[str]) -> set:
        """快速扫描文件，只寻找包含错误标志的Trace ID。"""
        error_trace_ids = set()
        for file in files:
            try:
                df = pd.read_parquet(file, columns=['traceID', 'tags'])
                if df.empty: continue
                
                # 高效地检查tags列
                # 假设tags是list of dicts
                exploded_tags = df.explode('tags')
                error_spans = exploded_tags[
                    (exploded_tags['tags'].apply(lambda x: isinstance(x, dict) and x.get('key') == 'error' and x.get('value') == True))
                ]
                
                if not error_spans.empty:
                    # 由于explode了，需要找到原始的traceID
                    original_indices = error_spans.index.unique()
                    error_trace_ids.update(df.loc[original_indices, 'traceID'].unique())

            except Exception:
                continue
        return error_trace_ids

    def _find_top_latency_trace_ids(self, files: List[str], k: int) -> set:
        """扫描文件，找到总耗时最长的前K个Trace ID。"""
        root_spans = []
        for file in files:
            try:
                df = pd.read_parquet(file, columns=['traceID', 'duration', 'references'])
                if df.empty: continue

                # 根span的references字段通常是空的或不包含CHILD_OF
                df['is_root'] = df['references'].apply(lambda x: not isinstance(x, list) or not any(ref.get('refType') == 'CHILD_OF' for ref in x if isinstance(ref, dict)))
                
                file_root_spans = df[df['is_root']]
                if not file_root_spans.empty:
                    root_spans.append(file_root_spans[['traceID', 'duration']])
            except Exception:
                continue
        
        if not root_spans:
            return set()
            
        all_root_spans_df = pd.concat(root_spans, ignore_index=True)
        top_k = all_root_spans_df.nlargest(k, 'duration')
        return set(top_k['traceID'].unique())

    def _load_traces_by_ids(self, files: List[str], trace_ids: set) -> pd.DataFrame:
        """根据给定的Trace ID列表，从文件中加载完整的Trace数据。"""
        if not trace_ids or not files:
            return pd.DataFrame()

        all_dfs = []
        # 使用pyarrow的filter进行高效过滤
        filters = [('traceID', 'in', list(trace_ids))]
        
        for file in files:
            try:
                table = pq.read_table(file, filters=filters)
                if table.num_rows > 0:
                    all_dfs.append(table.to_pandas())
            except Exception:
                continue
        
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def _get_all_trace_ids_in_range(self, files: List[str], start_dt: datetime, end_dt: datetime) -> List[str]:
        """
        高效扫描数据文件，仅加载traceID列，以获取在指定时间范围内的所有唯一trace ID。
        """
        if not files:
            return []

        start_ts = int(start_dt.timestamp() * 1_000_000)
        end_ts = int(end_dt.timestamp() * 1_000_000)
        
        filters = [('startTime', '>=', start_ts), ('startTime', '<=', end_ts)]
        
        all_trace_ids = set()
        
        for file in files:
            try:
                # 仅读取traceID列以节省内存
                table = pq.read_table(file, columns=['traceID'], filters=filters, use_threads=True)
                if table.num_rows > 0:
                    all_trace_ids.update(table.to_pandas()['traceID'].unique())
            except Exception:
                continue
                
        return list(all_trace_ids)

    def _get_relevant_columns(self, data_type: str, available_columns: List[str]) -> List[str]:
        """获取调用链数据需要的相关列。"""
        required_cols = list(set([
            "traceID", "spanID", "operationName", "startTime", "startTimeMillis", 
            "duration", "process", "tags", "references", "logs"
        ]))
        return [col for col in required_cols if col in available_columns]

    def _get_time_column(self, data_type: str) -> Optional[str]:
        """指定用于时间过滤的列名。"""
        return "startTime"

    def _analyze_latency(self, df: pd.DataFrame, operation_thresholds: Dict[Tuple[str, str], float]) -> List[Dict[str, Any]]:
        anomalies_list = []
        if df.empty or "duration" not in df.columns: return anomalies_list
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)

        for _, row in df.iterrows():
            pod = row.get("pod_name")
            operation = row.get("operationName")
            duration = row.get("duration", 0)

            if not pod or not isinstance(pod, str) or not operation: continue

            op_threshold = operation_thresholds.get((pod, operation))
            if op_threshold is None: continue
            
            if duration > op_threshold:
                anomalies_list.append({
                    "name": pod, "level": "pod", "metric": operation,
                    "value": duration, "threshold": op_threshold,
                    "reason": f"操作 '{operation}' 的耗时 {duration:.0f}µs 超过了阈值 {op_threshold:.0f}µs。",
                    "first_occurrence": str(pd.to_datetime(row.get("startTimeMillis", 0), unit="ms", utc=True)),
                    "traceID": row.get("traceID")
                })
        return anomalies_list

    def _analyze_errors(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        anomalies_list = []
        if df.empty or "status_code" not in df.columns: return anomalies_list
        
        # 确保 status_code 是数值类型
        df['status_code'] = pd.to_numeric(df["status_code"], errors='coerce').fillna(0)
        error_df = df[df["status_code"] >= 400]

        for _, row in error_df.iterrows():
            pod_name = row.get("pod_name", "unknown_pod")
            operation_name = row.get("operationName", "unknown_operation")
            
            if not isinstance(pod_name, str): pod_name = "unknown_pod"

            anomalies_list.append({
                "name": pod_name,
                "level": "pod",
                "metric": operation_name,
                "value": row.get("status_code"),
                "threshold": 399,
                "reason": f"操作 '{operation_name}' 返回了错误状态码 {row.get('status_code')}。",
                "first_occurrence": str(pd.to_datetime(row.get("startTimeMillis", 0), unit="ms", utc=True)),
                "traceID": row.get("traceID")
            })
        return anomalies_list
