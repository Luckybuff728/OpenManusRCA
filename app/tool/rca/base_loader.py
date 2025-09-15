import glob
import json
import os
import re
import traceback
from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytz

from app.tool.base import BaseTool, ToolResult


class BaseLoader(BaseTool):
    """数据加载器的基类"""
    # 移除测试钩子
    
    @property
    def dataset_dir(self) -> str:
        """返回数据集的基础目录路径"""
        # 假设脚本从项目根目录运行
        return os.path.join(os.getcwd(), "dataset", "phaseone")

    # 数据集路径
    BASE_DATA_DIR: ClassVar[str] = os.path.join("dataset", "phaseone")

    def convert_numpy_types(self, obj):
        """递归转换numpy类型为Python原生类型，用于JSON序列化

        Args:
            obj: 要转换的对象

        Returns:
            转换后的对象，可以被JSON序列化
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (dict, pd.Series)):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(item) for item in obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        else:
            return obj

    def _parse_time_range(
        self, start_time: str, end_time: str
    ) -> Tuple[datetime, datetime]:
        """
        解析UTC时间字符串为时区感知的datetime对象。
        此版本强制为所有解析出的时间附加UTC时区，以避免时区感知与朴素时间的计算错误。
        """
        try:
            # 优先处理标准的ISO格式，包含 'Z'
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        except ValueError:
            # 如果ISO格式解析失败，使用pandas进行更灵活的解析
            try:
                start_dt = pd.to_datetime(start_time, utc=True).to_pydatetime()
                end_dt = pd.to_datetime(end_time, utc=True).to_pydatetime()
            except Exception as e:
                raise ValueError(f"无法将 '{start_time}' 和 '{end_time}' 解析为有效的时间范围: {str(e)}")

        # 双重保险：确保最终返回的对象有时区信息
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=pytz.UTC)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=pytz.UTC)

        return start_dt, end_dt

    def _find_files_by_time_range(
        self,
        start_dt: datetime,
        end_dt: datetime,
        data_type: str,
        file_pattern: str,
        force_date: Optional[str] = None,
    ) -> List[str]:
        """
        根据给定的时间范围和文件模式查找相关的数据文件。
        此版本经过重构，以支持强制指定日期，解决了跨时区和跨天问题。
        """
        all_files = set()
        
        # --- 核心修复：简化逻辑，完全依赖 force_date ---
        # 移除所有内部日期计算逻辑。调用者（如apm_loader）现在全权负责
        # 计算正确的CST日期并将其作为 force_date 传入。
        
        if not force_date:
            # 如果没有提供 force_date，这是一个逻辑错误，因为上层加载器应该总是计算它。
            # 记录一个警告，并尝试使用start_dt作为后备，但这不应该是常规路径。
            print(f"警告: _find_files_by_time_range 在没有 force_date 的情况下被调用。文件查找可能不准确。")
            date_to_check = start_dt.strftime('%Y-%m-%d')
        else:
            date_to_check = force_date
            
        search_path = os.path.join(
            self.dataset_dir, date_to_check, data_type, file_pattern
        )
        # print(f"DEBUG: Searching for files in: {search_path}") # 增加日志
        found_files = glob.glob(search_path, recursive=True)
        # if found_files:
        #     print(f"DEBUG: Found files: {found_files}") # 增加日志
        all_files.update(found_files)

        return list(all_files)

    def _load_data(self, files: List[str], data_type: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        从文件中加载数据，并通过谓词下推（predicate pushdown）实现精准高效的加载。
        """
        dfs = []
        
        # 1. 构造用于谓词下推的过滤器
        time_col = self._get_time_column(data_type)
        filters = []
        if time_col and files:
            try:
                # --- 智能类型嗅探和动态过滤器构建 ---
                # 1. 读取第一个文件的schema来确定时间列的类型
                schema = pq.read_schema(files[0])
                time_col_type = str(schema.field(time_col).type)

                # 2. 根据类型动态构建过滤器
                if 'string' in time_col_type or 'object' in time_col_type:
                    # 类型为字符串: '2025-06-05T19:10:07.000Z'
                    start_str = start_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                    end_str = end_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                    filters.append((time_col, ">=", start_str))
                    filters.append((time_col, "<=", end_str))
                    # print(f"DEBUG: Applying STRING-based filters for column '{time_col}': {filters}")
                elif 'int64' in time_col_type or 'timestamp' in time_col_type:
                    # 类型为整数时间戳 (毫秒或微秒)
                    # TracesLoader 使用 'startTimeMillis'，其单位是毫秒。
                    # 我们将统一使用毫秒时间戳进行过滤，这是最直接且健壮的方案。
                    start_ts_ms = int(start_dt.timestamp() * 1000)
                    end_ts_ms = int(end_dt.timestamp() * 1000)
                    
                    filters.append((time_col, ">=", start_ts_ms))
                    filters.append((time_col, "<=", end_ts_ms))
                    # print(f"DEBUG: Applying INTEGER-based (millisecond) filters for column '{time_col}': {filters}")
                else:
                    print(f"警告: 未知的Parquet时间列类型 '{time_col_type}'，将回退到全量加载。")
            
            except Exception as e:
                print(f"DEBUG: 无法确定时间列类型 ({e})，将回退到全量加载。")


        for file in files:
            try:
                # 如果过滤器为空（例如类型嗅探失败），则不使用谓词下推
                effective_filters = filters if filters else None
                file_df = pd.read_parquet(file, filters=effective_filters, engine='pyarrow')

                if not file_df.empty:
                    file_df["source_file"] = os.path.basename(file)
                    dfs.append(file_df)
            except Exception as e:
                print(f"加载 Parquet 文件 {file} 时出错 (可能不支持谓词下推): {str(e)}")
                # 如果谓词下推失败，回退到加载整个文件再过滤
                try:
                    full_df = pd.read_parquet(file, engine='pyarrow')
                    filtered_df = self._filter_time_range(full_df, data_type, start_dt, end_dt)
                    if not filtered_df.empty:
                        filtered_df["source_file"] = os.path.basename(file)
                        dfs.append(filtered_df)
                except Exception as e2:
                    print(f"回退加载文件 {file} 时也出错: {str(e2)}")
                    continue

        if not dfs:
            return pd.DataFrame()

        final_df = pd.concat(dfs, ignore_index=True)
        # 确认并打印最终加载的数据范围
        if not final_df.empty and time_col in final_df.columns:
            final_df[time_col] = pd.to_datetime(final_df[time_col], errors='coerce', utc=True)
            min_ts = final_df[time_col].min()
            max_ts = final_df[time_col].max()
            # print(f"INFO: Successfully loaded {len(final_df)} rows for {data_type}. Final time range (UTC): {min_ts} to {max_ts}")
            
        return final_df

    def _get_relevant_columns(
        self, data_type: str, available_columns: List[str]
    ) -> List[str]:
        """根据数据类型和可用列确定需要加载的相关列"""
        raise NotImplementedError("子类必须实现此方法")

    def _filter_time_range(
        self, df: pd.DataFrame, data_type: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        """根据时间范围筛选数据，并确保时区正确处理。"""
        if df.empty:
            return df

        time_col = self._get_time_column(data_type)
        if not time_col or time_col not in df.columns:
            return df

        try:
            # 1. 确保输入的时间范围是时区感知的 (UTC)
            if start_dt.tzinfo is None:
                start_dt = start_dt.tz_localize("UTC")
            if end_dt.tzinfo is None:
                end_dt = end_dt.tz_localize("UTC")

            # 2. 转换DataFrame中的时间列为时区感知的datetime对象 (UTC)
            if pd.api.types.is_numeric_dtype(df[time_col]):
                # 处理时间戳 (毫秒或微秒)
                unit = 'ms' if 'Millis' in time_col else 'us'
                df[time_col] = pd.to_datetime(df[time_col], unit=unit, errors='coerce', utc=True)
            else:
                # 处理字符串格式的时间
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
            
            # 删除转换失败的行
            df.dropna(subset=[time_col], inplace=True)

            # 3. 执行安全的时区感知比较
            filtered_df = df[(df[time_col] >= start_dt) & (df[time_col] <= end_dt)]
            
            return filtered_df

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"过滤时间范围时出错: {str(e)}\n{error_trace}")
            return pd.DataFrame() # 返回空DataFrame以避免下游错误

    def _get_time_column(self, data_type: str) -> Optional[str]:
        """获取对应数据类型的时间列名"""
        raise NotImplementedError("子类必须实现此方法")

    # ----------------------------------------------------------------------------------
    #  下面的通用过滤方法将被废弃，因为不同数据源的组件列完全不同，
    #  强制使用通用逻辑是导致“无数据”错误的核心原因。
    #  每个子加载器都应该实现自己的、专门的组件过滤逻辑。
    # ----------------------------------------------------------------------------------

    # def _filter_component(
    #     self, df: pd.DataFrame, data_type: str, component: str
    # ) -> pd.DataFrame:
    #     """根据组件名称筛选数据"""
    #     if df.empty:
    #         return df

    #     # 确定组件列
    #     component_cols = self._get_component_columns(data_type)
    #     if not component_cols:
    #         return df

    #     # 根据组件列筛选
    #     mask = pd.Series(False, index=df.index)

    #     for col in component_cols:
    #         if col in df.columns:
    #             # 使用字符串包含匹配
    #             col_mask = (
    #                 df[col].astype(str).str.contains(component, case=False, na=False)
    #             )
    #             mask = mask | col_mask

    #     return df[mask]

    # def _get_component_columns(self, data_type: str) -> List[str]:
    #     """获取对应数据类型的组件列名"""
    #     raise NotImplementedError("子类必须实现此方法")
        
    # def _filter_component_type(
    #     self, df: pd.DataFrame, data_type: str, component_type: str
    # ) -> pd.DataFrame:
    #     """根据组件类型筛选数据"""
    #     if component_type not in self.POSSIBLE_COMPONENTS:
    #         return df  # 无效的组件类型，返回原始数据

    #     component_list = self.POSSIBLE_COMPONENTS[component_type]
    #     component_columns = self._get_component_columns(data_type)

    #     if not component_columns:
    #         return df  # 找不到组件列，返回原始数据

    #     # 创建筛选条件
    #     filter_condition = False
    #     for col in component_columns:
    #         if col in df.columns:
    #             # 对于每个组件列，检查是否有匹配的组件
    #             col_filter = df[col].isin(component_list)
    #             filter_condition = filter_condition | col_filter

    #     if isinstance(filter_condition, bool) and filter_condition is False:
    #         return df  # 没有创建有效的筛选条件

    #     return df[filter_condition]
    
    def _smart_sample(
        self, df: pd.DataFrame, data_type: str, limit: int
    ) -> pd.DataFrame:
        """智能采样数据，确保保留关键点"""
        if len(df) <= limit:
            return df

        # 确保时间排序列
        time_col = self._get_time_column(data_type)

        if time_col and time_col in df.columns:
            # 排序确保采样覆盖整个时间范围
            df = df.sort_values(time_col)

        # 计算采样间隔
        step = max(len(df) // limit, 1)

        # 系统采样，确保覆盖整个数据范围
        sampled_indices = list(range(0, len(df), step))

        # 如果采样结果仍超过限制，随机选择
        if len(sampled_indices) > limit:
            import random

            random.seed(42)
            sampled_indices = sorted(random.sample(sampled_indices, limit))

        return df.iloc[sampled_indices].reset_index(drop=True)
