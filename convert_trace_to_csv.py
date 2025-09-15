import glob
import json
import os

import numpy as np
import pandas as pd


def extract_parent_span_id(references):
    """从references字段提取父span ID，如果没有父span则返回'root'"""
    # 处理numpy.ndarray类型
    if isinstance(references, np.ndarray):
        if len(references) > 0:
            ref = references[0]  # 获取第一个引用
            if isinstance(ref, dict) and ref.get("refType") == "CHILD_OF":
                return ref.get("spanID")
    # 处理list类型
    elif isinstance(references, list) and len(references) > 0:
        ref = references[0]
        if isinstance(ref, dict) and ref.get("refType") == "CHILD_OF":
            return ref.get("spanID")

    # 如果没有父span，则标记为'root'
    return "root"


def extract_tag_value(tags, key, fallback_key=None):
    """从tags字段中提取指定key的值，如果未找到且提供了fallback_key，则尝试提取fallback_key的值"""
    # 如果是在查找状态码，只使用http.status_code
    if key == "status.code" and fallback_key == "http.status_code":
        # 只提取http.status_code
        http_status = None

        # 处理numpy.ndarray类型
        if isinstance(tags, np.ndarray):
            for tag in tags:
                if isinstance(tag, dict) and tag.get("key") == "http.status_code":
                    http_status = tag.get("value")
                    if http_status is not None:
                        try:
                            # 转换为整数
                            return int(http_status)
                        except (ValueError, TypeError):
                            # 如果转换失败但是字符串可以转为整数，则转换
                            if isinstance(http_status, str) and http_status.isdigit():
                                return int(http_status)
                            return http_status
        # 处理list类型
        elif isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, dict) and tag.get("key") == "http.status_code":
                    http_status = tag.get("value")
                    if http_status is not None:
                        try:
                            # 转换为整数
                            return int(http_status)
                        except (ValueError, TypeError):
                            # 如果转换失败但是字符串可以转为整数，则转换
                            if isinstance(http_status, str) and http_status.isdigit():
                                return int(http_status)
                            return http_status

        # 如果没有找到http.status_code，返回0
        return 0

    # 如果不是在查找状态码，则正常提取
    # 处理numpy.ndarray类型
    if isinstance(tags, np.ndarray):
        for tag in tags:
            if isinstance(tag, dict) and tag.get("key") == key:
                value = tag.get("value")
                return value
    # 处理list类型
    elif isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict) and tag.get("key") == key:
                value = tag.get("value")
                return value

    # 如果没有找到主键且有备用键（且不是状态码查询），则尝试提取备用键
    if fallback_key and key != "status.code":
        # 处理numpy.ndarray类型
        if isinstance(tags, np.ndarray):
            for tag in tags:
                if isinstance(tag, dict) and tag.get("key") == fallback_key:
                    value = tag.get("value")
                    return value
        # 处理list类型
        elif isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, dict) and tag.get("key") == fallback_key:
                    value = tag.get("value")
                    return value

    # 如果都没找到，返回None
    return None


def extract_process_tag_value(process, key):
    """从process.tags字段中提取指定key的值，支持备选字段"""
    if not isinstance(process, dict) or "tags" not in process:
        return None

    tags = process.get("tags", [])
    primary_key = key
    fallback_key = None

    # 设置备选字段
    if key == "name":
        fallback_key = "podName"
    elif key == "node_name":
        fallback_key = "nodeName"

    # 首先尝试查找主键
    value = None
    # 处理numpy.ndarray类型
    if isinstance(tags, np.ndarray):
        for tag in tags:
            if isinstance(tag, dict) and tag.get("key") == primary_key:
                value = tag.get("value")
                return value
    # 处理list或其他可迭代类型
    elif isinstance(tags, list) or hasattr(tags, "__iter__"):
        for tag in tags:
            if isinstance(tag, dict) and tag.get("key") == primary_key:
                value = tag.get("value")
                return value

    # 如果没有找到主键，尝试查找备选键
    if value is None and fallback_key is not None:
        # 处理numpy.ndarray类型
        if isinstance(tags, np.ndarray):
            for tag in tags:
                if isinstance(tag, dict) and tag.get("key") == fallback_key:
                    value = tag.get("value")
                    return value
        # 处理list或其他可迭代类型
        elif isinstance(tags, list) or hasattr(tags, "__iter__"):
            for tag in tags:
                if isinstance(tag, dict) and tag.get("key") == fallback_key:
                    value = tag.get("value")
                    return value

    return None


def extract_service_name(process):
    """从process中提取服务名称"""
    if not isinstance(process, dict):
        return None

    # 首先尝试从serviceName字段提取
    if "serviceName" in process:
        service_name = process["serviceName"]
        # 将redis服务名称修改为redis-cart
        if service_name == "redis":
            return "redis-cart"
        return service_name

    # 如果没有serviceName字段，尝试从tags中提取name字段
    if "tags" in process:
        tags = process.get("tags", [])
        # 处理numpy.ndarray类型
        if isinstance(tags, np.ndarray):
            for tag in tags:
                if isinstance(tag, dict) and tag.get("key") == "name":
                    name = tag.get("value")
                    # 如果name包含实例编号（如frontend-2），则提取基础服务名称
                    if name and "-" in name:
                        return name.rsplit("-", 1)[0]
                    return name
        # 处理list或其他可迭代类型
        elif isinstance(tags, list) or hasattr(tags, "__iter__"):
            for tag in tags:
                if isinstance(tag, dict) and tag.get("key") == "name":
                    name = tag.get("value")
                    # 如果name包含实例编号（如frontend-2），则提取基础服务名称
                    if name and "-" in name:
                        return name.rsplit("-", 1)[0]
                    return name

    return None


def convert_trace_to_csv(input_file, output_file):
    """将trace parquet文件转换为CSV文件"""
    print(f"处理文件: {input_file}")

    # 读取parquet文件
    df = pd.read_parquet(input_file)

    # 提取父span ID - 确保从references中正确提取，如果没有父span则标记为'root'
    df["ParentspanID"] = df["references"].apply(extract_parent_span_id)

    # 从process.tags中提取pod_name和node_name
    df["pod_name"] = df["process"].apply(
        lambda x: extract_process_tag_value(x, "name") if isinstance(x, dict) else None
    )
    df["node_name"] = df["process"].apply(
        lambda x: (
            extract_process_tag_value(x, "node_name") if isinstance(x, dict) else None
        )
    )

    # 从process中提取服务名称
    df["service"] = df["process"].apply(extract_service_name)

    # 只从http.status_code提取状态码，不使用status.code
    df["status_code"] = df["tags"].apply(
        lambda x: extract_tag_value(x, "status.code", "http.status_code")
    )

    # 选择需要的列
    result_df = df[
        [
            "traceID",
            "spanID",
            "ParentspanID",
            "operationName",
            "service",
            "startTime",
            "startTimeMillis",
            "duration",
            "pod_name",
            "node_name",
            "status_code",
        ]
    ]

    # 按traceID排序，确保相同traceID的记录挨在一起
    result_df = result_df.sort_values(by=["traceID", "startTime"])

    # 保存为CSV
    result_df.to_csv(output_file, index=False)
    print(f"已保存CSV文件: {output_file}")

    # 统计根span和子span的数量
    root_count = (result_df["ParentspanID"] == "root").sum()
    child_count = len(result_df) - root_count
    print(
        f"总记录数: {len(result_df)}, 根span数: {root_count}, 子span数: {child_count}, 子span比例: {child_count/len(result_df):.2%}"
    )

    # 检查status_code字段
    status_code_counts = result_df["status_code"].value_counts()
    print(f"status_code值分布: {status_code_counts.to_dict()}")

    # 检查service字段
    service_counts = result_df["service"].value_counts()
    print(f"服务分布 (前10个): {dict(service_counts.head(10).to_dict())}")

    # 检查pod_name和node_name是否为空
    pod_name_null = result_df["pod_name"].isna().sum()
    node_name_null = result_df["node_name"].isna().sum()
    service_null = result_df["service"].isna().sum()
    print(f"pod_name为空的行数: {pod_name_null} ({pod_name_null/len(result_df):.2%})")
    print(
        f"node_name为空的行数: {node_name_null} ({node_name_null/len(result_df):.2%})"
    )
    print(f"service为空的行数: {service_null} ({service_null/len(result_df):.2%})")

    # 检查traceID分组情况
    trace_counts = result_df.groupby("traceID").size()
    print(f"不同的traceID数量: {len(trace_counts)}")
    print(f"每个trace的平均span数: {trace_counts.mean():.2f}")
    print(f"最大span数的trace: {trace_counts.max()}")
    print(f"最小span数的trace: {trace_counts.min()}")

    return result_df


def process_directory(input_dir, output_dir=None):
    """处理目录中的所有trace文件"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), "trace-csv")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有parquet文件
    parquet_files = glob.glob(os.path.join(input_dir, "trace_jaeger-span_*.parquet"))

    print(f"找到 {len(parquet_files)} 个parquet文件")

    # 可以选择处理部分文件进行测试
    test_mode = False  # 设置为False处理所有文件
    if test_mode:
        parquet_files = parquet_files[:1]  # 只处理第一个文件
        print(f"测试模式: 只处理 {len(parquet_files)} 个文件")

    for input_file in parquet_files:
        # 构建输出文件名
        base_name = os.path.basename(input_file).replace(".parquet", ".csv")
        output_file = os.path.join(output_dir, base_name)

        # 转换文件
        convert_trace_to_csv(input_file, output_file)

        if test_mode:
            # 在测试模式下检查生成的CSV文件
            print("\n检查生成的CSV文件:")
            result_df = pd.read_csv(output_file)
            print(f"CSV文件列: {result_df.columns.tolist()}")
            print(f"前5行数据:\n{result_df.head()}")

            # 检查traceID是否连续
            print("\n检查traceID是否连续:")
            sample_trace_id = result_df["traceID"].iloc[0]
            trace_sample = result_df[result_df["traceID"] == sample_trace_id]
            print(f"示例traceID: {sample_trace_id}, 包含 {len(trace_sample)} 个span")

            # 检查status_code字段类型
            status_codes = result_df[result_df["status_code"].notna()]
            if not status_codes.empty:
                sample_row = status_codes.iloc[0]
                print(
                    f"\nstatus_code示例值: {sample_row['status_code']}, 类型: {type(sample_row['status_code'])}"
                )

            # 检查ParentspanID
            print("\nParentspanID示例:")
            # 显示一些根span
            root_samples = result_df[result_df["ParentspanID"] == "root"].head(3)
            print("根span示例:")
            for _, row in root_samples.iterrows():
                print(f"SpanID: {row['spanID']}, ParentspanID: {row['ParentspanID']}")

            # 显示一些子span
            child_samples = result_df[result_df["ParentspanID"] != "root"].head(3)
            print("\n子span示例:")
            for _, row in child_samples.iterrows():
                print(f"SpanID: {row['spanID']}, ParentspanID: {row['ParentspanID']}")


if __name__ == "__main__":
    # 设置输入目录
    input_dir = "dataset/phaseone/2025-06-06/trace-parquet"
    output_dir = "dataset/phaseone/2025-06-06/trace-csv"

    # 处理目录
    process_directory(input_dir, output_dir)

    print("转换完成！")
