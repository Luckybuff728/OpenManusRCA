#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.agent.rca import RCAAgent
from app.logger import logger
from app.schema import AgentState
from app.prompt.v1_rca_context import TOPOLOGY


# V-Final-Fix: Standardize the final reason. Order matters: most specific first.
REASON_MAP = {
    # 1. General error catch-all (should come before symptoms it might cause)
    r"error|exception|panic|fail|timeout": "Application Error",
    
    # 1. Specific technical root causes
    r"jvm|memory|oom": "Service JVM Issue",
    r"i/o|disk|storage|io": "Node Disk I/O Consumption High",
    r"database|tidb|qps|sql": "Database Error",
    
    # 2. Specific error types
    r"config": "Configuration Error",
    r"code": "Code Error",
    
    # # 3. General error catch-all (should come before symptoms it might cause)
    # r"error|exception|panic|fail|timeout": "Application Error",
    
    # 4. Symptom-based issues (often caused by errors, so checked later)
    r"network|latency|rrt": "Pod Network Latency",
    # r"timeout|canceled": "Application Timeout Anomaly",
    # r"log": "Application Error Log",
}

# V-Final-Fix-3: Ensure component name is standardized and in English
KNOWN_COMPONENTS = set(TOPOLOGY["services"].keys())
for service_data in TOPOLOGY["services"].values():
    KNOWN_COMPONENTS.update(service_data.get("pods", []))

def standardize_component(name: str) -> str:
    """Standardizes the component name against a known list."""
    if name in KNOWN_COMPONENTS:
        return name
    
    # Try a case-insensitive match
    name_lower = name.lower()
    for known_comp in KNOWN_COMPONENTS:
        if name_lower == known_comp.lower():
            logger.info(f"Standardized component '{name}' to '{known_comp}'.")
            return known_comp
            
    logger.warning(f"Component '{name}' not found in known topology. Using original value.")
    return name

def standardize_reason(reason_text: str) -> str:
    """Converts a free-form reason text into a standardized category."""
    reason_lower = reason_text.lower()
    for pattern, standard_reason in REASON_MAP.items():
        if re.search(pattern, reason_lower):
            return standard_reason
    # Fallback: if no keyword matches, use the original text but try to clean it
    # by taking the part before the first colon or sentence, keeping it short.
    return reason_text.split(':')[0].split('.')[0].strip()


# 将固定的指令部分定义为常量，提高可读性和可维护性
INIT_PROMPT_TEMPLATE = textwrap.dedent(
    """
    # 故障分析任务

    请对以下故障案例进行根因分析。

    - **案例ID**: {uuid}
    - **故障时间范围**: 从 {start_time} 到 {end_time}
    - **用户查询**: {query}

    你的分析将遵循系统提示中定义的、由假设驱动的科学方法。请立即开始你的分析。
    """
)


async def run_rca(
    uuid: str,
    query: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """运行根因分析代理

    Args:
        uuid: 故障案例的唯一标识符
        query: 用户查询文本
        start_time: 可选的分析开始时间
        end_time: 可选的分析结束时间
        output_file: 可选的输出文件路径

    Returns:
        分析结果
    """
    # 如果未提供时间范围，从查询中尝试提取
    if not start_time or not end_time:
        # 尝试从查询中提取时间信息 - 支持多种常见格式
        time_patterns = [
            # 匹配 "from YYYY-MM-DDThh:mm:ssZ to YYYY-MM-DDThh:mm:ssZ"
            r"from\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+to\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
            # 匹配 "between YYYY-MM-DDThh:mm:ssZ and YYYY-MM-DDThh:mm:ssZ"
            r"between\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+and\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
            # 匹配 "YYYY-MM-DDThh:mm:ssZ to YYYY-MM-DDThh:mm:ssZ"
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+to\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
            # 匹配 "during YYYY-MM-DDThh:mm:ssZ to YYYY-MM-DDThh:mm:ssZ"
            r"during\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+to\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
        ]

        for pattern in time_patterns:
            match = re.search(pattern, query)
            if match:
                extracted_start = match.group(1)
                extracted_end = match.group(2)

                if not start_time:
                    start_time = extracted_start
                if not end_time:
                    end_time = extracted_end
                break

    # 创建RCA代理实例
    logger.info(f"创建RCA代理分析故障案例 {uuid}, 时间范围: {start_time} 到 {end_time}")
    agent = await RCAAgent.create(uuid=uuid, query=query)

    # 构造初始化提示
    init_prompt = INIT_PROMPT_TEMPLATE.format(
        uuid=uuid, start_time=start_time, end_time=end_time, query=query
    )

    try:
        # 运行代理
        start_time_exec = time.time()
        result = await agent.run(init_prompt)
        duration = time.time() - start_time_exec
        logger.info(f"分析完成，耗时 {duration:.2f} 秒")

        # --- 核心改进：更严格的结果验证 ---
        try:
            # result 应该是 final_report 工具返回的、干净的JSON字符串
            final_json_result = json.loads(result)
            
            # 检查是否是错误报告
            if "error" in final_json_result:
                logger.error(
                    f"分析未成功完成。最终状态: {agent.state}。"
                    f"错误信息: {final_json_result.get('error')}"
                )
                return final_json_result

            # 确保结果包含必要字段
            raw_reason = final_json_result.get("reason", "unknown")
            raw_component = final_json_result.get("component", "unknown")
            
            complete_result = {
                "uuid": uuid,
                "component": standardize_component(raw_component),
                "reason": standardize_reason(raw_reason),
                "time": final_json_result.get("time", datetime.now().isoformat()),
                "reasoning_trace": final_json_result.get("reasoning_trace", [])
            }
            
            # 验证结果有效性
            if complete_result["component"] != "unknown" and complete_result["reason"] != "unknown":
                logger.info(f"生成有效分析结果: 组件={complete_result['component']}, 原因={complete_result['reason']}")
            else:
                logger.warning("分析结果包含unknown值，可能分析不完整")

            # 保存结果
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(complete_result, f, ensure_ascii=False, indent=2)
                logger.info(f"分析结果已保存至 {output_file}")

            return complete_result

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"处理或解析最终结果时出错: {e}. 原始结果: {result[:1000]}")
            return {"error": "无法解析最终报告的JSON", "raw_result": result[:1000]}
        except Exception as e:
            logger.error(f"处理结果时发生未知错误: {str(e)}")
            return {"error": str(e), "raw_result": result[:1000] if 'result' in locals() and result else "无结果"}

    finally:
        # 确保资源被清理
        await agent.cleanup()

    return {"error": "未知错误", "raw_result": ""}


def load_cases_from_input_json(input_file: str) -> List[Dict[str, Any]]:
    """从input.json加载故障案例列表

    Args:
        input_file: input.json文件路径

    Returns:
        故障案例列表
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            cases = json.load(f)

        logger.info(f"从 {input_file} 加载了 {len(cases)} 个故障案例")
        return cases
    except Exception as e:
        logger.error(f"加载故障案例文件 {input_file} 失败: {str(e)}")
        return []


def extract_time_range(description: str) -> Tuple[Optional[str], Optional[str]]:
    """从故障描述中提取时间范围

    Args:
        description: 故障描述文本

    Returns:
        (开始时间, 结束时间) 元组
    """
    # 匹配各种时间范围模式
    patterns = [
        # 匹配 "from YYYY-MM-DDThh:mm:ssZ to YYYY-MM-DDThh:mm:ssZ"
        r"from\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+to\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
        # 匹配 "between YYYY-MM-DDThh:mm:ssZ and YYYY-MM-DDThh:mm:ssZ"
        r"between\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+and\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
        # 匹配 "YYYY-MM-DDThh:mm:ssZ to YYYY-MM-DDThh:mm:ssZ"
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+to\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
        # 匹配 "during YYYY-MM-DDThh:mm:ssZ to YYYY-MM-DDThh:mm:ssZ"
        r"during\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+to\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
    ]

    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            return match.group(1), match.group(2)

    return None, None


async def batch_analyze(cases: List[Dict[str, Any]], output_dir: str) -> None:
    """批量分析多个故障案例

    Args:
        cases: 故障案例列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for case in cases:
        uuid = case.get("uuid")
        description = case.get("Anomaly Description", "")

        if not uuid or not description:
            logger.warning(f"跳过无效案例: {case}")
            continue

        start_time, end_time = extract_time_range(description)

        if not start_time or not end_time:
            logger.warning(f"无法从描述中提取时间范围: {description}")
            continue

        output_file = os.path.join(output_dir, f"{uuid}.json")

        # 增加检查，如果结果文件已存在，则跳过
        if os.path.exists(output_file):
            logger.info(f"案例 {uuid} 的结果文件已存在，跳过分析。")
            # 记录结果为成功，因为它已经存在
            results.append(
                {
                    "uuid": uuid,
                    "success": True,
                    "status": "skipped",
                    "output_file": output_file,
                }
            )
            continue

        logger.info(f"开始分析案例 {uuid}")
        try:
            result = await run_rca(
                uuid=uuid,
                query=description,
                start_time=start_time,
                end_time=end_time,
                output_file=output_file,
            )

            results.append(
                {
                    "uuid": uuid,
                    "success": "error" not in result,
                    "status": "completed",
                    "output_file": output_file,
                }
            )

            logger.info(f"案例 {uuid} 分析完成，结果保存至 {output_file}")

        except Exception as e:
            logger.error(f"分析案例 {uuid} 失败: {str(e)}")
            results.append({"uuid": uuid, "success": False, "status": "failed", "error": str(e)})

    # 保存批处理结果摘要
    summary_file = os.path.join(output_dir, "batch_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_cases": len(cases),
                "success_count": sum(1 for r in results if r.get("success", False)),
                "failure_count": sum(1 for r in results if not r.get("success", False)),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"批处理完成，摘要已保存至 {summary_file}")


async def main():
    """入口函数"""
    parser = argparse.ArgumentParser(description="微服务根因分析工具")

    # 单案例分析参数
    parser.add_argument("--uuid", type=str, help="故障案例的唯一标识符")
    parser.add_argument("--query", type=str, help="用户查询文本")
    parser.add_argument(
        "--start-time", type=str, help="分析开始时间 (UTC格式，如 2025-06-05T16:10:00Z)"
    )
    parser.add_argument(
        "--end-time", type=str, help="分析结束时间 (UTC格式，如 2025-06-05T16:40:00Z)"
    )
    parser.add_argument("--output", type=str, help="结果输出文件路径")

    # 批量分析参数
    parser.add_argument("--input-json", type=str, default='dataset/phaseone/input.json', help="故障案例输入JSON文件路径")
    parser.add_argument(
        "--case-id", type=str, help="仅分析指定ID的案例 (与--input-json一起使用)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="批量分析结果输出目录"
    )

    args = parser.parse_args()

    # 批量分析模式
    if args.input_json:
        cases = load_cases_from_input_json(args.input_json)
        if not cases:
            logger.error("没有找到有效的故障案例，退出")
            sys.exit(1)

        # 如果指定了特定案例ID，只分析该案例
        if args.case_id:
            cases = [case for case in cases if case.get("uuid") == args.case_id]
            if not cases:
                logger.error(f"在输入文件中未找到ID为 {args.case_id} 的案例")
                sys.exit(1)

        await batch_analyze(cases, args.output_dir)
        return

    # 单案例分析模式
    if not args.uuid:
        logger.error("未指定UUID，请使用--uuid参数提供故障案例的唯一标识符")
        sys.exit(1)

    if not args.query:
        # 如果未提供查询，使用默认查询模板
        if args.start_time and args.end_time:
            args.query = f"The system experienced an anomaly from {args.start_time} to {args.end_time}. Please infer the possible cause."
        else:
            args.query = (
                "Please analyze this case and infer the possible cause of the anomaly."
            )

    try:
        logger.info(f"开始分析故障案例 {args.uuid}")
        result = await run_rca(
            uuid=args.uuid,
            query=args.query,
            start_time=args.start_time,
            end_time=args.end_time,
            output_file=args.output,
        )

        # 打印结果
        if "error" not in result:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"错误: {result['error']}", file=sys.stderr)
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("操作被用户中断")
    except Exception as e:
        logger.error(f"执行分析时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "345fbe93-80" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "74a44ae7-81" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "38ee3d45-82" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "b1ab098d-83" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "27956295-84" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "d99a98a0-233" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "0718e0f9-92" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "0410d710-226" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "aa100326-220" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "e4ef8a12-193" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input-1-4.json --output-dir results 
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "96e4b852-189" --output-dir results 
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "bc2f1dfa-114" --output-dir results 
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "2e8873a5-370" --output-dir results 
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "5028909d-391" --output-dir results 
# python rca_main.py --input-json phasetwo/input-two-4.json --output-dir results 
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "08e3187d-131" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "fc44280d-133" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "a061c961-225" --output-dir results


# python rca_main.py --input-json dataset/phaseone/input.json --case-id "fe25ee9b-143" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "555d145f-99" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "54002a5c-155" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "934844b2-156" --output-dir results

# python rca_main.py --input-json dataset/phaseone/input.json --case-id "046aefa7-170" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "cc451a15-174" --output-dir results

# 11 python rca_main.py --input-json dataset/phaseone/input.json --case-id "49e9814e-266" --output-dir results
# 11 python rca_main.py --input-json dataset/phaseone/input.json --case-id "d0c7cd07-267" --output-dir results

# python rca_main.py --input-json dataset/phaseone/input.json --case-id "0a9fce75-340" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "5028909d-391" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "91b14fe1-386" --output-dir results


# 11 python rca_main.py --input-json dataset/phaseone/input.json --case-id "d6b961d4-587" --output-dir results
# 11 python rca_main.py --input-json dataset/phaseone/input.json --case-id "8ce0cd7e-578" --output-dir results



# python rca_main.py --input-json dataset/phaseone/input.json --case-id "251c4f53-179" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "eaf978d7-208" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "bc6f6978-239" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "aa100326-220" --output-dir results ad
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "8c7ed41d-249" --output-dir results ad
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "099ce7e2-274" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "009be6db-313" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "e539dea4-322" --output-dir results

# python rca_main.py --input-json dataset/phaseone/input.json --case-id "d32bcd36-104" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "3df3b41f-105" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "68727566-367" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "0419ba04-373" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "d0ecec7e-381" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "9b0004fb-390" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "cd049902-400" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "e566fa68-404" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "a7ed866b-406" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "959cfb67-502" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "c9ba0b22-566" --output-dir results

# python rca_main.py --input-json dataset/phaseone/input.json --case-id "a0761b38-172" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "aa100326-220" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "980853cb-240" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "53ab3ccb-326" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "c3377c5c-337" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "8c7ed41d-249" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "1ae8a5c3-555" --output-dir results
# python rca_main.py --input-json dataset/phaseone/input.json --case-id "52c080d4-556" --output-dir results
