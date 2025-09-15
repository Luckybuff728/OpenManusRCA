import json
from typing import Any, Dict, List

from app.tool.base import BaseTool, ToolResult

class FinalReport(BaseTool):
    name: str = "final_report"
    description: str = "当你已经明确了故障的单一根本原因后，调用此工具来提交最终的分析报告。"

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "uuid": {
                "type": "string",
                "description": "当前故障案例的唯一标识符。",
            },
            "component": {
                "type": "string",
                "description": "导致故障的根因组件。必须是具体、有效的组件名称（例如 `adservice`, `productcatalogservice-1`, 或 `aiops-k8s-08`）。严禁包含层级前缀。",
            },
            "reason": {
                "type": "string",
                "description": "对根本原因的简洁、准确的英文描述。例如：'High disk I/O utilization on the node caused resource contention.'",
            },
            "reasoning_trace": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "number"},
                        "action": {"type": "string", "description": "对该步骤执行的工具调用或分析行为的英文描述。"},
                        "observation": {"type": "string", "description": "对该步骤观察到的、基于证据的关键发现的英文描述。"},
                    },
                    "required": ["step", "action", "observation"],
                },
                "description": "一个详细的、基于证据的英文分析步骤列表。",
            },
        },
        "required": ["uuid", "component", "reason", "reasoning_trace"],
    }

    async def execute(
        self,
        uuid: str,
        component: str,
        reason: str,
        reasoning_trace: List[Dict],
        **kwargs,
    ) -> ToolResult:
        """
        接收并格式化最终报告。这个工具的主要作用是作为一个信号，让Agent知道分析已完成，并携带结构化的数据。
        """
        final_report_dict = {
            "uuid": uuid,
            "component": component,
            "reason": reason,
            "reasoning_trace": reasoning_trace,
        }
        # 工具的输出就是格式化后的JSON报告字符串
        return ToolResult(output=json.dumps(final_report_dict, ensure_ascii=False, indent=2)) 