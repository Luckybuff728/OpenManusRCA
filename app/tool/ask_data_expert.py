import json
from typing import Any, Dict

from app.agent.data_expert import DataExpertAgent
from app.tool.base import BaseTool, ToolResult


class AskDataExpert(BaseTool):
    """
    一个特殊的工具，用于向DataExpertAgent提出数据查询请求。
    """

    name: str = "ask_data_expert"
    description: str = "当你需要查询指标、日志或追踪数据时，使用此工具向数据专家提问。"

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "uuid": {"type": "string", "description": "当前故障案例的唯一标识符。"},
            "start_time": {
                "type": "string",
                "description": "查询的开始时间，UTC格式 (例如 '2025-06-05T16:00:00Z')。",
            },
            "end_time": {
                "type": "string",
                "description": "查询的结束时间，UTC格式 (例如 '2025-06-05T18:00:00Z')。",
            },
            "query": {
                "type": "string",
                "description": "用自然语言清晰地描述你想要查询的数据。例如：'检查服务 adservice 的错误率和延迟' 或 '获取 pod adservice-0 的CPU和内存指标'。",
            },
        },
        "required": ["uuid", "start_time", "end_time", "query"],
    }

    async def execute(
        self,
        uuid: str,
        start_time: str,
        end_time: str,
        query: str,
        **kwargs,
    ) -> ToolResult:
        """
        执行向DataExpertAgent的查询。
        """
        try:
            # 1. 创建DataExpertAgent实例
            data_expert = DataExpertAgent()

            # 2. **关键修复**: 将必要的上下文参数强制注入到查询的开头，
            #    以确保DataExpertAgent在生成工具调用时始终包含它们。
            injected_query = (
                f"使用以下上下文参数：uuid='{uuid}', start_time='{start_time}', end_time='{end_time}'。"
                f"基于此上下文，为以下请求生成工具调用：'{query}'"
            )

            # 3. 构建传递给DataExpertAgent的结构化输入
            expert_prompt_dict = {
                "uuid": uuid,
                "start_time": start_time,
                "end_time": end_time,
                "query": injected_query, # 使用注入了上下文的查询
            }
            expert_prompt_str = json.dumps(expert_prompt_dict)

            # 4. 运行DataExpertAgent并获取其返回的原始数据
            result_str = await data_expert.run(expert_prompt_str)

            # 5. 将DataExpertAgent的结果直接包装成ToolResult返回
            return ToolResult(output=result_str)

        except Exception as e:
            # 在执行过程中捕获任何异常
            return ToolResult(error=f"调用数据专家时出错: {str(e)}") 