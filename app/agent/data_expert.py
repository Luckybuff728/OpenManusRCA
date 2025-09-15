import asyncio
import json
import re

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.data_expert import SYSTEM_PROMPT
from app.schema import Message
from app.tool import ToolCollection
from app.tool.base import ToolResult
from app.tool.rca import (
    APMMetricsLoader,
    InfraMetricsLoader,
    LogsLoader,
    TracesLoader,
    TiDBMetricsLoader,
)


class DataExpertAgent(ToolCallAgent):
    """
    数据专家代理，负责将自然语言查询转换为对数据加载工具的调用，并执行它们。
    """

    name: str = "DataExpert"
    description: str = "一个专门用于将自然语言转换为数据查询的代理。"

    system_prompt: str = SYSTEM_PROMPT

    # 为数据专家配置专用的工具集
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            APMMetricsLoader(),
            InfraMetricsLoader(),
            LogsLoader(),
            TracesLoader(),
            TiDBMetricsLoader(),
        )
    )

    async def think(self) -> bool:
        """
        V21: 增强思考过程，在调用父类方法后，增加一个清理步骤。
        此步骤确保即使模型在 <tool_code> 块之外生成了多余的文本（如 "json"），
        也只保留工具调用代码，从而防止下游的AST解析失败。
        """
        # First, let the parent class handle the main logic of getting the LLM's thought
        # and extracting tool calls.
        if not await super().think():
            return False

        # Now, perform the cleanup on the raw response content if it exists.
        if self.memory.messages and self.memory.messages[-1].role == "assistant":
            last_message = self.memory.messages[-1]
            raw_content = last_message.content
            
            if raw_content and "<tool_code>" in raw_content:
                # Use regex to find all content within <tool_code>...</tool_code> blocks
                tool_code_blocks = re.findall(r'<tool_code>(.*?)</tool_code>', raw_content, re.DOTALL)
                
                if tool_code_blocks:
                    # Join all found tool code blocks. This handles multiple tool calls.
                    cleaned_content = "\n".join(tool_code_blocks)
                    
                    # Also reconstruct the full message with the <tool_code> tags,
                    # as the parent class might expect it.
                    last_message.content = f"<tool_code>{cleaned_content}</tool_code>"
                    
                    # Log the cleaning action for debugging purposes.
                    if raw_content != last_message.content:
                        logger.info(f"DataExpert response cleaned. Original: '{raw_content[:100]}...' | Cleaned: '{last_message.content[:100]}...'")

        return True

    async def act(self) -> str:
        """
        重写基础act方法以支持并行工具调用，并将所有结果聚合成一个JSON数组字符串。
        此版本包含了健壮的错误处理机制。
        """
        if not self.tool_calls:
            return '{"error": "No tools to call."}'

        # 并行执行所有工具调用，并捕获异常
        tasks = [self.execute_tool(command) for command in self.tool_calls]
        results_with_wrapper = await asyncio.gather(*tasks, return_exceptions=True)

        structured_results = []
        for i, raw_result_obj in enumerate(results_with_wrapper):
            command = self.tool_calls[i]
            final_content = None

            try:
                if isinstance(raw_result_obj, Exception):
                    # 情况1: asyncio.gather 捕获到了一个异常
                    final_content = {"error": f"Tool {command.function.name} failed during execution.", "details": str(raw_result_obj)}
                    logger.error(f"💥 Tool '{command.function.name}' raised an exception: {raw_result_obj}", exc_info=True)
                
                elif isinstance(raw_result_obj, ToolResult):
                    if raw_result_obj.error:
                        # 情况2: 工具执行成功，但返回了一个包含错误信息的 ToolResult
                        final_content = {"error": f"Tool {command.function.name} reported an error.", "details": raw_result_obj.error}
                    else:
                        # 情况3: 完全成功
                        final_content = raw_result_obj.output
                        if isinstance(final_content, str):
                            try:
                                final_content = json.loads(final_content)
                            except json.JSONDecodeError:
                                pass  # 如果不是有效的JSON，则保持为字符串
                else:
                    # 情况4: 未知的结果类型 (防御性编程)
                    final_content = {"error": "Unknown result type from tool execution.", "details": str(raw_result_obj)}

                # 为日志记录准备显示内容
                log_content_for_display = json.dumps(final_content, ensure_ascii=False) if isinstance(final_content, (dict, list)) else str(final_content)
                
                logger.info(f"🎯 Tool '{command.function.name}' completed! Result: {log_content_for_display[:]}")

            except Exception as e:
                # 捕获在上述结果处理过程中发生的任何意外错误
                logger.error(f"💥 Unexpected error while processing result for tool '{command.function.name}': {e}", exc_info=True)
                final_content = {"error": f"Failed to process result for {command.function.name}", "details": str(e)}

            # 为每个工具调用创建并添加消息到内存
            tool_msg = Message.tool_message(
                content=final_content,
                tool_call_id=command.id,
                name=command.function.name,
            )
            self.memory.add_message(tool_msg)
            structured_results.append({"tool_name": command.function.name, "result": final_content})

        # 返回包含所有结果的JSON数组字符串
        return json.dumps(structured_results, ensure_ascii=False, default=str) # 使用 default=str 以处理任何不可序列化的数据

    async def run(self, query: str) -> str:
        """
        运行DataExpertAgent，处理单个或多个工具调用。
        """
        # 1. 将查询添加到内存
        self.memory.add_user_message(query)

        # 2. 思考需要调用哪个或哪些工具
        if not await self.think():
            return '{"error": "思考阶段未能选择任何工具。"}'

        # 3. 检查是否有工具被选中
        if not self.tool_calls:
            return '{"error": "无法根据您的请求选择合适的数据查询工具。"}'

        # 4. 执行工具调用 (调用重写后的act方法)
        result = await self.act()

        # 5. 返回聚合后的JSON字符串结果
        # 不再对单个结果进行特殊处理，始终返回JSON数组字符串
        return result 