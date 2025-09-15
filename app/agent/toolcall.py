import asyncio
import json
import re
import ast
from typing import Any, List, Optional, Union, Dict, Tuple

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice, Function
from app.tool import CreateChatCompletion, Terminate, ToolCollection
from app.tool.base import ToolResult

class ToolCallAgent(ReActAgent):
    """一个能够执行工具调用的基础代理类，具有增强的抽象能力"""

    name: str = "toolcall"
    description: str = "一个可以执行工具调用的代理。"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    async def think(self) -> bool:
        """
        核心思考过程。准备并发送请求给LLM以获取下一步行动。
        该方法现在只依赖于子类（如RCAAgent）提供的上下文和提示。
        """
        # 系统提示直接来自于 self.system_prompt，由RCAAgent提供
        system_prompts_for_llm = [Message.system_message(self.system_prompt)] if self.system_prompt else []

        try:
            # 使用 memory 中的消息，这些消息由 RCAAgent.step 动态构建
            response_message = await self.llm.ask_tool(
                messages=self.memory.messages,
                system_msgs=system_prompts_for_llm,
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except TokenLimitExceeded as e:
            logger.error(f"🚨 Token limit error: {e}")
            self.memory.add_message(Message.assistant_message(f"已达到最大Token限制，无法继续执行: {e}"))
            self.state = AgentState.FINISHED
            return False
        except Exception as e:
            logger.error(f"🚨 LLM call failed: {e}")
            self.memory.add_message(Message.assistant_message(f"LLM调用失败: {e}"))
            return False

        if not response_message:
            logger.warning("LLM did not return a response.")
            return False

        # V45: Overhaul parsing logic to strictly enforce the "thought -> tool_code" structure.
        # This prevents contaminated data from entering the memory when the LLM
        # produces malformed output (e.g., code before thought).
        thinking_content, tool_code_str = self._split_thought_and_code(response_message.content or "")
        
        # If the LLM returns code without structured tool_calls, parse it manually.
        if tool_code_str and not response_message.tool_calls:
            try:
                self.tool_calls = self._create_tool_calls_from_str(tool_code_str)
                if self.tool_calls:
                    logger.info(f"🛠️ {self.name} manually parsed {len(self.tool_calls)} tools from content.")
            except Exception as e:
                logger.error(f"Error manually parsing tool code: {e}")
                self.tool_calls = [] # Clear any partial results on error

        # Log the thinking part.
        if not thinking_content.strip() and self.tool_calls:
            thinking_content = "[Agent is acting without verbal thought...]"
        
        logger.info(f"✨ {self.name}'s thoughts: \n{thinking_content}")

        if self.tool_calls:
            logger.info(f"🛠️ {self.name} selected {len(self.tool_calls)} tools to use.")

        # Update the response message with the cleaned thinking content and tool calls.
        # This is crucial for maintaining a clean memory for subsequent steps.
        response_message.content = thinking_content
        response_message.tool_calls = self.tool_calls or [] # Ensure tool_calls is a list
        self.memory.add_message(response_message)
        
        return bool(self.tool_calls or thinking_content.strip())
        # --- END of REFACTOR ---

    async def act(self) -> str:
        """执行工具调用并处理其结果"""
        if not self.tool_calls:
            last_message = self.memory.messages[-1] if self.memory.messages else None
            if last_message:
                return last_message.content or ""
            return "没有工具可执行。"

        results = []
        for command in self.tool_calls:
            self._current_base64_image = None

            raw_result_obj = await self.execute_tool(command)
            
            # 提取真实的数据，而不是ToolResult包装器
            final_content_for_memory = raw_result_obj
            if isinstance(raw_result_obj, ToolResult):
                final_content_for_memory = raw_result_obj.output
                # 尝试自动解析JSON字符串为字典
                if isinstance(final_content_for_memory, str):
                    try:
                        parsed_json = json.loads(final_content_for_memory)
                        final_content_for_memory = parsed_json
                    except json.JSONDecodeError:
                        # 如果不是合法的JSON，则保留其原始字符串形式
                        pass

            # --- 优化日志记录 ---
            # 直接使用未处理的 final_content_for_memory 进行日志记录，以避免双重转义
            log_content = final_content_for_memory
            if isinstance(log_content, (dict, list)):
                # 如果是字典或列表，为了日志可读性，转为JSON字符串
                log_content = json.dumps(log_content, ensure_ascii=False)

            result_str = str(log_content)
            if self.max_observe and len(result_str) > self.max_observe:
                result_str = result_str[: self.max_observe] + "..."
            # --- 日志记录优化结束 ---

            # 如果是ask_data_expert，则不记录结果，避免重复
            if command.function.name != "ask_data_expert":
                logger.info(
                    f"🎯 Tool '{command.function.name}' completed its mission! Result: {result_str}"
                )

            tool_msg = Message.tool_message(
                content=final_content_for_memory,  # 传递解析后的原始结果
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result_str)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> Any:
        """Execute a single tool call and return its raw result."""
        if not command or not command.function or not command.function.name:
            error_msg = "Error: Invalid command format"
            logger.error(f"❌ {error_msg}")
            return error_msg

        name = command.function.name
        if name not in self.available_tools.tool_map:
            error_msg = f"Error: Unknown tool '{name}'"
            logger.error(f"❌ {error_msg}")
            return error_msg

        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")

            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}' with args: {args}")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            logger.info(f"✅ Tool '{name}' executed successfully.")

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image
                # If result is a simple object with just base64_image and maybe a message,
                # return the whole object so the image is not lost.
                if hasattr(result, "content"):
                    return result.content
                return result

            return result
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format. Arguments: '{command.function.arguments}'"
            logger.error(f"❌ Tool '{name}' failed. Error: {error_msg}")
            return ToolResult(error=error_msg)
        except Exception as e:
            error_msg = f"Tool '{name}' encountered a problem: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return ToolResult(error=error_msg)

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        try:
            return await super().run(request)
        finally:
            await self.cleanup()

    def _extract_thought(self, text: str) -> str:
        """从原始文本中提取思考过程，通过移除整个工具代码块。"""
        # 使用正则表达式安全地移除所有匹配的代码块
        # 这比字符串替换更稳健，特别是当tool_code为空或包含特殊字符时
        
        # 匹配 <tool_code>...</tool_code> 或 <tool_code>...</tool_call>
        pattern_tool_code = r"<tool_code>.*?</tool_code>|<tool_code>.*?</tool_call>"
        thought = re.sub(pattern_tool_code, "", text, flags=re.DOTALL)
        
        # 匹配 ```...```
        pattern_markdown = r"```(?:python)?.*?```"
        thought = re.sub(pattern_markdown, "", thought, flags=re.DOTALL)

        return thought.strip()

    def _parse_tool_code(self, text: str) -> str:
        """
        从模型的思考文本中提取并返回可执行的Python代码。
        此方法经过增强，可以处理多种格式，并忽略代码块前的任何无关前缀。
        
        支持的格式:
        1. <tool_code>...</tool_code>
        2. ```python...```
        3. 原始的 <|...|> 分隔符
        """
        # 增强的正则表达式，匹配可选的前缀、不同的代码块标记，并提取内容
        patterns = [
            # 匹配 <tool_code>...</tool_code> 或 <tool_code>...</tool_call>
            r"<tool_code>(.*?)(?:</tool_code>|</tool_call>)",
            # 匹配 ```python...``` 或 ```...```
            r"```(?:python)?(.*?)```",
            # 匹配原始的、混乱的 <|...|> 格式
            r"function<｜tool sep｜>(.*?)<｜tool call end｜>",
        ]

        all_code_blocks = []
        for pattern in patterns:
            # 使用 findall 捕获所有匹配项
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # 清理匹配结果，移除空字符串和纯空白字符串
                cleaned_matches = [m.strip() for m in matches if m and m.strip()]
                if cleaned_matches:
                    all_code_blocks.extend(cleaned_matches)

        # V43: 移除了不安全的后备逻辑，该逻辑在LLM未遵循格式时会导致错误。
        # 新的行为是：只返回在显式代码块标记中找到的代码。
        return "\n".join(all_code_blocks)

    def _split_thought_and_code(self, text: str) -> Tuple[str, str]:
        """
        Splits the raw LLM response into a thought part and a code part.
        It strictly assumes that thoughts come before the <tool_code> block.
        Any content after the first tool_code block is ignored.
        """
        tool_code_pattern = r"<tool_code>.*?</tool_code>"
        match = re.search(tool_code_pattern, text, re.DOTALL)

        if not match:
            # No tool_code found, the entire text is considered thinking.
            return text.strip(), ""

        # The thought is everything before the tool_code block.
        thought = text[:match.start()].strip()
        
        # The code is the content inside the tool_code block.
        code_content = self._parse_tool_code(match.group(0))

        return thought, code_content

    def _create_tool_calls_from_str(self, code_str: str) -> List[ToolCall]:
        """
        V21: Unified AST Parser.
        This parser handles both single and multi-line tool calls robustly by parsing the entire code block at once.
        It replaces the previous fragile combination of regex and line-by-line parsing.
        """
        tool_calls = []
        code_str = code_str.strip()
        if not code_str:
            return tool_calls

        # V-Final-Fix-2: First, try to parse the string as a JSON object, which is a common
        # failure mode for LLMs that are asked to produce a function call string but
        # instead produce the underlying API JSON for the tool.
        try:
            data = json.loads(code_str)
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                tool_calls.append(ToolCall(
                    id=f"call_manual_json_{len(tool_calls)}",
                    type="function",
                    function=Function(
                        name=data["name"],
                        arguments=json.dumps(data["arguments"], ensure_ascii=False)
                    ),
                ))
                return tool_calls
        except (json.JSONDecodeError, TypeError):
            # If it's not a valid JSON object, it's not a problem.
            # We'll proceed to the AST parser.
            pass

        # V44: Pre-process the code string to replace common non-ASCII punctuation
        # that LLMs might generate with their ASCII counterparts before parsing.
        code_str = code_str.replace('：', ':')

        try:
            # Parse the entire code block into an abstract syntax tree
            parsed_ast = ast.parse(code_str)
            
            # Iterate through the top-level nodes in the AST body
            for node in parsed_ast.body:
                # We are looking for function calls that are expressions
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    call_node = node.value
                    tool_name = getattr(call_node.func, 'id', '')
                    
                    if tool_name:
                        args_dict = {}
                        for kw in call_node.keywords:
                            # ast.literal_eval safely evaluates the node value
                            args_dict[kw.arg] = ast.literal_eval(kw.value)
                        
                        tool_calls.append(ToolCall(
                            id=f"call_manual_{len(tool_calls)}",
                            type="function",
                            function=Function(name=tool_name, arguments=json.dumps(args_dict, ensure_ascii=False)),
                        ))
        except (SyntaxError, ValueError) as e:
            logger.error(
                f"Failed to parse tool code block with AST. Error: {e}\n"
                f"Original code: '{code_str[:500]}...'"
            )
            # On failure, return an empty list to prevent crashes
            return []

        if not tool_calls:
            logger.warning(
                f"Unified AST parser could not extract any valid calls from the tool code. "
                f"Original code: '{code_str[:500]}...'"
            )
        return tool_calls

    def _extract_tool_calls_from_code_block(self, block_content: str, lang_type: str) -> List[Dict]:
        """
        V40: 重构 - 回归到统一的、基于AST的解析逻辑。
        此解析器现在可以同等地处理 ask_data_expert 和 final_report 的函数调用语法，
        确保了所有工具调用格式和解析逻辑的一致性。
        """
        tool_calls = []
        try:
            tree = ast.parse(block_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    tool_name = node.func.id
                    
                    # 允许解析 ask_data_expert 和 final_report
                    if tool_name not in ["ask_data_expert", "final_report"]:
                        continue

                    params = {}
                    for keyword in node.keywords:
                        # 使用更健壮的 ast.literal_eval 来解析参数值
                        params[keyword.arg] = ast.literal_eval(keyword.value)
                    
                    tool_calls.append({"tool_name": tool_name, "parameters": params})
            return tool_calls
        except (SyntaxError, ValueError) as e:
            self.logger.warning(f"无法将工具代码块解析为Python AST: {e}")
            return []
