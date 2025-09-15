import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Set

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.prompt.rca import SYSTEM_PROMPT
from app.schema import AgentState, Message
from app.tool import ToolCollection
from app.tool.ask_data_expert import AskDataExpert
from app.tool.base import ToolResult



class RCAAgent(ToolCallAgent):
    """微服务故障根因分析专用代理"""

    name: str = "RCA"
    description: str = "一个专门用于微服务故障根因分析的代理"

    # 简化：移除 current_strategy 和 base_system_prompt，使用单一的 system_prompt
    system_prompt: str = Field(default=SYSTEM_PROMPT, description="The single, focused system prompt for the agent.")
    last_thought: str = Field(default="", description="上一步的思考过程，用于反思")


    # 增大观察和步骤限制，以处理更复杂的分析任务
    max_observe: int = 20000
    max_steps: int = 30

    # 故障案例信息
    case_uuid: Optional[str] = None
    case_query: Optional[str] = None
    analysis_start_time: Optional[str] = None
    analysis_end_time: Optional[str] = None

    # 专用工具集 - 替换为新的FinalReport工具
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            AskDataExpert(),
        )
    )

    # V-Final-Cleanup: 移除不再使用的 investigation_board
    # investigation_board: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # 已使用的工具和参数跟踪，避免重复分析
    used_tools: Dict[str, List[Dict]] = Field(default_factory=dict)

    # 当前步骤计数
    step_count: int = 0

    # 记录最后一次清理上下文的步骤数
    last_context_clean_step: int = 0

    @classmethod
    async def create(cls, uuid: str, query: str, **kwargs) -> "RCAAgent":
        """创建一个RCA代理实例"""
        instance = cls(**kwargs)
        instance.case_uuid = uuid
        instance.case_query = query

        # V3: 更稳健地从查询中提取时间范围
        # 支持 "from ... to ...", "时间范围: ... 到 ...", "during ... to ...", 以及 "between ... and ..."
        match = re.search(r"(?:from|时间范围:|during|between)\s*([\w\-\:\.T]+Z?)\s*(?:to|到|and)\s*([\w\-\:\.T]+Z?)", query, re.IGNORECASE)
        
        if match:
            instance.analysis_start_time = match.group(1)
            instance.analysis_end_time = match.group(2)
            logger.info(f"成功从查询中解析到时间范围: {instance.analysis_start_time} to {instance.analysis_end_time}")
        else:
            logger.warning(f"无法从查询字符串中解析时间范围: '{query}'")


        return instance

    async def run(self, prompt: str) -> str:
        self.state = AgentState.RUNNING
        self.memory.add_user_message(prompt)
        final_response = '{"error": "分析未在最大步骤内完成。"}'

        # 简化：只在开始时注入一次系统提示
        self.memory.add_system_message(self.system_prompt)

        while self.state == AgentState.RUNNING and self.step_count < self.max_steps:
            self.step_count += 1
            logger.info(f"--- 分析步骤 {self.step_count}/{self.max_steps} ---")

            # --- 步骤 1: 动态构建上下文感知的提示 ---
            case_info = f"""
## 案例信息
- 案例ID: {self.case_uuid}
- 分析时间范围: {self.analysis_start_time} 到 {self.analysis_end_time}
- 当前步骤: {self.step_count}/{self.max_steps}
            """
            # V-Final-Cleanup: 移除案件板和其摘要
            
            # 简化：移除动态策略，使用更直接的提示
            enhanced_prompt = f"""
            {case_info}

            ## 你的任务：严格执行下一步分析
            请严格遵循你在系统角色说明中被赋予的所有核心原则和诊断流程。现在，生成你的回复，**必须**先提供完整的、使用模板的“思考过程”，然后紧跟一个工具调用。
            """
            self.memory.add_user_message(enhanced_prompt)

            try:
                # --- 步骤 2: 思考 ---
                if not await self.think():
                    logger.warning("思考阶段未产生有效结果，终止分析。")
                    self.state = AgentState.FINISHED
                    # V-Final-Simplify: 检查最后的消息是否是最终报告
                    final_response = self._check_for_final_conclusion()
                    if final_response == '{"error": "思考阶段未选择任何工具，分析中止。"}':
                        logger.info("最后的消息不是最终结论。")
                    else:
                        logger.info("检测到最终结论，分析完成。")
                        self.state = AgentState.FINISHED
                        continue
                    
                    final_response = '{"error": "思考阶段未选择任何工具，分析中止。"}'
                    continue

                is_terminating = self._is_final_conclusion(self.last_thought)
                
                if is_terminating:
                    logger.info("检测到最终结论，分析完成。")
                    final_response = self.last_thought
                    self.state = AgentState.FINISHED
                    continue

                # --- 步骤 3: 行动 ---
                observations = await self.act()

                if is_terminating:
                    logger.info("检测到 final_report 工具调用，分析完成。")
                    final_response = observations
                    self.state = AgentState.FINISHED
                    continue

                # V-Final-Cleanup: 移除废弃的 reflect 方法

            except Exception as e:
                logger.error(f"执行步骤时出错: {str(e)}", exc_info=True)
                self.state = AgentState.ERROR
                final_response = f'{{"error": "执行步骤时出错: {str(e)}"}}'
                continue
        
        logger.info(f"分析完成，总步骤数: {self.step_count}")
        return final_response
    
    def _extract_final_json(self, content: str) -> Optional[str]:
        """
        Tries to extract a final conclusion JSON block from the raw LLM response.
        V-Final-Fix-4: More robust JSON extraction, handling markdown code blocks.
        It finds all potential JSON blocks and checks the last valid one.
        """
        pattern = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*?\})")
        
        last_valid_json = None
        for match in pattern.finditer(content):
            # One of the groups must have matched.
            json_str = match.group(1) or match.group(2)
            if json_str:
                if self._is_final_conclusion(json_str.strip()):
                    last_valid_json = json_str.strip()

        return last_valid_json

    def _is_final_conclusion(self, content: str) -> bool:
        """检查给定的内容是否是一个格式正确的最终结论JSON。"""
        content = content.strip()
        if not content.startswith("{") or not content.endswith("}"):
            return False
        
        try:
            data = json.loads(content)
            # 检查是否包含所有必需的键
            required_keys = {"uuid", "component", "reason", "reasoning_trace"}
            return required_keys.issubset(data.keys())
        except json.JSONDecodeError:
            return False

    def _check_for_final_conclusion(self) -> str:
        """在内存中检查最后一条消息是否是最终结论。"""
        if self.memory.messages:
            last_message_content = self.memory.messages[-1].content or ""
            if self._is_final_conclusion(last_message_content):
                return last_message_content
        return '{"error": "思考阶段未选择任何工具，分析中止。"}'

    # V-Final-Cleanup: 移除废弃的 reflect 和 _generate_investigation_summary 方法

    async def think(self) -> bool:
        """
        增强思考过程，增加带纠错提示的重试机制以应对模型随机性导致的意外输出和逻辑违规。
        """
        max_retries = 2
        for attempt in range(max_retries):
            # 调用父类方法获取思考结果，它会填充 self.tool_calls
            # 父类的 think 方法现在会处理思考内容的提取和日志记录
            if not await super().think():
                # 如果父类 think 返回 False，说明没有找到工具调用，直接进入重试逻辑
                self.memory.add_user_message("你上次没有选择任何工具。请严格遵循指示，在你的思考过程后，必须立即调用一个工具。")
                continue

            # 捕获上一步的思考过程，用于反思
            if self.memory.messages and self.memory.messages[-1].role == "assistant":
                self.last_thought = self.memory.messages[-1].content or ""

            # V-Final-Fix: Prioritize final conclusion JSON over any other output
            final_json_report = self._extract_final_json(self.last_thought)
            if final_json_report:
                logger.info("检测到最终结论 JSON。将忽略所有工具调用，并将此作为最终结果。")
                self.tool_calls = []
                self.last_thought = final_json_report
                
                # Clean up the message in memory as well
            if self.memory.messages and self.memory.messages[-1].role == "assistant":
                last_message = self.memory.messages[-1]
                last_message.content = self.last_thought
                last_message.tool_calls = []
                
                return True # This is a valid, final step

            # --- V18: 简化RCAAgent的think方法 ---
            # 思考内容的提取和日志记录已移至基类 ToolCallAgent
            # 此处只需专注于 RCA 相关的规则检查
            
            raw_response_content = self.memory.messages[-1].content or ""
            thought_content = raw_response_content # 基类已确保 content 是纯粹的思考文本

            # --- 规则检查 ---
            error_msg = None

            # 新增规则：如果调用 ask_data_expert，则必须有行动计划
            is_asking_data_expert = any(tc.function.name == "ask_data_expert" for tc in self.tool_calls)
            has_action_plan = "行动计划" in thought_content
            
            if is_asking_data_expert and not has_action_plan:
                error_msg = "你的思考过程不完整。当你调用 `ask_data_expert` 时，你必须在思考过程中明确提供一个以“行动计划:”开头的指令。请修正你的思考过程并重试。"

            # 规则 1: (已移除) 不再检查 final_report 工具调用
            
            # 规则 2: (已移除)

            # 规则 3: 如果根本没有选择工具怎么办？
            elif not self.tool_calls:
                # 检查这是否是一个最终结论
                if not self._is_final_conclusion(thought_content):
                    error_msg = "你上次没有选择任何工具。请严格遵循指示，在你的思考过程后，必须立即调用一个工具，或者直接输出最终结论的JSON。"

            # --- 如果没有错误，则成功 ---
            if error_msg is None:
                return True

            # --- 如果有错误，则执行重试逻辑 ---
            if error_msg:
                # 移除最后一条无效的助手消息
                if self.memory.messages and self.memory.messages[-1].role == "assistant":
                    self.memory.messages.pop()
                
                # 注入纠错信息
                self.memory.add_user_message(error_msg)
            
            # 清空无效的工具调用，准备重试
            self.tool_calls = []

        logger.error(f"RCA 思考步骤在 {max_retries} 次尝试后仍然失败。")
        return False

    async def act(self) -> str:
        """
        重写基础act方法以支持并行工具调用，并将所有结果聚合并更新到案件板。
        增加了对 final_report 的特殊处理，以确保返回有效的最终JSON。
        """
        if not self.tool_calls:
            last_message = self.memory.messages[-1].content
            return last_message if last_message else "没有工具被调用，分析中止。"

        # V-Final-Cleanup: 移除废弃的 final_report 工具处理逻辑

        # V-Plus: 强制从思考过程中提取行动计划，作为工具调用的唯一指令来源，防止LLM“言行不一”
        action_plan_query = None
        if self.last_thought:
            # V-Plus-Fix: 使用更健壮的正则表达式，以匹配以"-"或空格开头的列表项格式，并处理可选的"**"
            # V-Final-Fix: 增加 re.DOTALL 标志，确保可以捕获多行的行动计划。
            # V-Final-Fix-2: 使用非贪婪匹配 (.*?) 并匹配直到下一个代码块或字符串末尾，防止捕获多余的JSON。
            match = re.search(r"^\s*(?:-\s)?(?:\*\*)?行动计划(?:\*\*)?\s*[:：]\s*(.*?)(?=\n```|$)", self.last_thought, re.MULTILINE | re.DOTALL)
            if match:
                action_plan_query = match.group(1).strip()
                logger.info(f"从思考过程中提取到行动计划，将强制执行: '{action_plan_query}'")
            else:
                logger.warning("在思考过程中未找到明确的'行动计划'，将依赖LLM生成的工具调用。")

        # --- V3: 智能上下文注入 for ask_data_expert ---
        for command in self.tool_calls:
            if command.function.name == "ask_data_expert":
                try:
                    if action_plan_query:
                        # 如果成功提取到行动计划，则忽略LLM生成的query，强制使用计划内容
                        args = {"query": action_plan_query}
                    else:
                        # 否则，回退到使用LLM生成的参数
                        args = json.loads(command.function.arguments)
                    
                    # V4: 强制覆盖上下文参数，确保绝对一致性，杜绝模型幻觉。
                    # 无论LLM在query中生成了什么，都以Agent持有的原始案例信息为准。
                    args["uuid"] = self.case_uuid
                    args["start_time"] = self.analysis_start_time
                    args["end_time"] = self.analysis_end_time
                    
                    command.function.arguments = json.dumps(args, ensure_ascii=False)
                    logger.info(f"已为 ask_data_expert 强制覆盖上下文参数。最终参数: {command.function.arguments}")
                except json.JSONDecodeError:
                    logger.warning("无法解析 ask_data_expert 的参数以注入上下文，将按原样执行。")
        # --- 结束V3修改 ---


        # --- 原有的并行工具执行逻辑 ---
        tasks = [self.execute_tool(command) for command in self.tool_calls]
        results_with_wrapper: List[ToolResult] = await asyncio.gather(*tasks)

        all_observations = []

        for i, tool_result_obj in enumerate(results_with_wrapper):
            command = self.tool_calls[i]
            tool_name = command.function.name
            
            # --- V2 错误处理增强: 检查ToolResult本身的错误以及output内容 ---
            observation = ""
            is_error = False
            error_message = ""

            if tool_result_obj is None:
                is_error = True
                error_message = "工具执行没有返回任何结果。"
            elif tool_result_obj.error:
                is_error = True
                error_message = tool_result_obj.error
            else:
                observation = tool_result_obj.output
                # 尝试解析JSON，检查内部是否有错误
                try:
                    parsed_obs = json.loads(observation)
                    # DataExpert总是返回列表
                    if isinstance(parsed_obs, list) and parsed_obs:
                        # 检查列表内的每个字典
                        for item in parsed_obs:
                           if isinstance(item, dict) and "error" in item:
                                is_error = True
                                error_message = item.get("details", item["error"])
                                break # 找到第一个错误就跳出
                except (json.JSONDecodeError, TypeError):
                    # 不是JSON或格式不对，正常处理
                    pass
            
            if is_error:
                logger.error(f"工具 '{tool_name}' (并行分支) 返回了一个错误: {error_message}")
                all_observations.append(f"工具 '{tool_name}' 的执行失败: {error_message}")
                tool_msg = Message.tool_message(content=f'{{"error": "{error_message}"}}', tool_call_id=command.id, name=tool_name)
                self.memory.add_message(tool_msg)
                continue # 跳到下一个并行结果
            # --- 结束：错误处理逻辑 ---
            
            # V17: 更新观察结果处理以适应 DataExpert 的新输出格式
            all_observations.append(f"工具 '{tool_name}' 的观察结果:\n{observation}")
            
            tool_msg_content = observation
            try:
                # 尝试解析DataExpert返回的JSON字符串
                data_expert_result_list = json.loads(observation)
                if isinstance(data_expert_result_list, list):
                    # 从每个结果中提取真实的工具名和输出
                    formatted_observations = []
                    for res in data_expert_result_list:
                        if isinstance(res, dict) and "tool_name" in res and "result" in res:
                            actual_tool_name = res["tool_name"]
                            actual_result = res["result"]
                            # 将实际结果（通常是字典）格式化为字符串以便记录
                            actual_result_str = json.dumps(actual_result, ensure_ascii=False) if isinstance(actual_result, dict) else str(actual_result)
                            formatted_observations.append(f"工具 '{actual_tool_name}' 的观察结果:\n{actual_result_str}")
                    if formatted_observations:
                        # 用更详细、准确的观察结果替换通用的观察结果
                        tool_msg_content = "\n".join(formatted_observations)

            except (json.JSONDecodeError, TypeError):
                # 如果不是预期的格式，则保持原始观察结果不变
                pass

            tool_msg = Message.tool_message(content=tool_msg_content, tool_call_id=command.id, name=tool_name)
            self.memory.add_message(tool_msg)
            
            # V-Final-Cleanup: 移除案件板更新逻辑
        
        # 返回所有观察结果的合并字符串
        return "\n\n".join(all_observations)

    async def cleanup(self):
        """清理资源并生成最终报告"""
        logger.info(f"🧹 清理RCA代理资源")

        # 记录分析统计信息
        # V-Final-Cleanup: 移除案件板相关的统计
        stats = {
            "total_steps": self.step_count,
        }

        logger.info(f"📊 分析统计: {json.dumps(stats)}")

        # 调用父类的清理方法
        await super().cleanup()

    # V-Final-Cleanup: 移除未使用的 _create_reasoning_trace 方法
