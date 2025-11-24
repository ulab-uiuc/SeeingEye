import asyncio
import json
import re
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


def parse_tool_calls_multiple_formats(content: str) -> List[ToolCall]:
    """
    Try multiple parsing formats and use the first one that succeeds WITH VALID ARGUMENTS.
    Priority: 1) Standard XML format, 2) Repeated-tag XML format, 3) Loose JSON format
    """
    if not content or not isinstance(content, str):
        return []

    def has_valid_arguments(tool_calls: List[ToolCall]) -> bool:
        """Check if any tool call has non-empty arguments"""
        return any(tc.function.arguments and tc.function.arguments != "{}" for tc in tool_calls)

    # Priority 1: Try XML format with proper closing tags
    tool_calls = parse_xml_tool_calls_standard(content)
    if tool_calls and has_valid_arguments(tool_calls):
        return tool_calls

    # Priority 2: Try XML format with repeated opening tags
    tool_calls = parse_xml_tool_calls_repeated_tags(content)
    if tool_calls and has_valid_arguments(tool_calls):
        return tool_calls

    # Priority 3: Try loose JSON extraction
    tool_calls = parse_loose_json_tool_calls(content)
    if tool_calls and has_valid_arguments(tool_calls):
        return tool_calls

    # Fallback: If we found tool calls but none with valid arguments, return the first successful parse
    # This handles cases where empty arguments are actually valid (rare but possible)
    for _, parser_func in [
        ("standard XML", parse_xml_tool_calls_standard),
        ("repeated-tag XML", parse_xml_tool_calls_repeated_tags),
        ("loose JSON", parse_loose_json_tool_calls)
    ]:
        tool_calls = parser_func(content)
        if tool_calls:
            return tool_calls
    return []


def parse_xml_tool_calls_standard(content: str) -> List[ToolCall]:
    """
    Parse standard <tool_call>...</tool_call> XML format.
    """
    # Find all <tool_call>...</tool_call> blocks
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

    # Also look for incomplete tool calls (missing closing tag due to truncation)
    incomplete_pattern = r'<tool_call>\s*(.*?)$'
    if not matches:
        incomplete_matches = re.findall(incomplete_pattern, content, re.DOTALL | re.IGNORECASE)
        if incomplete_matches:
            logger.warning("Found incomplete tool call (possibly truncated)")
            matches = incomplete_matches

    return process_tool_call_matches(matches, "standard XML")


def parse_xml_tool_calls_repeated_tags(content: str) -> List[ToolCall]:
    """
    Parse <tool_call>...<tool_call> format (repeated opening tags).
    Extract JSON objects that appear between <tool_call> tags.
    """
    # Split by <tool_call> tags and extract JSON content
    segments = re.split(r'<tool_call>', content, flags=re.IGNORECASE)

    json_matches = []
    for segment in segments[1:]:  # Skip the first segment (before any <tool_call>)
        segment = segment.strip()
        if not segment:
            continue

        # Look for JSON object at the start of the segment
        # Find the complete JSON object using brace matching
        json_content = extract_first_complete_json(segment)
        if json_content:
            json_matches.append(json_content)

    return process_tool_call_matches(json_matches, "repeated-tag XML")


def extract_first_complete_json(text: str) -> str:
    """
    Extract the first complete JSON object from text by matching braces.
    """
    text = text.strip()
    if not text.startswith('{'):
        return ""

    brace_count = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON object
                    return text[:i+1]

    return ""  # Incomplete or malformed JSON


def parse_loose_json_tool_calls(content: str) -> List[ToolCall]:
    """
    Extract JSON objects that look like tool calls without XML tags.
    """
    # Look for JSON objects with "name" and "arguments" fields - improved pattern
    # This pattern handles nested braces better
    json_pattern = r'\{(?:[^{}]|{[^{}]*})*"name"(?:[^{}]|{[^{}]*})*"arguments"(?:[^{}]|{[^{}]*})*\}'
    matches = re.findall(json_pattern, content, re.DOTALL)

    # If that doesn't work, try a simpler pattern
    if not matches:
        simple_pattern = r'\{.*?"name".*?"arguments".*?\}'
        matches = re.findall(simple_pattern, content, re.DOTALL)

    return process_tool_call_matches(matches, "loose JSON")


def process_tool_call_matches(matches: List[str], _: str) -> List[ToolCall]:
    """
    Process matched strings into ToolCall objects.
    """
    tool_calls = []

    for i, match in enumerate(matches):
        try:
            match_content = match.strip()

            # Try to fix common JSON truncation issues
            if not match_content.endswith('}'):
                # Attempt to close incomplete JSON
                brace_count = match_content.count('{') - match_content.count('}')
                if brace_count > 0:
                    match_content += '}' * brace_count
                    logger.warning(f"Attempted to fix truncated JSON by adding {brace_count} closing braces")

            # Parse the JSON content
            tool_data = json.loads(match_content)

            if isinstance(tool_data, dict) and "name" in tool_data:
                # Import Function for proper schema
                from app.schema import Function

                # Extract arguments properly
                arguments = tool_data.get("arguments", {})

                # Create ToolCall object compatible with existing schema
                tool_call = ToolCall(
                    id=f"call_{i}",
                    function=Function(
                        name=tool_data["name"],
                        arguments=json.dumps(arguments)
                    )
                )

                tool_calls.append(tool_call)

        except json.JSONDecodeError:
            # Try to extract at least the function name for basic tool calling
            name_match = re.search(r'"name":\s*"([^"]+)"', match)
            if name_match:
                function_name = name_match.group(1)

                from app.schema import Function
                tool_call = ToolCall(
                    id=f"call_{i}",
                    function=Function(
                        name=function_name,
                        arguments="{}"  # Empty args as fallback
                    )
                )
                tool_calls.append(tool_call)
            continue
        except Exception:
            continue

    return tool_calls


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    _current_base64_image: Optional[str] = None
    _flow_iteration_context: Optional[str] = None
    in_tool_parsing_retry: bool = Field(default=False)

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    def set_flow_iteration_context(self, context: str) -> None:
        """Set the flow iteration context for enhanced logging"""
        self._flow_iteration_context = context

    def set_iteration_info(self, current_iteration: int, max_iterations: int) -> None:
        """Set current and max iteration information for prompt selection"""
        self.current_iteration = current_iteration
        self.max_iterations = max_iterations

    def set_tool_parsing_retry_mode(self, enabled: bool) -> None:
        """Set whether we're in tool parsing retry mode"""
        self.in_tool_parsing_retry = enabled

    def _build_step_prompt_context(self) -> str:
        """Build step context with appropriate prompt based on current execution state"""
        flow_context = ""
        if hasattr(self, '_flow_iteration_context'):
            flow_context = f" [Iteration {self._flow_iteration_context}]"

        # Choose appropriate prompt based on step and iteration conditions
        # Priority: final_iteration_prompt > final_step_prompt > first_step_prompt > next_step_prompt
        if (hasattr(self, 'final_iteration_prompt') and
            self.max_iterations and
            self.current_iteration >= self.max_iterations):
            # Final iteration takes highest priority
            prompt_to_use = getattr(self, 'final_iteration_prompt', None) or self.final_step_prompt or self.next_step_prompt
        elif self.current_step >= self.max_steps:
            # Final step (but not final iteration)
            prompt_to_use = self.final_step_prompt or self.next_step_prompt
        elif (self.current_step == 1 and self.current_iteration == 1 and
              hasattr(self, 'first_step_prompt') and self.first_step_prompt):
            # Very first step of first iteration
            prompt_to_use = self.first_step_prompt
        else:
            # Regular step
            prompt_to_use = self.next_step_prompt

        if self.current_step <= 1:
            step_context = f"{flow_context}:"
        else:
            step_context = f"STEP {self.current_step}{flow_context}: You are CONTINUING your own work from previous steps"

        # Combine step context with appropriate prompt
        if prompt_to_use:
            return f"{step_context}\n\n{prompt_to_use}"
        else:
            return step_context

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        # Skip adding step continuation context during tool parsing retries
        # to avoid conflicting messages with error feedback
        if not self.in_tool_parsing_retry:
            # Build step context with integrated prompt selection
            continuation_prompt = self._build_step_prompt_context()
            user_msg = Message.user_message(continuation_prompt)
            self.messages += [user_msg]

        try:
            # Get response with tool options
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        # Extract tool calls from response (standard OpenAI format)
        tool_calls = response.tool_calls if response and response.tool_calls else []
        content = response.content if response and response.content else ""

        # If no standard tool calls found, try multiple format parsers
        if not tool_calls and content:
            alternative_tool_calls = parse_tool_calls_multiple_formats(content)
            if alternative_tool_calls:
                tool_calls = alternative_tool_calls

        self.tool_calls = tool_calls

        # Get context for logging
        flow_context = ""
        if hasattr(self, '_flow_iteration_context'):
            flow_context = f" [Flow Iteration {self._flow_iteration_context}]"
        step_context = f" [Inner Step {self.current_step}]"

        # Store thinking summary for consolidated logging (reduce verbosity)
        self._current_thinking_summary = f"{content[:1000]}..." if len(content) > 1000 else content
        self._current_tool_summary = [call.function.name for call in tool_calls] if tool_calls else []

        # Log key information
        if tool_calls:
            logger.info(f"{flow_context}{step_context} Using tools: {[call.function.name for call in tool_calls]}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            self._current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            raw_arguments = command.function.arguments or "{}"
            args = json.loads(raw_arguments)

            # Get context for logging
            flow_context = ""
            if hasattr(self, '_flow_iteration_context'):
                flow_context = f" [Flow Iteration {self._flow_iteration_context}]"
            step_context = f" [Inner Step {self.current_step}]"

            # Execute the tool
            logger.info(f"{flow_context}{step_context} Executing tool: {name}")

            # Record tool execution start time
            import time
            start_time = time.time()

            result = await self.available_tools.execute(name=name, tool_input=args)

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Tool execution details are now included in the consolidated step log to reduce verbosity
            # Individual tool logs removed - information available in agent_step_complete metadata

            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image

            # Format result for display
            if name == "terminate_and_answer" and result:
                # For terminate_and_answer, return clean result without verbose wrapper
                observation = str(result)
            else:
                # Standard case with verbose execution details
                observation = (
                    f"Observed output of cmd `{name}` executed:\n{str(result)}"
                    if result
                    else f"Cmd `{name}` completed with no output"
                )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )

            # Log failed tool execution
            self._log_failed_tool_execution(name, command.function.arguments, error_msg, "json_decode_error")
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)

            # Log failed tool execution
            self._log_failed_tool_execution(name, command.function.arguments or "{}", error_msg, "execution_error")
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**_) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    def _log_thinking_process(self, content: str, tool_calls: List[ToolCall]) -> None:
        """Thinking process is now logged in consolidated step log to reduce verbosity."""
        # Thinking details are included in the agent_step_complete log entry
        # This method is kept for compatibility but no longer creates separate log entries
        pass

    def _log_failed_tool_execution(self, tool_name: str, arguments: str, error_msg: str, error_type: str) -> None:
        """Log failed tool execution to structured logging if available."""
        if hasattr(self, '_flow_context') and hasattr(self._flow_context, 'log_save'):
            log_save = self._flow_context.log_save
            iteration = getattr(self, '_flow_iteration_context', 1)
            if isinstance(iteration, str):
                iteration = int(iteration.split('/')[0]) if '/' in iteration else 1

            # Use individual question logging if available (new format)
            if hasattr(log_save, 'add_individual_question_message') and log_save.current_question:
                log_save.add_individual_question_message(
                    role=f"tool_{self.name}",
                    content=f"Tool '{tool_name}' failed: {error_msg}",
                    step_type="tool_execution_failed",
                    tool_calls=[{
                        "id": f"tool_failed_{self.current_step}",
                        "name": tool_name,
                        "arguments": arguments,
                        "error": error_msg,
                        "execution_status": "failed",
                        "error_type": error_type
                    }],
                    metadata={
                        "step_number": self.current_step,
                        "iteration": iteration,
                        "agent_state": self.state.value if self.state else "unknown",
                        "tool_name": tool_name,
                        "error_type": error_type,
                        "error_message": error_msg
                    }
                )
            # Use comprehensive logging if available (comprehensive format)
            elif hasattr(log_save, 'add_question_message') and log_save.current_question:
                log_save.add_question_message(
                    role=f"tool_{self.name}",
                    content=f"Tool '{tool_name}' failed: {error_msg}",
                    step_type="tool_execution_failed",
                    tool_calls=[{
                        "id": f"tool_failed_{self.current_step}",
                        "name": tool_name,
                        "arguments": arguments,
                        "error": error_msg,
                        "execution_status": "failed",
                        "error_type": error_type
                    }],
                    metadata={
                        "step_number": self.current_step,
                        "iteration": iteration,
                        "agent_state": self.state.value if self.state else "unknown",
                        "tool_name": tool_name,
                        "error_type": error_type,
                        "error_message": error_msg
                    }
                )
            else:
                # Fallback to old system - log as custom step
                log_save.log_custom_step(
                    content=f"Tool '{tool_name}' failed: {error_msg}",
                    role=self.name,
                    step_type="tool_execution_failed",
                    iteration=iteration,
                    metadata={
                        "step_number": self.current_step,
                        "tool_name": tool_name,
                        "error_type": error_type,
                        "error_message": error_msg,
                        "arguments": arguments
                    }
                )

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        try:
            return await super().run(request)
        finally:
            await self.cleanup()
