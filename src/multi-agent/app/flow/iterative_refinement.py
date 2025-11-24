import base64
import asyncio
from typing import Dict, List, Optional, Union, Tuple, Any

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.logger import logger
from app.utils.log_save import LogSave
from app.config import config


class IterativeRefinementFlow(BaseFlow):
    """
    A flow that manages iterative refinement between translator and reasoning agents.
    
    The flow orchestrates:
    1. Translator generates initial SIR from image
    2. Reasoning agent analyzes SIR and either provides answer or feedback
    3. If feedback provided, translator refines SIR based on feedback
    4. Repeat until final answer or max iterations
    """
    
    max_iterations: int = Field(default_factory=lambda: config.flow_config.max_iterations)
    current_iteration: int = Field(default=0)
    base64_image: Optional[str] = Field(default=None)
    image_path: Optional[str] = Field(default=None)
    current_sir: Optional[str] = Field(default=None)
    previous_sir: Optional[str] = Field(default=None)  # SIR from previous iteration
    log_save: LogSave = Field(default_factory=LogSave)

    def __init__(
        self,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **data
    ):
        super().__init__(agents, **data)

        # Validate required agents
        if "translator" not in self.agents:
            raise ValueError("IterativeRefinementFlow requires a 'translator' agent")
        if "reasoning" not in self.agents:
            raise ValueError("IterativeRefinementFlow requires a 'reasoning' agent")

    @property
    def translator_agent(self) -> BaseAgent:
        """Get the translator agent"""
        return self.agents["translator"]

    @property
    def reasoning_agent(self) -> BaseAgent:
        """Get the reasoning agent"""
        return self.agents["reasoning"]

    def _reset_agent_memory_for_iteration(self, agent: BaseAgent, iteration: int) -> None:
        """Reset agent memory but preserve SIR from previous iteration for context."""
        # Clear current memory
        agent.memory.clear()

        # For iterations > 1, provide access to previous SIR for context
        if iteration > 1 and self.previous_sir:
            if agent == self.translator_agent:
                context_msg = f"Your previous visual description (iteration {iteration-1}): {self.previous_sir}\n\nUse this as reference to improve your current description."
            else:  # reasoning agent
                context_msg = f"Previous visual description (iteration {iteration-1}): {self.previous_sir}\n\nThis shows the previous attempt at visual analysis. You can reference this for continuity."

            # Add system message directly to avoid base64_image parameter issue
            from app.schema import Message
            system_msg = Message.system_message(context_msg)
            agent.memory.add_message(system_msg)

    def _update_sir_history(self, new_sir: str) -> None:
        """Update SIR history - store current as previous, set new as current."""
        # Store current SIR as previous
        if self.current_sir:
            self.previous_sir = self.current_sir

        # Set new SIR as current
        self.current_sir = new_sir
        logger.info(f"📝 SIR updated")

    def _append_feedback_to_sir(self, feedback: str) -> None:
        """Append new feedback directly to current SIR for persistent context."""
        if not feedback or not feedback.strip():
            logger.warning("⚠️ Attempted to append empty feedback to SIR")
            return

        if not self.current_sir:
            logger.warning("⚠️ No current SIR to append feedback to")
            return

        # Clean feedback text
        clean_feedback = feedback.strip()
        if clean_feedback.startswith("FEEDBACK"):
            clean_feedback = clean_feedback.replace("FEEDBACK from reasoning agent:", "").strip()

        # Append feedback to existing SIR
        self.current_sir = f"""{self.current_sir}

--- REASONING FEEDBACK ---
{clean_feedback}"""

        logger.info(f"💬 Appended feedback to SIR (new length: {len(self.current_sir)} characters)")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert an image file to base64 encoded string."""
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string

    def _parse_input(self, input_text: str, image_path: Optional[str] = None) -> Tuple[str, List[str], str]:
        """Parse input text to extract question, options, and image path."""
        lines = input_text.strip().split('\n')

        # Find question and options by splitting on "Options:"
        input_parts = input_text.split("Options:")
        if len(input_parts) >= 2:
            question = input_parts[0].strip()
            options_and_image = input_parts[1].strip()

            # Extract all option lines, not just the first one
            import re
            options_lines = []
            for line in options_and_image.split('\n'):
                line = line.strip()
                if line and not line.startswith("image_path:"):
                    # Check if this line looks like an option (starts with letter followed by period or parenthesis)
                    if re.match(r'^[A-J]\.', line) or re.match(r'^\([A-J]\)', line) or not options_lines:
                        options_lines.append(line)
                else:
                    # Stop at image_path or empty lines after we've found options
                    if options_lines:
                        break

            # Join all option lines and parse them
            options_text = '\n'.join(options_lines)
            options = self._parse_multiline_options(options_text)
        else:
            question = lines[0]
            options = []

        # Extract image path from input or use provided parameter
        if image_path is None:
            for line in lines:
                if line.startswith("image_path:"):
                    image_path = line.split(":", 1)[1].strip()
                    break

        if not image_path:
            raise ValueError("No image path provided")

        return question, options, image_path

    def _parse_options(self, options_line: str) -> List[str]:
        """Parse options line handling various formats."""
        if not options_line or options_line.lower() in ['none', 'n/a', '']:
            return []

        try:
            if options_line.startswith('[') and options_line.endswith(']'):
                import ast
                return ast.literal_eval(options_line)
            else:
                # Smart parsing for financial amounts
                import re
                option_pattern = r',\s+(?=\$?[A-Za-z0-9])'
                parts = re.split(option_pattern, options_line)
                options = [opt.strip() for opt in parts if opt.strip()]

                if not options and options_line.strip():
                    options = [opt.strip() for opt in options_line.split(',') if opt.strip()]
                return options
        except:
            return [opt.strip() for opt in options_line.split(',') if opt.strip()]

    def _parse_multiline_options(self, options_text: str) -> List[str]:
        """Parse multiline options text, preserving the full formatted options."""
        if not options_text or options_text.lower() in ['none', 'n/a', '']:
            return []

        options = []
        import re

        # Split by newlines and process each line
        for line in options_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check if line has a letter prefix (A., B., (A), (B), etc.)
            if re.match(r'^(?:\([A-J]\)|\b[A-J]\.)', line):
                # Keep the full formatted option with letter label
                options.append(line)
            elif line:
                # If no letter prefix found, assume it's a continuation or standalone option
                options.append(line)

        return options

    def _setup_image(self, image_path: str) -> None:
        """Load and encode image for processing."""
        self.image_path = image_path
        self.base64_image = self._encode_image_to_base64(image_path)
        logger.info(f"Image loaded: {len(self.base64_image)} characters")
        logger.info(f"Image path stored: {self.image_path}")

    def _setup_task_exception_handling(self) -> None:
        """Set up global task exception handling to prevent unhandled task warnings."""
        def task_exception_handler(task):
            """Handle exceptions from background tasks that might not be awaited."""
            try:
                # Try to get the exception
                exception = task.exception()
                if exception:
                    # Check if it's the common event loop closure issue
                    if isinstance(exception, RuntimeError) and "Event loop is closed" in str(exception):
                        logger.debug(f"Background task failed due to event loop closure - this is expected during shutdown")
                    else:
                        logger.warning(f"Background task failed with exception: {exception}")
            except Exception:
                # If we can't get the exception, just ignore
                pass

        # Set up exception handling for all current tasks
        try:
            loop = asyncio.get_event_loop()
            current_tasks = asyncio.all_tasks(loop)
            for task in current_tasks:
                if not task.done():
                    task.add_done_callback(task_exception_handler)
        except Exception as e:
            logger.debug(f"Could not set up task exception handling: {e}")

    async def execute(self, input_text: str, image_path: Optional[str] = None, log_save: Optional[LogSave] = None) -> str:
        """
        Execute the iterative refinement flow.

        Two-loop architecture:
        - Outer loop: Iterations between translator and reasoning (max 3)
        - Inner loop: Each agent's multi-step execution (max 3 steps per agent)

        Returns:
            Final answer from the reasoning agent
        """
        try:
            # Set up task exception handling to prevent unhandled task warnings
            self._setup_task_exception_handling()

            # Use external log_save if provided, otherwise use the instance one
            if log_save is not None:
                self.log_save = log_save

            # Parse and setup
            question, options, image_path = self._parse_input(input_text, image_path)
            self._setup_image(image_path)

            logger.info(f"Starting iterative refinement flow (max {self.max_iterations} iterations)")
            logger.info(f"Question: {question}")
            logger.info(f"Options: {options}")

            # Individual question logging is managed externally by the adapter
            session_id = None

            # Collect and display experiment configuration
            experiment_config = self._collect_experiment_config()
            self._display_experiment_config(experiment_config)

            # Outer loop: Iterative refinement between agents
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"=== Outer Loop Iteration {iteration}/{self.max_iterations} ===")

                # CRITICAL: Isolate each iteration to prevent bug propagation
                try:
                    # Reset iteration state to ensure clean slate
                    await self._prepare_iteration_isolation(iteration)

                    # Log iteration start if available
                    if hasattr(self.log_save, 'log_iteration_start'):
                        self.log_save.log_iteration_start(iteration)

                    # Reset memory for new iteration but preserve previous SIR context
                    self._reset_agent_memory_for_iteration(self.translator_agent, iteration)
                    self._reset_agent_memory_for_iteration(self.reasoning_agent, iteration)

                    # Set flow context and iteration info for agents
                    self.translator_agent.set_flow_iteration_context(f"{iteration}/{self.max_iterations}")
                    self.reasoning_agent.set_flow_iteration_context(f"{iteration}/{self.max_iterations}")
                    self.translator_agent.set_iteration_info(iteration, self.max_iterations)
                    self.reasoning_agent.set_iteration_info(iteration, self.max_iterations)

                    # Inner loop 1: Translator generates/refines SIR
                    sir_result = await self._run_translator_inner_loop(question, options, iteration)
                    if sir_result.startswith("Error:"):
                        return self._finalize_with_error(session_id, sir_result)

                    # Update SIR history before setting new current
                    self._update_sir_history(sir_result)
                    self._validate_sir(sir_result, iteration)

                    # Inner loop 2: Reasoning analyzes SIR and decides next action
                    reasoning_result, decision = await self._run_reasoning_inner_loop(question, options, iteration)
                    if reasoning_result.startswith("Error:"):
                        return self._finalize_with_error(session_id, reasoning_result)

                    # Note: Reasoning step logging is now handled by _log_agent_step with actual agent input
                    # The old log_reasoner_step created artificial input and is replaced by accurate logging

                    # Handle reasoning decision
                    if decision == "FINAL_ANSWER":
                        logger.info(f"Final answer obtained in {iteration} iteration(s)")
                        return self._finalize_with_success(session_id, reasoning_result)
                    elif decision == "CONTINUE_WITH_FEEDBACK" and iteration < self.max_iterations:
                        # Append feedback directly to current SIR for persistent context
                        self._append_feedback_to_sir(reasoning_result)

                        logger.info(f"💬 Feedback received and appended to SIR, proceeding to iteration {iteration + 1}")
                        continue
                    else:
                        logger.warning(f"Max iterations reached or unclear response")
                        return self._finalize_with_success(session_id, reasoning_result)

                except Exception as iteration_error:
                    # Isolate iteration errors - don't let them affect subsequent iterations
                    clean_error = self._clean_error_message(str(iteration_error))
                    logger.error(f"💥 Iteration {iteration} failed with error: {clean_error}")

                    # If this is the last iteration or a critical error, fail completely
                    if iteration == self.max_iterations or self._is_critical_error(iteration_error):
                        logger.error(f"Critical error in iteration {iteration} or max iterations reached - failing completely")
                        return self._finalize_with_error(session_id, f"Iteration {iteration} failed: {clean_error}")

                    # For non-critical errors, try to recover and continue to next iteration
                    logger.warning(f"🔄 Attempting recovery - will try iteration {iteration + 1}")
                    await self._recover_from_iteration_failure(iteration, iteration_error)
                    continue

            # Max iterations reached
            final_result = "Error: Maximum iterations reached without final answer"
            return self._finalize_with_error(session_id, final_result)

        except Exception as e:
            # Clean error message to avoid printing base64 data
            error_msg = str(e)

            # Provide more specific error identification for common issues
            if "APIConnectionError" in error_msg:
                logger.error("API Connection Error in IterativeRefinementFlow - possible network/server issue")
                return "Flow execution failed: API Connection Error (network/server issue)"
            elif "RetryError" in error_msg:
                logger.error("Retry Error in IterativeRefinementFlow - API request failed after multiple attempts")
                return "Flow execution failed: API request failed after retries"
            else:
                clean_error = self._clean_error_message(error_msg)
                logger.error(f"Error in IterativeRefinementFlow: {clean_error}")
                return f"Flow execution failed: {clean_error}"

        finally:
            # CRITICAL: Ensure proper cleanup to prevent event loop closure issues
            try:
                await self._ensure_final_cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Final cleanup had minor issues: {cleanup_error}")
                # Don't propagate cleanup errors as they shouldn't affect the main result

    async def _ensure_final_cleanup(self) -> None:
        """Ensure all async resources are properly cleaned up before flow completion."""
        logger.debug("🧹 Performing final cleanup to prevent event loop issues")

        try:
            # Close HTTP clients properly
            await self._safe_close_http_clients()

            # Clean up agent resources
            for agent in [self.translator_agent, self.reasoning_agent]:
                if hasattr(agent, 'cleanup') and callable(agent.cleanup):
                    try:
                        await agent.cleanup()
                    except Exception as e:
                        logger.debug(f"Agent {agent.name} cleanup had minor issues: {e}")

            logger.debug("✅ Final cleanup completed successfully")

        except Exception as e:
            logger.debug(f"Final cleanup had issues: {e}")
            # Don't propagate cleanup errors

    async def _prepare_iteration_isolation(self, iteration: int) -> None:
        """
        Prepare clean state for iteration to prevent bug propagation.
        This ensures each iteration starts with a clean slate.
        """
        logger.info(f"🧹 Preparing iteration {iteration} isolation")

        try:
            # 1. Reset agent states to clean values
            await self._reset_agent_states()

            # 2. Clean up any corrupted connection pools
            await self._cleanup_connection_pools()

            # 3. Reset tool states
            await self._reset_tool_states()

            # 4. Clear any transient error flags
            self._clear_error_flags()

            logger.info(f"✅ Iteration {iteration} isolation prepared successfully")

        except Exception as e:
            logger.warning(f"⚠️ Iteration isolation preparation had minor issues: {e}")
            # Don't fail the iteration for isolation issues - just log and continue

    async def _reset_agent_states(self) -> None:
        """Reset both agents to clean, ready states."""
        # Reset agent states
        from app.schema import AgentState

        # Reset translator agent
        self.translator_agent.state = AgentState.IDLE
        self.translator_agent.current_step = 0
        self.translator_agent.tool_calls = []
        if hasattr(self.translator_agent, 'final_caption_json'):
            self.translator_agent.final_caption_json = None

        # Reset reasoning agent
        self.reasoning_agent.state = AgentState.IDLE
        self.reasoning_agent.current_step = 0
        self.reasoning_agent.tool_calls = []

        # Clear any retry flags
        if hasattr(self.translator_agent, 'in_tool_parsing_retry'):
            self.translator_agent.in_tool_parsing_retry = False
        if hasattr(self.reasoning_agent, 'in_tool_parsing_retry'):
            self.reasoning_agent.in_tool_parsing_retry = False

    async def _cleanup_connection_pools(self) -> None:
        """Clean up potentially corrupted HTTP connection pools with proper async cleanup."""
        try:
            # Properly close HTTP clients to prevent event loop closure issues
            await self._safe_close_http_clients()

            # Force LLM connection reset if available
            if hasattr(self.translator_agent, 'llm') and self.translator_agent.llm:
                if hasattr(self.translator_agent.llm, '_reset_connection_pool'):
                    await self.translator_agent.llm._reset_connection_pool()

            if hasattr(self.reasoning_agent, 'llm') and self.reasoning_agent.llm:
                if hasattr(self.reasoning_agent.llm, '_reset_connection_pool'):
                    await self.reasoning_agent.llm._reset_connection_pool()

        except Exception as e:
            logger.debug(f"Connection pool cleanup had minor issues: {e}")

    async def _safe_close_http_clients(self) -> None:
        """Safely close HTTP clients to prevent event loop closure race conditions."""
        import asyncio
        close_tasks = []

        try:
            # Find and close all HTTPX AsyncClient instances
            for agent in [self.translator_agent, self.reasoning_agent]:
                if hasattr(agent, 'llm') and agent.llm:
                    # Check for OpenAI client (which uses httpx internally)
                    if hasattr(agent.llm, '_client') and agent.llm._client:
                        client = agent.llm._client
                        # Check if it's an httpx AsyncClient or has aclose method
                        if hasattr(client, 'aclose'):
                            task = asyncio.create_task(self._safe_aclose(client, f"{agent.name}_llm_client"))
                            close_tasks.append(task)

            # Wait for all cleanup tasks with timeout to prevent hanging
            if close_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*close_tasks, return_exceptions=True),
                    timeout=5.0  # 5 second timeout for cleanup
                )

        except asyncio.TimeoutError:
            logger.warning("HTTP client cleanup timed out - some connections may not be properly closed")
        except Exception as e:
            logger.debug(f"HTTP client cleanup had issues: {e}")

    async def _safe_aclose(self, client, client_name: str) -> None:
        """Safely close an async client with proper error handling."""
        try:
            logger.debug(f"Closing {client_name}")
            await client.aclose()
            logger.debug(f"Successfully closed {client_name}")
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                logger.debug(f"Event loop already closed for {client_name} - ignoring")
            else:
                logger.warning(f"Runtime error closing {client_name}: {e}")
        except Exception as e:
            logger.warning(f"Error closing {client_name}: {e}")

    async def _reset_tool_states(self) -> None:
        """Reset tool states to prevent state contamination."""
        try:
            # Reset tool states for translator
            if hasattr(self.translator_agent, 'available_tools'):
                for tool_name, tool in self.translator_agent.available_tools.tool_map.items():
                    if hasattr(tool, 'reset_state'):
                        await tool.reset_state()

            # Reset tool states for reasoning agent
            if hasattr(self.reasoning_agent, 'available_tools'):
                for tool_name, tool in self.reasoning_agent.available_tools.tool_map.items():
                    if hasattr(tool, 'reset_state'):
                        await tool.reset_state()

        except Exception as e:
            logger.debug(f"Tool state reset had minor issues: {e}")

    def _clear_error_flags(self) -> None:
        """Clear transient error flags and state."""
        # Clear any error tracking flags
        if hasattr(self, '_iteration_error_count'):
            self._iteration_error_count = 0
        if hasattr(self, '_last_error_type'):
            self._last_error_type = None

    def _is_critical_error(self, error: Exception) -> bool:
        """
        Determine if an error is critical enough to stop all iterations.
        Non-critical errors allow recovery and continuation.
        """
        error_str = str(error).lower()

        # Critical errors that should stop all iterations
        critical_patterns = [
            'permission denied',
            'authentication failed',
            'authorization failed',
            'file not found',
            'module not found',
            'import error',
            'syntax error',
            'memory error',
            'disk full',
            'no space left'
        ]

        # Non-critical errors that allow recovery
        recoverable_patterns = [
            'connection error',
            'timeout',
            'api error',
            'retry error',
            'network error',
            'service unavailable',
            'rate limit',
            'token limit'
        ]

        # Check for critical errors first
        for pattern in critical_patterns:
            if pattern in error_str:
                return True

        # Check for recoverable errors
        for pattern in recoverable_patterns:
            if pattern in error_str:
                return False

        # Default: treat unknown errors as critical to be safe
        return True

    async def _recover_from_iteration_failure(self, failed_iteration: int, error: Exception) -> None:
        """
        Attempt recovery from iteration failure.
        This method implements strategies to recover from common error types.
        """
        error_str = str(error).lower()
        logger.info(f"🔧 Attempting recovery from iteration {failed_iteration} failure")

        try:
            # Strategy 1: Connection errors - wait and reset connections
            if any(pattern in error_str for pattern in ['connection error', 'network error', 'api error']):
                logger.info("🔗 Connection error detected - resetting connections")
                await self._cleanup_connection_pools()
                # Small delay to allow network recovery
                import asyncio
                await asyncio.sleep(1.0)

            # Strategy 2: Memory/token limit errors - clear memory aggressively
            elif any(pattern in error_str for pattern in ['token limit', 'memory error']):
                logger.info("💾 Memory/token error detected - aggressive memory cleanup")
                # Clear non-essential memory
                self.translator_agent.memory.messages = []
                self.reasoning_agent.memory.messages = []
                # Reset SIR history but keep the latest
                if hasattr(self, 'sir_history') and len(self.sir_history) > 1:
                    self.sir_history = [self.sir_history[-1]]  # Keep only the latest

            # Strategy 3: Tool/parsing errors - reset tool states
            elif any(pattern in error_str for pattern in ['tool', 'parse', 'json']):
                logger.info("🔧 Tool/parsing error detected - resetting tool states")
                await self._reset_tool_states()

            # Strategy 4: Generic recovery - full state reset
            else:
                logger.info("🧹 Generic error - performing full state cleanup")
                await self._reset_agent_states()
                await self._cleanup_connection_pools()
                await self._reset_tool_states()

            logger.info(f"✅ Recovery attempt for iteration {failed_iteration} completed")

        except Exception as recovery_error:
            logger.warning(f"⚠️ Recovery attempt had issues: {recovery_error}")
            # Don't fail on recovery issues - just continue

    async def _run_agent_with_retry(self, agent, agent_name: str, max_retries: int = 2):
        """
        Run agent with automatic retry for truncated tool calls.

        Args:
            agent: The agent instance to run
            agent_name: Name of the agent for logging
            max_retries: Maximum number of retry attempts

        Returns:
            Result from successful agent execution
        """
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"🔄 {agent_name} execution attempt {attempt + 1}/{max_retries + 1}")
                result = await agent.run()

                # Check if tool calls were parsed successfully with valid arguments
                if self._has_invalid_tool_calls(result):
                    if attempt < max_retries:
                        logger.warning(f"⚠️ Detected invalid/truncated tool calls in {agent_name}, retrying...")

                        # Clean recent failed attempt from memory to prevent pollution
                        self._clean_failed_attempt_from_memory(agent)

                        # Provide concise, specific feedback
                        feedback = (
                            f"🔧 RETRY {attempt + 1}: Previous tool call was incomplete. "
                            f"Please provide a complete tool call with valid JSON arguments."
                        )
                        agent.update_memory("user", feedback)

                        # Set retry flag to prevent conflicting step continuation messages
                        if hasattr(agent, 'set_tool_parsing_retry_mode'):
                            agent.set_tool_parsing_retry_mode(True)

                        continue
                    else:
                        logger.warning(f"⚠️ {agent_name} still showing invalid tool calls after {max_retries} retries")

                # Reset retry flag after successful execution
                if hasattr(agent, 'set_tool_parsing_retry_mode'):
                    agent.set_tool_parsing_retry_mode(False)

                return result

            except Exception as e:
                error_str = str(e).lower()

                # Check for tool argument errors (likely due to truncation)
                if any(pattern in error_str for pattern in [
                    "missing.*required.*argument",
                    "missing 1 required positional argument",
                    "unexpected keyword argument",
                    "arguments.*missing",
                    "invalid arguments"
                ]) and attempt < max_retries:

                    logger.warning(f"⚠️ Tool argument error in {agent_name} (attempt {attempt + 1}): {e}")

                    # Clean recent failed attempt from memory to prevent pollution
                    self._clean_failed_attempt_from_memory(agent)

                    # Provide concise, specific feedback about the tool error
                    feedback = (
                        f"🔧 RETRY {attempt + 1}: Tool error - {str(e)[:100]}... "
                        f"Please provide complete tool call with all required arguments."
                    )
                    agent.update_memory("user", feedback)

                    # Set retry flag to prevent conflicting step continuation messages
                    if hasattr(agent, 'set_tool_parsing_retry_mode'):
                        agent.set_tool_parsing_retry_mode(True)

                    continue

                # Reset retry flag and re-raise non-recoverable errors
                if hasattr(agent, 'set_tool_parsing_retry_mode'):
                    agent.set_tool_parsing_retry_mode(False)
                raise e

        # If we get here, all retries failed - ensure retry flag is reset
        if hasattr(agent, 'set_tool_parsing_retry_mode'):
            agent.set_tool_parsing_retry_mode(False)

        logger.error(f"❌ {agent_name} failed after {max_retries + 1} attempts")
        raise Exception(f"{agent_name} execution failed after {max_retries + 1} attempts")


    def _setup_agent_memory(self, agent, question: str, options: Optional[List[str]], iteration: int) -> str:
        """Setup agent memory with appropriate inputs based on agent type and iteration

        Returns:
            str: The actual input text that was given to the agent for logging purposes
        """
        agent_type = agent.name.lower()

        if agent_type == "translator":
            if iteration == 1:
                # First iteration: initial request with image
                options_text = f"\n\nOptions: {', '.join(options)}" if options else ""
                translator_input = f"You are a visual captioner. Your task is to generate a descriptive caption for the image that describes what you see visually. Question: {question}{options_text}\n\nImage file: {self.image_path}"
                agent.update_memory("user", translator_input, base64_image=self.base64_image)
                logger.info("Added initial request to translator memory")
                return translator_input
            else:
                # Subsequent iterations: merge context and SIR improvement into single message
                options_text = f"\n\nOptions: {', '.join(options)}" if options else ""

                if hasattr(self, 'current_sir') and self.current_sir:
                    # Sync the translator's SIR with the flow's SIR
                    if hasattr(agent, 'current_sir'):
                        agent.current_sir = self.current_sir
                        logger.info(f"🔄 Synced translator SIR with flow SIR ({len(self.current_sir)} chars)")

                    # Merge base context with SIR improvement task
                    merged_translator_input = f"""You are refining your visual description based on reasoning feedback. Question: {question}{options_text}
Image file: {self.image_path}

Your current SIR with reasoning feedback (iteration {iteration-1}):
{self.current_sir}

IMPROVEMENT TASK:
1. Analyze the reasoning feedback carefully to understand what visual details are needed
2. Look at the image again with this feedback in mind
3. UPDATE your current SIR to address the feedback - don't start fresh
4. Focus on visual details that help answer the question: {question}
5. Maintain objectivity - describe what you see, don't infer answers

Remember: Update your existing SIR incrementally, don't recreate it from scratch."""
                    agent.update_memory("user", merged_translator_input, base64_image=self.base64_image)
                    logger.info(f"Added merged SIR improvement task to translator memory (iteration {iteration}, SIR length: {len(self.current_sir)} chars)")
                    return merged_translator_input
                else:
                    # Fallback for when no previous SIR exists
                    base_context = f"You are refining your visual description based on reasoning feedback. Question: {question}{options_text}\n\nImage file: {self.image_path}"
                    agent.update_memory("user", base_context, base64_image=self.base64_image)
                    logger.info(f"Added context to translator memory (iteration {iteration}) - no previous SIR")
                    return base_context

        elif agent_type == "text_only_reasoning":
            # Merge question, options, and SIR into single reasoning message
            if hasattr(self, 'current_sir') and self.current_sir:
                # Merge all reasoning inputs into single cohesive message
                options_text = f"\n\nOptions: {', '.join(options)}" if options else ""
                merged_reasoning_input = f"""Question: {question}{options_text}

Visual analysis from translator (iteration {iteration}):
{self.current_sir}

REASONING TASK:
1. Analyze this visual description (SIR) provided by the translator
2. If you need more specific visual details to answer confidently, provide feedback for the translator"""
                agent.update_memory("user", merged_reasoning_input)
                logger.info(f"Added merged SIR reasoning task to reasoning agent memory (iteration {iteration}, SIR length: {len(self.current_sir)} chars)")
                return merged_reasoning_input
            else:
                # Fallback when no SIR exists - just question and options
                context_input = f"Question: {question}"
                if options:
                    context_input += f"\n\nOptions: {', '.join(options)}"
                agent.update_memory("user", context_input)
                logger.warning(f"No SIR available for reasoning agent in iteration {iteration}")
                return context_input

        # Default fallback
        return f"Question: {question}"

    def _log_system_prompt(self, agent_name: str, system_prompt: str, iteration: int) -> None:
        """Log agent's system prompt"""
        if hasattr(self.log_save, 'add_individual_question_message') and self.log_save.current_question:
            self.log_save.add_individual_question_message(
                role="system",
                content=system_prompt,
                metadata={
                    "agent": agent_name.lower(),
                    "iteration": iteration,
                    "prompt_type": "system_prompt"
                }
            )

    def _log_user_prompt(self, agent_name: str, user_input: str, iteration: int, prompt_type: str = "user_input") -> None:
        """Log user prompt/question to agent"""
        if hasattr(self.log_save, 'add_individual_question_message') and self.log_save.current_question:
            self.log_save.add_individual_question_message(
                role="user",
                content=user_input,
                metadata={
                    "agent": agent_name.lower(),
                    "iteration": iteration,
                    "prompt_type": prompt_type
                }
            )

    def _log_agent_response(self, agent_name: str, agent_response: str, iteration: int, tool_calls: Optional[List] = None) -> None:
        """Log agent's thinking + action response"""
        if hasattr(self.log_save, 'add_individual_question_message') and self.log_save.current_question:
            self.log_save.add_individual_question_message(
                role=agent_name.lower(),
                content=agent_response,
                tool_calls=tool_calls,
                metadata={
                    "iteration": iteration,
                    "agent_type": agent_name.lower(),
                    "response_length": len(agent_response),
                    "has_tool_calls": bool(tool_calls)
                }
            )

    def _log_tool_execution(self, tool_name: str, tool_input: str, tool_output: str, iteration: int) -> None:
        """Log tool execution and results"""
        if hasattr(self.log_save, 'add_individual_question_message') and self.log_save.current_question:
            self.log_save.add_individual_question_message(
                role="tool",
                content=f"Tool: {tool_name}\nInput: {tool_input}\nOutput: {tool_output}",
                metadata={
                    "tool_name": tool_name,
                    "iteration": iteration,
                    "input_length": len(tool_input),
                    "output_length": len(tool_output)
                }
            )

    def _clean_failed_attempt_from_memory(self, agent) -> None:
        """Clean recent failed assistant responses from agent memory to prevent pollution during retries"""
        try:
            messages = agent.memory.messages
            if len(messages) < 2:
                return

            # Remove the last assistant message if it contained failed tool calls
            # This prevents accumulation of failed attempts in memory
            if messages[-1].role == "assistant":
                logger.debug(f"Removing failed assistant message from {agent.name} memory")
                messages.pop()

            # Also remove any trailing tool messages from the failed attempt
            while messages and messages[-1].role == "tool":
                logger.debug(f"Removing failed tool message from {agent.name} memory")
                messages.pop()

        except Exception as e:
            logger.warning(f"Failed to clean memory for {agent.name}: {e}")
            # Don't let memory cleaning failures break the retry mechanism

    def _has_invalid_tool_calls(self, result: str) -> bool:
        """
        Check if the result contains invalid or incomplete tool calls using existing parsing logic.

        Args:
            result: The agent's response to check

        Returns:
            True if tool calls are invalid or incomplete
        """
        # Import the existing tool parsing function
        from app.agent.toolcall import parse_tool_calls_multiple_formats

        # If the response doesn't contain any tool call patterns, it's fine
        if not any(pattern in result for pattern in ['<tool_call>', 'function', 'python_execute']):
            return False

        try:
            # Use the existing parser to extract tool calls
            tool_calls = parse_tool_calls_multiple_formats(result)

            # If parsing found no tool calls but response seems to contain tool patterns, it's likely truncated
            if not tool_calls:
                # Check if response contains partial tool call indicators
                tool_indicators = ['<tool_call>', 'function', '"name":', 'python_execute']
                if any(indicator in result for indicator in tool_indicators):
                    logger.warning("Tool call patterns found but parsing failed - likely truncated")
                    return True
                return False

            # Check if any tool call has empty or invalid arguments
            for tool_call in tool_calls:
                if not hasattr(tool_call, 'function') or not tool_call.function:
                    return True

                # Check if arguments are empty or just "{}"
                args = tool_call.function.arguments
                if not args or args.strip() in ['{}', '""', "''", '']:
                    logger.warning(f"Tool call '{tool_call.function.name}' has empty arguments")
                    return True

            return False

        except Exception as e:
            logger.warning(f"Error parsing tool calls: {e}")
            # If parsing failed with an exception, consider it invalid
            return True

    async def _run_translator_inner_loop(self, question: str, options: List[str], iteration: int) -> str:
        """Run translator agent's inner loop (up to max_steps)."""
        try:
            translator = self.translator_agent

            logger.info(f"🌐 [Flow Iteration {iteration}/{self.max_iterations}] Starting TRANSLATOR inner loop")

            # Set flow context for structured logging
            translator._flow_context = self
            translator.set_flow_iteration_context(f"{iteration}/{self.max_iterations}")
            translator.set_iteration_info(iteration, self.max_iterations)

            # Reset step counter for new iteration
            translator.current_step = 0

            # Setup agent memory using modular helper and capture actual input
            actual_translator_input = self._setup_agent_memory(translator, question, options, iteration)

            # Log SIR preview for translator iterations > 1
            if iteration > 1 and hasattr(self, 'current_sir') and self.current_sir:
                sir_preview = self.current_sir[:300] + "..." if len(self.current_sir) > 300 else self.current_sir
                logger.info(f"📄 Previous SIR Preview for Translator: {sir_preview}")

            logger.info(f"Starting translator inner loop (iteration {iteration})")
            if options:
                logger.info(f"Multiple-choice question with {len(options)} options")
            else:
                logger.info("Open-ended question (no options provided)")

            # Run translator inner loop with retry logic for truncated tool calls
            # Agent will see full conversation history in memory
            raw_result = await self._run_agent_with_retry(translator, "Translator")

            # Extract result - prioritize tool-stored JSON
            result = self._extract_translator_result(translator, raw_result)

            # Sync flow's SIR with translator's evolving SIR
            if hasattr(translator, 'current_sir') and translator.current_sir:
                self.current_sir = translator.current_sir
                logger.info(f"📝 Flow synced with translator SIR ({len(self.current_sir)} chars)")
            else:
                # Fallback: use extracted result as SIR
                self.current_sir = result
                logger.info(f"📝 Flow using extracted result as SIR ({len(result)} chars)")

            # Log translator execution with improved structure following ReAct pattern
            if iteration == 1 and hasattr(translator, 'system_prompt') and translator.system_prompt:
                self._log_system_prompt("Translator", translator.system_prompt, iteration)
            self._log_user_prompt("Translator", actual_translator_input, iteration)
            self._log_agent_response("Translator", result, iteration)

            return result

        except Exception as e:
            # Clean error message to avoid printing base64 data
            error_msg = str(e)

            # Provide more specific error identification for common issues
            if "APIConnectionError" in error_msg:
                logger.error("API Connection Error in translator inner loop - possible network/server issue")
                return "Error: Translator inner loop failed: API Connection Error (network/server issue)"
            elif "RetryError" in error_msg:
                logger.error("Retry Error in translator inner loop - API request failed after multiple attempts")
                return "Error: Translator inner loop failed: API request failed after retries"
            elif "TokenLimitExceeded" in error_msg:
                logger.error("Token Limit Error in translator inner loop")
                return "Error: Translator inner loop failed: Token limit exceeded"
            else:
                # Generic error but clean the message
                clean_error = self._clean_error_message(error_msg)
                logger.error(f"Error in translator inner loop: {clean_error}")
                return f"Error: Translator inner loop failed: {clean_error}"

    def _extract_translator_result(self, translator, raw_result: str) -> str:
        """Extract result from translator, prioritizing tool-stored JSON."""
        if hasattr(translator, 'final_caption_json') and translator.final_caption_json:
            logger.info("Using stored JSON caption from translator tool")
            return translator.final_caption_json
        else:
            logger.info("Using raw step output")
            return raw_result

    async def _run_reasoning_inner_loop(self, question: str, options: List[str], iteration: int) -> Tuple[str, str]:
        """Run reasoning agent's inner loop and determine decision.

        Returns:
            Tuple of (result, decision) where decision is one of:
            - FINAL_ANSWER: Agent provided final answer
            - CONTINUE_WITH_FEEDBACK: Agent wants more info from translator
            - MAX_ITERATIONS_OR_UNCLEAR: Unclear response or max iterations
        """
        try:
            reasoning = self.reasoning_agent

            logger.info(f"🧠 [Flow Iteration {iteration}/{self.max_iterations}] Starting REASONING inner loop")

            # Set flow context for structured logging
            reasoning._flow_context = self
            reasoning.set_flow_iteration_context(f"{iteration}/{self.max_iterations}")
            reasoning.set_iteration_info(iteration, self.max_iterations)

            # Reset step counter for new iteration
            reasoning.current_step = 0

            # Setup reasoning agent memory using modular helper and capture actual input
            actual_reasoning_input = self._setup_agent_memory(reasoning, question, options, iteration)

            logger.info(f"✅ Added SIR to reasoning agent memory (iteration {iteration}, length: {len(self.current_sir)} chars)")

            # Log preview of SIR content for debugging
            if hasattr(self, 'current_sir') and self.current_sir:
                sir_preview = self.current_sir[:300] + "..." if len(self.current_sir) > 300 else self.current_sir
                logger.info(f"📄 SIR Preview for Reasoning: {sir_preview}")

            logger.info("Starting reasoning inner loop")

            # Run reasoning inner loop with retry logic for truncated tool calls
            # Agent will see: initial question + all previous SIRs + previous reasoning attempts
            result = await self._run_agent_with_retry(reasoning, "Reasoning")

            # Determine decision based on result
            decision = self._determine_reasoning_decision(result)

            # Log reasoning execution with improved structure following ReAct pattern
            if iteration == 1 and hasattr(reasoning, 'system_prompt') and reasoning.system_prompt:
                self._log_system_prompt("Reasoning", reasoning.system_prompt, iteration)
            self._log_user_prompt("Reasoning", actual_reasoning_input, iteration)
            self._log_agent_response("Reasoning", result, iteration)

            return result, decision

        except Exception as e:
            # Clean error message to avoid printing base64 data
            error_msg = str(e)

            # Provide more specific error identification for common issues
            if "APIConnectionError" in error_msg:
                logger.error("API Connection Error in reasoning inner loop - possible network/server issue")
                return "Error: Reasoning inner loop failed: API Connection Error (network/server issue)", "MAX_ITERATIONS_OR_UNCLEAR"
            elif "RetryError" in error_msg:
                logger.error("Retry Error in reasoning inner loop - API request failed after multiple attempts")
                return "Error: Reasoning inner loop failed: API request failed after retries", "MAX_ITERATIONS_OR_UNCLEAR"
            elif "TokenLimitExceeded" in error_msg:
                logger.error("Token Limit Error in reasoning inner loop")
                return "Error: Reasoning inner loop failed: Token limit exceeded", "MAX_ITERATIONS_OR_UNCLEAR"
            else:
                # Generic error but clean the message
                clean_error = self._clean_error_message(error_msg)
                logger.error(f"Error in reasoning inner loop: {clean_error}")
                return f"Error: Reasoning inner loop failed: {clean_error}", "MAX_ITERATIONS_OR_UNCLEAR"

    def _determine_reasoning_decision(self, result: str) -> str:
        """Determine the decision from reasoning agent result."""
        if self._is_final_answer(result):
            return "FINAL_ANSWER"
        elif self._is_feedback(result):
            return "CONTINUE_WITH_FEEDBACK"
        else:
            return "MAX_ITERATIONS_OR_UNCLEAR"


    def _validate_sir(self, sir_result: str, iteration: int) -> None:
        """Validate SIR format and log accordingly."""
        try:
            import json
            if sir_result.startswith('{') and sir_result.endswith('}'):
                parsed_json = json.loads(sir_result)
                if 'global_caption' in parsed_json:
                    logger.info(f"Generated valid JSON SIR (iteration {iteration})")
                else:
                    logger.warning(f"SIR missing 'global_caption' field (iteration {iteration})")
            else:
                logger.info(f"Generated text-based SIR (iteration {iteration})")
        except json.JSONDecodeError:
            logger.warning(f"SIR is not valid JSON format (iteration {iteration})")

    def _is_final_answer(self, response: str) -> bool:
        """Check if the response contains a final answer."""
        response_lower = response.lower()
        return (
            response.strip().upper().startswith("FINAL ANSWER:") or
            "terminate_and_answer" in response_lower or
            "has been completed successfully" in response_lower
        )

    def _is_feedback(self, response: str) -> bool:
        """Check if the response contains feedback for refinement."""
        response_lower = response.lower()
        return (
            response.strip().upper().startswith("FEEDBACK:") or
            "terminate_and_ask_translator" in response_lower or
            "missing" in response_lower or
            "need" in response_lower or
            "unclear" in response_lower or
            "more information" in response_lower
        )

    def _finalize_with_success(self, session_id: Optional[str], result: str) -> str:
        """Finalize logging session with successful result."""
        return result

    def _finalize_with_error(self, session_id: Optional[str], error_result: str) -> str:
        """Finalize logging session with error result."""
        return error_result

    def _collect_experiment_config(self) -> Dict[str, Any]:
        """Collect experiment configuration from agents and flow settings."""
        try:
            # Get model configurations from agents
            translator_config = {}
            reasoning_config = {}

            if hasattr(self.translator_agent, 'llm'):
                llm = self.translator_agent.llm
                translator_config = {
                    "model": getattr(llm, 'model', 'Unknown'),
                    "api_type": getattr(llm, 'api_type', 'Unknown'),
                    "temperature": getattr(llm, 'temperature', 'Unknown'),
                    "max_tokens": getattr(llm, 'max_tokens', 'Unknown'),
                    "base_url": getattr(llm, 'base_url', 'N/A'),
                }
                # Add vLLM specific parameters if available
                if hasattr(llm, 'provider'):
                    provider = llm.provider
                    if hasattr(provider, 'tensor_parallel_size'):
                        translator_config["tensor_parallel_size"] = provider.tensor_parallel_size
                    if hasattr(provider, 'gpu_memory_utilization'):
                        translator_config["gpu_memory_utilization"] = provider.gpu_memory_utilization
                    if hasattr(provider, 'model_name'):
                        translator_config["provider_model"] = provider.model_name

            if hasattr(self.reasoning_agent, 'llm'):
                llm = self.reasoning_agent.llm
                reasoning_config = {
                    "model": getattr(llm, 'model', 'Unknown'),
                    "api_type": getattr(llm, 'api_type', 'Unknown'),
                    "temperature": getattr(llm, 'temperature', 'Unknown'),
                    "max_tokens": getattr(llm, 'max_tokens', 'Unknown'),
                    "base_url": getattr(llm, 'base_url', 'N/A'),
                }
                # Add vLLM specific parameters if available
                if hasattr(llm, 'provider'):
                    provider = llm.provider
                    if hasattr(provider, 'tensor_parallel_size'):
                        reasoning_config["tensor_parallel_size"] = provider.tensor_parallel_size
                    if hasattr(provider, 'gpu_memory_utilization'):
                        reasoning_config["gpu_memory_utilization"] = provider.gpu_memory_utilization
                    if hasattr(provider, 'model_name'):
                        reasoning_config["provider_model"] = provider.model_name

            # Flow configuration
            flow_config = {
                "max_iterations": self.max_iterations,
                "flow_type": "iterative_refinement"
            }

            # System information
            import torch
            system_config = {
                "gpu_available": torch.cuda.is_available() if torch is not None else False,
                "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
            }

            return {
                "models": {
                    "translator": translator_config,
                    "reasoning": reasoning_config
                },
                "flow": flow_config,
                "system": system_config
            }

        except Exception as e:
            logger.warning(f"Failed to collect experiment config: {e}")
            return {
                "models": {"translator": {}, "reasoning": {}},
                "flow": {"max_iterations": self.max_iterations, "flow_type": "iterative_refinement"},
                "system": {"gpu_available": "Unknown", "gpu_memory": "Unknown"}
            }

    def _clean_error_message(self, error_msg: str) -> str:
        """Clean error message to remove base64 image data and other sensitive content."""
        # Remove base64 image data
        clean_msg = error_msg.replace('"base64_image":', '"base64_image": "[EXCLUDED]",')

        # Also handle cases where base64 strings might appear
        import re
        # Remove long base64-like strings (30+ characters of base64 pattern)
        clean_msg = re.sub(r'"[A-Za-z0-9+/]{30,}={0,2}"', '"[BASE64_DATA_EXCLUDED]"', clean_msg)

        # Truncate extremely long error messages
        if len(clean_msg) > 1000:
            clean_msg = clean_msg[:1000] + "... [TRUNCATED]"

        return clean_msg

    def _display_experiment_config(self, experiment_config: Dict[str, Any]):
        """Display experiment configuration in terminal."""
        logger.info("🔧 EXPERIMENT CONFIGURATION")
        logger.info("=" * 50)

        # Models configuration
        models = experiment_config.get("models", {})
        if models:
            logger.info("📋 MODELS:")
            for agent_name, model_config in models.items():
                if model_config:
                    logger.info(f"  • {agent_name.upper()}:")
                    logger.info(f"    - Model: {model_config.get('model', 'N/A')}")
                    logger.info(f"    - API Type: {model_config.get('api_type', 'N/A')}")
                    logger.info(f"    - Temperature: {model_config.get('temperature', 'N/A')}")
                    logger.info(f"    - Max Tokens: {model_config.get('max_tokens', 'N/A')}")
                    logger.info(f"    - Base URL: {model_config.get('base_url', 'N/A')}")

                    # vLLM specific parameters
                    if model_config.get('api_type') == 'vllm':
                        logger.info(f"    - Provider Model: {model_config.get('provider_model', 'N/A')}")
                        logger.info(f"    - GPU Memory: {model_config.get('gpu_memory_utilization', 'N/A')}")
                        logger.info(f"    - Tensor Parallel: {model_config.get('tensor_parallel_size', 'N/A')}")

        # Flow configuration
        flow_config = experiment_config.get("flow", {})
        if flow_config:
            logger.info("")
            logger.info("⚙️ FLOW SETTINGS:")
            logger.info(f"  • Max Iterations: {flow_config.get('max_iterations', 'N/A')}")
            logger.info(f"  • Flow Type: {flow_config.get('flow_type', 'N/A')}")

        # System information
        system_config = experiment_config.get("system", {})
        if system_config:
            logger.info("")
            logger.info("💻 SYSTEM INFO:")
            logger.info(f"  • GPU Available: {system_config.get('gpu_available', 'N/A')}")
            logger.info(f"  • GPU Memory: {system_config.get('gpu_memory', 'N/A')}")

        logger.info("=" * 50)