from abc import ABC, abstractmethod
from typing import Optional, Dict

from pydantic import Field

from app.agent.base import BaseAgent
from app.llm import LLM
from app.schema import AgentState, Memory


class ReActAgent(BaseAgent, ABC):
    name: str
    description: Optional[str] = None

    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None

    llm: Optional[LLM] = Field(default_factory=LLM)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE

    max_steps: int = 10
    current_step: int = 0

    @abstractmethod
    async def think(self) -> bool:
        """Process current state and decide next action"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""

    async def step(self) -> str:
        """Execute a single step: think and act with detailed phase-by-phase logging."""
        # Track step execution state to prevent duplicate logging
        step_started = False

        try:
            # Phase 1: Log step start with prompt and initial memory state
            self._log_step_start()
            step_started = True

            # Phase 2: Think - get LLM response and plan actions
            should_act = await self.think()
            self._log_thinking_phase(should_act)

            if not should_act:
                result = "Thinking complete - no action needed"
                self._log_step_complete(result)
                return result

            # Phase 3: Act - execute tool calls and collect results
            result = await self.act()
            self._log_action_phase(result)

            # Phase 4: Log step completion with final memory state
            self._log_step_complete(result)

            return result

        except Exception as e:
            # If step failed after logging start, log failure
            if step_started:
                self._log_step_failure(str(e))
            raise e

    async def run(self, request: Optional[str] = None) -> str:
        """Override run method to ensure step counter is reset on retry."""
        # Reset step counter at the start of each run
        # This prevents step counter increment issues when agent is retried
        self.current_step = 0
        return await super().run(request)


    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics from the agent's LLM"""
        if hasattr(self, 'llm') and hasattr(self.llm, 'token_tracker'):
            return self.llm.token_tracker.get_usage_summary()
        return {}

    def _log_prompts_for_step(self) -> None:
        """Log system prompt and next step prompt at appropriate points in execution sequence."""
        if not (hasattr(self, '_flow_context') and hasattr(self._flow_context, 'log_save')):
            return

        log_save = self._flow_context.log_save
        if not (hasattr(log_save, 'add_individual_question_message') and log_save.current_question):
            return

        # Log system prompt on first step only
        if self.current_step == 1:  # Step 1
            self._log_system_prompt(log_save)
            # Store full prompts in session metadata for reference
            self._store_prompts_in_session_metadata(log_save)

        # Log next step prompt for step 2 and beyond
        if self.current_step >= 2:  # Step 2, 3, etc.
            self._log_next_step_prompt(log_save)

    def _log_system_prompt(self, log_save) -> None:
        """Log the actual system prompt content."""
        if hasattr(self, 'system_prompt') and self.system_prompt:
            log_save.add_individual_question_message(
                role="system",
                content=self.system_prompt,  # Log actual system prompt content
                metadata={
                    "agent_name": self.name,
                    "prompt_type": "system_prompt",
                    "step_number": self.current_step,
                    "prompt_length": len(self.system_prompt)
                }
            )

    def _log_next_step_prompt(self, log_save) -> None:
        """Log the actual next step prompt content."""
        # Determine which prompt is actually being used based on current step and max steps
        prompt_to_log = None
        prompt_type = None

        if hasattr(self, '_build_step_prompt_context'):
            # This agent uses the new conditional prompt system
            prompt_context = self._build_step_prompt_context()
            prompt_to_log = prompt_context
            prompt_type = "step_prompt_context"
        elif self.current_step >= self.max_steps and hasattr(self, 'final_step_prompt') and self.final_step_prompt:
            # Final step
            prompt_to_log = self.final_step_prompt
            prompt_type = "final_step_prompt"
        elif hasattr(self, 'next_step_prompt') and self.next_step_prompt:
            # Regular step
            prompt_to_log = self.next_step_prompt
            prompt_type = "next_step_prompt"

        if prompt_to_log:
            log_save.add_individual_question_message(
                role="system",
                content=prompt_to_log,  # Log actual prompt content
                                metadata={
                    "agent_name": self.name,
                    "prompt_type": prompt_type,
                    "step_number": self.current_step,
                    "max_steps": self.max_steps,
                    "prompt_length": len(prompt_to_log),
                    "is_final_step": self.current_step >= self.max_steps
                }
            )

    def _store_prompts_in_session_metadata(self, log_save) -> None:
        """Store full prompt content in session metadata for reference."""
        if not log_save.current_session:
            return

        # Initialize agent_prompts in experiment_config if not exists
        if 'agent_prompts' not in log_save.current_session.experiment_config:
            log_save.current_session.experiment_config['agent_prompts'] = {}

        # Store full prompts for this agent
        agent_prompts = {
            'system_prompt': getattr(self, 'system_prompt', ''),
            'next_step_prompt': getattr(self, 'next_step_prompt', ''),
            'agent_class': self.__class__.__name__,
            'max_steps': getattr(self, 'max_steps', 0)
        }

        log_save.current_session.experiment_config['agent_prompts'][self.name] = agent_prompts

    def _log_step_start(self) -> None:
        """Log step initiation with prompt and initial memory state."""
        if not (hasattr(self, '_flow_context') and hasattr(self._flow_context, 'log_save')):
            return

        log_save = self._flow_context.log_save
        if not (hasattr(log_save, 'add_individual_question_message') and log_save.current_question):
            return

        # Log system prompt on first step only
        if self.current_step == 1:
            self._log_system_prompt(log_save)
            self._store_prompts_in_session_metadata(log_save)
        # Log next step prompt for step 2 and beyond
        elif self.current_step >= 2:
            self._log_next_step_prompt(log_save)

        # Log step initialization
        iteration = getattr(self, '_flow_iteration_context', 1)
        step_num = self.current_step  # Current step is already correctly numbered

        initial_memory = []
        for msg in self.memory.messages:
            msg_dict = msg.to_dict()
            if 'base64_image' in msg_dict:
                msg_dict['base64_image'] = "[IMAGE_DATA_EXCLUDED]"
            initial_memory.append(msg_dict)

    def _log_thinking_phase(self, should_act: bool) -> None:
        """Log the thinking phase results."""
        if not (hasattr(self, '_flow_context') and hasattr(self._flow_context, 'log_save')):
            return

        log_save = self._flow_context.log_save
        if not (hasattr(log_save, 'add_individual_question_message') and log_save.current_question):
            return

        iteration = getattr(self, '_flow_iteration_context', 1)
        step_num = self.current_step

        # Get the latest LLM response (should be assistant message)
        latest_response = None
        if self.memory.messages:
            for msg in reversed(self.memory.messages):
                if msg.role == "assistant":
                    latest_response = msg.content
                    break

        # Capture current memory state after thinking
        thinking_memory = []
        for msg in self.memory.messages:
            msg_dict = msg.to_dict()
            if 'base64_image' in msg_dict:
                msg_dict['base64_image'] = "[IMAGE_DATA_EXCLUDED]"
            thinking_memory.append(msg_dict)

        log_save.add_individual_question_message(
            role="system",
            content=f"[STEP_{step_num}_THINKING_COMPLETE]",
                        metadata={
                "agent_name": self.name,
                "step_number": step_num,
                "iteration": iteration,
                "phase": "thinking",
                "should_act": should_act,
                "llm_response": latest_response or "No response captured",
                "planned_actions": len(getattr(self, 'tool_calls', [])),
                "memory_size": len(self.memory.messages),
                "memory_state_after_thinking": thinking_memory
            }
        )

    def _log_action_phase(self, result: str) -> None:
        """Log the action phase results."""
        if not (hasattr(self, '_flow_context') and hasattr(self._flow_context, 'log_save')):
            return

        log_save = self._flow_context.log_save
        if not (hasattr(log_save, 'add_individual_question_message') and log_save.current_question):
            return

        iteration = getattr(self, '_flow_iteration_context', 1)
        step_num = self.current_step

        # Get tool execution results from memory
        tool_results = []
        for msg in reversed(self.memory.messages):
            if msg.role == "tool":
                tool_results.append({
                    "tool_name": msg.name,
                    "tool_call_id": msg.tool_call_id,
                    "result_length": len(msg.content) if msg.content else 0
                })

        # Capture current memory state after action
        action_memory = []
        for msg in self.memory.messages:
            msg_dict = msg.to_dict()
            if 'base64_image' in msg_dict:
                msg_dict['base64_image'] = "[IMAGE_DATA_EXCLUDED]"
            action_memory.append(msg_dict)

        log_save.add_individual_question_message(
            role="system",
            content=f"[STEP_{step_num}_ACTION_COMPLETE]",
                        metadata={
                "agent_name": self.name,
                "step_number": step_num,
                "iteration": iteration,
                "phase": "action",
                "tools_executed": len(tool_results),
                "tool_results": tool_results,
                "action_result_summary": result[:200] + "..." if len(result) > 200 else result,
                "memory_size": len(self.memory.messages),
                "memory_state_after_action": action_memory
            }
        )

    def _log_step_complete(self, result: str) -> None:
        """Log step completion with final memory state."""
        if not (hasattr(self, '_flow_context') and hasattr(self._flow_context, 'log_save')):
            return

        log_save = self._flow_context.log_save
        if not (hasattr(log_save, 'add_individual_question_message') and log_save.current_question):
            return

        iteration = getattr(self, '_flow_iteration_context', 1)
        step_num = self.current_step

        # Get final memory state
        final_memory = []
        for msg in self.memory.messages:
            msg_dict = msg.to_dict()
            if 'base64_image' in msg_dict:
                msg_dict['base64_image'] = "[IMAGE_DATA_EXCLUDED]"
            final_memory.append(msg_dict)

        log_save.add_individual_question_message(
            role=self.name,
            content=result,
                        metadata={
                "step_number": step_num,
                "iteration": iteration,
                "phase": "completion",
                "agent_state": self.state.value if self.state else "unknown",
                "memory_size": len(self.memory.messages),
                "final_memory_state": final_memory,
                "token_usage": self.get_token_usage()
            }
        )

    def _log_step_failure(self, error_message: str) -> None:
        """Log step failure with error details."""
        if not (hasattr(self, '_flow_context') and hasattr(self._flow_context, 'log_save')):
            return

        log_save = self._flow_context.log_save
        if not (hasattr(log_save, 'add_individual_question_message') and log_save.current_question):
            return

        iteration = getattr(self, '_flow_iteration_context', 1)
        step_num = self.current_step

        # Clean error message to avoid base64 data
        clean_error = error_message
        if 'base64' in error_message.lower():
            clean_error = error_message.replace('"base64_image":', '"base64_image": "[EXCLUDED]",')
            import re
            clean_error = re.sub(r'"[A-Za-z0-9+/]{30,}={0,2}"', '"[BASE64_DATA_EXCLUDED]"', clean_error)

        if len(clean_error) > 500:
            clean_error = clean_error[:500] + "... [TRUNCATED]"

        log_save.add_individual_question_message(
            role="system",
            content=f"[STEP_{step_num}_FAILED]",
                        metadata={
                "agent_name": self.name,
                "step_number": step_num,
                "iteration": iteration,
                "phase": "failure",
                "error_message": clean_error,
                "memory_size": len(self.memory.messages),
                "note": "Step failed before completion - step counter should not advance"
            }
        )

    def _log_agent_step_complete(self, result: str, thinking_summary: str = None, tool_summary: str = None) -> None:
        """Log agent step completion with all context included to reduce log verbosity."""
        if hasattr(self, '_flow_context') and hasattr(self._flow_context, 'log_save'):
            log_save = self._flow_context.log_save
            iteration = getattr(self, '_flow_iteration_context', 1)
            if isinstance(iteration, str):
                iteration = int(iteration.split('/')[0]) if '/' in iteration else 1

            # Get memory messages for inclusion in the step log
            memory_messages = []
            for msg in self.memory.messages:
                msg_dict = msg.to_dict()
                # Remove base64_image to prevent huge log files, just keep a marker
                if 'base64_image' in msg_dict:
                    msg_dict['base64_image'] = "[IMAGE_DATA_EXCLUDED]"
                memory_messages.append(msg_dict)

            # Build comprehensive step content
            step_content = result
            if thinking_summary:
                step_content = f"Thinking: {thinking_summary}\n\nResult: {result}"

            # Build comprehensive metadata
            metadata = {
                "step_number": self.current_step,
                "iteration": iteration,
                "agent_state": self.state.value if self.state else "unknown",
                "memory_size": len(self.memory.messages),
                "memory_messages": memory_messages,
                "token_usage": self.get_token_usage()
            }

            if tool_summary:
                metadata["tools_used"] = tool_summary

            # Use the streamlined logging interface
            if hasattr(log_save, 'add_individual_question_message') and log_save.current_question:
                log_save.add_individual_question_message(
                    role=self.name,
                    content=step_content,
                                        metadata=metadata
                )
