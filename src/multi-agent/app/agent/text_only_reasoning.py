from typing import List
from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.llm import LLM
from app.prompt.text_only_reasoning import SYSTEM_PROMPT, NEXT_STEP_PROMPT, FINAL_STEP_PROMPT, FINAL_ITERATION_PROMPT
from app.utils.agent_utils import create_llm_setup_validator
from app.tool import TerminateAndAnswer, ToolCollection
from app.tool.python_execute import PythonExecute
from app.tool.terminate_and_ask_translator import TerminateAndAskTranslator
from app.tool.think import Think
from app.logger import logger
from app.schema import ToolChoice, TOOL_CHOICE_TYPE

class TextOnlyReasoningAgent(ToolCallAgent):
    """
    A text-only reasoning agent that analyzes and reasons about textual descriptions,
    answering questions based purely on provided text without any visual input.
    
    This agent is designed to work with textual descriptions of various scenarios
    and apply logical reasoning to answer questions based solely on the text.
    """
    
    name: str = "text_only_reasoning"
    description: str = "Analyzes and reasons about textual descriptions to answer questions using logical reasoning"

    # Skip default LLM initialization since we handle it in model validator
    _skip_default_llm_init: bool = True
    
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    final_step_prompt: str = FINAL_STEP_PROMPT
    final_iteration_prompt: str = FINAL_ITERATION_PROMPT
    
    # Tool collection for reasoning and answering
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            TerminateAndAnswer(),
            TerminateAndAskTranslator()
        )
    )
    #tool_choices: TOOL_CHOICE_TYPE = ToolChoice.REQUIRED  # Force tool usage
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate_and_answer", "terminate_and_ask_translator"])
    max_steps: int = 3
    max_observe: int = 2000  # Sufficient for text-based reasoning

    # Use utility function for LLM setup
    setup_reasoning_llm = create_llm_setup_validator("reasoning_api", "TextOnlyReasoningAgent")

    # force_termination implementation inherited from BaseAgent