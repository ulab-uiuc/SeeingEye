from typing import List
from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.vqa import SYSTEM_PROMPT, NEXT_STEP_PROMPT
from app.tool import TerminateAndAnswer, ToolCollection
from app.tool.python_execute import PythonExecute

class VQAAgent(ToolCallAgent):
    """
    A Visual Question Answering (VQA) agent that directly answers questions about images.
    
    This agent analyzes visual content and provides direct answers to questions,
    selecting from multiple choice options based on visual reasoning.
    """
    
    name: str = "vqa"
    description: str = "Answers questions about images by selecting from multiple choice options"
    
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    
    # Tools for VQA - includes Python execution for calculations
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            TerminateAndAnswer()  # Must provide answer when terminating
        )
    )

    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate_and_answer"])
    max_steps: int = 3
    max_observe: int = 2000  # Sufficient for VQA responses
    