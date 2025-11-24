from typing import List, Optional
from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.llm import LLM
from app.prompt.translator import SYSTEM_PROMPT, NEXT_STEP_PROMPT, FINAL_STEP_PROMPT, FIRST_STEP_PROMPT
from app.utils.agent_utils import create_llm_setup_validator
from app.tool import ToolCollection
from app.logger import logger
from app.tool.ocr import OCR
from app.tool.read_table import ReadTable
from app.tool.terminate_and_output_caption import TerminateAndOutputCaption
from app.tool.smart_grid_caption import SmartGridCaption
from app.tool.think import Think
from app.schema import ToolChoice, TOOL_CHOICE_TYPE
class TranslatorAgent(ToolCallAgent):
    """
    A specialized VLM agent that provides simple image captioning.
    
    This agent analyzes visual content and produces JSON responses with
    the question and a global caption describing the image.
    """
    
    name: str = "translator"
    description: str = "Generates image captions with question and global description in JSON format"

    # Skip default LLM initialization since we handle it in model validator
    _skip_default_llm_init: bool = True
    
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    final_step_prompt: str = FINAL_STEP_PROMPT
    first_step_prompt: str = FIRST_STEP_PROMPT
    
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            SmartGridCaption(),
            TerminateAndOutputCaption(),
            OCR(),
            ReadTable(),
        )
    )
    #tool_choices: TOOL_CHOICE_TYPE = ToolChoice.REQUIRED  # Force tool usage
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate_and_output_caption"])
    max_steps: int = 3
    max_observe: int = 10000  # Sufficient for detailed smart_grid_caption analysis

    # Store JSON result from terminate_and_output_caption tool
    final_caption_json: Optional[str] = Field(default=None)

    # SIR Management - evolving structured intermediate representation
    current_sir: Optional[str] = Field(default=None)

    # Use utility function for LLM setup
    setup_translator_llm = create_llm_setup_validator("translator_api", "TranslatorAgent")

    def update_sir(self, new_information: str) -> None:
        """Update the current SIR with new information"""
        if not self.current_sir:
            self.current_sir = new_information
            logger.info(f"📝 SIR initialized: {len(new_information)} chars")
        else:
            self.current_sir = f"{self.current_sir}\n\n--- UPDATED SIR ---\n{new_information}"
            logger.info(f"📝 SIR updated: {len(self.current_sir)} chars total")

    def get_current_sir(self) -> str:
        """Get the current SIR, or empty string if none exists"""
        return self.current_sir or ""

    async def step(self) -> str:
        """Override step to track SIR updates"""
        try:
            # Call parent step method
            result = await super().step()

            # Extract any SIR updates from the step result
            # Look for SIR-related content in the assistant's response
            if self.memory.messages:
                latest_assistant_msg = None
                for msg in reversed(self.memory.messages):
                    if msg.role == "assistant":
                        latest_assistant_msg = msg.content
                        break

                # If assistant mentioned updating SIR or provided visual details, extract it
                if latest_assistant_msg and ("SIR" in latest_assistant_msg or "visual" in latest_assistant_msg.lower()):
                    # Extract meaningful visual information from the response
                    sir_update = self._extract_sir_from_response(latest_assistant_msg)
                    if sir_update:
                        self.update_sir(sir_update)

            return result
        except Exception as e:
            logger.error(f"Error in translator step: {e}")
            raise

    def _extract_sir_from_response(self, response: str) -> str:
        """Extract SIR-relevant information from assistant response"""
        # Simple heuristic: look for visual descriptions
        lines = response.split('\n')
        sir_content = []

        for line in lines:
            line = line.strip()
            # Skip tool calls and meta-commentary
            if (line and
                not line.startswith('<tool_call>') and
                not line.startswith('I ') and
                not line.startswith('Let me') and
                not line.startswith('Now') and
                'tool' not in line.lower()):
                sir_content.append(line)

        return '\n'.join(sir_content) if sir_content else ""

    # force_termination implementation inherited from BaseAgent
