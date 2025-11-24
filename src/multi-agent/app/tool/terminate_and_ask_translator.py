from app.tool.base import BaseTool


_TERMINATE_AND_ASK_TRANSLATOR_DESCRIPTION = """Terminate current reasoning step and request more specific visual observations from the translator.

Use this tool when:
- The current SIR (visual description) is insufficient for answering the question
- You need more specific details about certain parts of the image
- Important visual elements seem to be missing from the description
- You need clarification about spatial relationships, text content, or visual elements
- The translator's description lacks crucial information needed for reasoning

This signals that you need additional visual analysis before you can provide a final answer."""


class TerminateAndAskTranslator(BaseTool):
    name: str = "terminate_and_ask_translator"
    description: str = _TERMINATE_AND_ASK_TRANSLATOR_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "feedback": {
                "type": "string",
                "description": "Specific feedback about what additional visual information you need from the translator. Be precise about what's missing or unclear in the current description.",
            }
        },
        "required": ["feedback"],
    }

    async def execute(self, feedback: str) -> str:
        """Terminate reasoning step and provide feedback for translator"""
        return f"feedback: {feedback}"