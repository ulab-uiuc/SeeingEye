from app.tool.base import BaseTool, ToolResult


class Think(BaseTool):
    """A placeholder tool for reasoning and thinking when no other action is needed."""

    name: str = "think"
    description: str = "Use this tool when you need to think, reason, or analyze without performing any specific action. This allows you to continue your thought process and provide reasoning before deciding on next steps."
    parameters: dict = {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Your thoughts, analysis, or reasoning about the current situation. Describe what you're thinking about or what you've concluded.",
            },
        },
        "required": ["reasoning"],
    }

    async def execute(self, reasoning: str) -> ToolResult:
        """
        Execute the think tool - simply acknowledge the reasoning.

        Args:
            reasoning (str): The agent's thoughts or analysis.

        Returns:
            ToolResult: Confirmation that thinking occurred.
        """
        return ToolResult(
            output=f"Thinking complete. Reasoning: {reasoning}"
        )