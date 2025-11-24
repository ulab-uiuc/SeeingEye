from app.tool.base import BaseTool


_TERMINATE_AND_ANSWER_DESCRIPTION = """Terminate the reasoning process and provide a final answer when you have sufficient information from the SIR to confidently answer the question.

Use this tool when:
- The SIR contains all necessary visual details to answer the question
- You can identify the correct answer from the available options
- No additional information or refinement is needed from the translator agent
- Your answer matches one of the multiple choice options (if applicable)

IMPORTANT: For multiple choice questions, ensure your answer corresponds to one of the given options (A, B, C, D).

This signals that the iterative feedback loop should end with your final answer."""


class TerminateAndAnswer(BaseTool):
    name: str = "terminate_and_answer"
    description: str = _TERMINATE_AND_ANSWER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Your final answer to the question. Please include short answer only. For multiple choice, only include option",
            },
            "confidence": {
                "type": "string",
                "description": "Your confidence level in this answer.",
                "enum": ["high", "medium", "low"],
            },
            "reasoning": {
                "type": "string", 
                "description": "Brief explanation of how the SIR information led to this answer.",
            }
        },
        "required": ["answer", "confidence", "reasoning"],
    }

    async def execute(self, answer: str, confidence: str, reasoning: str) -> str:
        """Provide final answer and terminate the reasoning process"""
        return f"FINAL ANSWER: {answer}\n\nConfidence: {confidence}\n\nReasoning: {reasoning}\n\nThe reasoning process has been completed successfully."