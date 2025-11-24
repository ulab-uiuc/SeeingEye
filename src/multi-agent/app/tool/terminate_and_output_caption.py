import json
from app.tool.base import BaseTool


_TERMINATE_AND_OUTPUT_CAPTION_DESCRIPTION = """Terminate visual analysis and provide comprehensive image caption in JSON format.

Use this tool when:
- You have completed your visual analysis of the image
- You have gathered all necessary visual information through direct observation or tools
- You are ready to provide a comprehensive description of all visual elements
- You need to output the final caption in proper JSON format

Example call:
terminate_and_output_caption(
    global_caption="A food stall scene with two individuals, one wearing blue synthetic gloves handling food items. The gloves appear to be made of nitrile or similar disposable material, with a smooth, non-textured surface. Various food containers and utensils are visible in the background.",
    confidence="mid",
    summary_of_this_turn="1. **Initial Visual Analysis**: Observed a food stall scene with two individuals, one wearing blue gloves, likely made of a synthetic material. 2. **Text Extraction Attempt**: No text was detected from the image using the OCR tool. 3. **Feedback Evaluation**: The description lacked specific details about the gloves' material, prompting the need for more focused analysis. 4. **Patch Grid Creation**: A grid overlay was applied to the image to allow for detailed examination of specific regions. 5. **Focused Analysis Attempt**: I attempted to analyze specific patches to gather more detailed visual information about the gloves."
)

This signals that the visual analysis process should end with your formatted caption."""


class TerminateAndOutputCaption(BaseTool):
    name: str = "terminate_and_output_caption"
    description: str = _TERMINATE_AND_OUTPUT_CAPTION_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "global_caption": {
                "type": "string",
                "description": "A comprehensive description of ALL visual elements in sentence form or table form, including: text content, numerical values, table structures, objects, layouts, colors, spatial relationships, and any other visual information. Be factual and descriptive - do not infer anything not exists in the original image.",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "mid", "high"],
                "description": "Your confidence level in the completeness and accuracy of this global caption. 'low' = incomplete analysis or unclear image, 'mid' = good analysis with some limitations, 'high' = comprehensive and thorough analysis.",
            },
            "summary_of_this_turn": {
                "type": "string",
                "description": "A detailed summary of your analysis journey in this turn: (1) Start with your initial thoughts/observations/caption, (2) Describe which tools you used and their results, (3) Explain how tool results or feedback from text_only_reasoning agent helped you improve your observations, (4) Detail your step-by-step refinement process. Example format: '1. **Initial Visual Analysis**: Observed... 2. **Tool Usage**: Used OCR and found... 3. **Feedback Integration**: Reasoning agent requested more details about X, so I... 4. **Final Refinement**: Applied focused analysis to...'",
            }
        },
        "required": ["global_caption", "confidence", "summary_of_this_turn"],
    }

    async def execute(self, global_caption: str, confidence: str, summary_of_this_turn: str) -> str:
        """Terminate visual analysis and output enhanced JSON caption"""
        caption_data = {
            "global_caption": global_caption,
            "confidence": confidence,
            "summary_of_this_turn": summary_of_this_turn
        }
        return json.dumps(caption_data, indent=2, ensure_ascii=False)